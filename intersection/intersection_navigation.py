"""
Scenario: Intersection Behavior (Right-of-Way & Yielding)
===========================================================
Town03 — Complex signalised intersection.

Description:
  - ADS approaches a 4-way intersection under standard traffic light operation.
  - The ADS is tasked with executing a left turn across the intersection.
  - Oncoming vehicles are present, forcing the ADS to yield right-of-way.
  - The Traffic Manager dictates natural deceleration, queuing, and gap acceptance.

Usage:
    python intersection_behavior.py [--host HOST] [--port PORT] [--output OUTPUT]
"""

import argparse
import math
import random
import time
import carla

import numpy as np
from PIL import Image

# ── tuneable constants ────────────────────────────────────────────────────────
SEED              = 42
FIXED_DELTA_T     = 0.05          # s per tick (20 fps)
GIF_FPS           = 20
RECORD_SECONDS    = 15.0          # Longer duration to capture approach, wait, and turn
TRAFFIC_DENSITY   = 200           # Number of background vehicles around the intersection

IMG_W, IMG_H      = 1280, 720
CAMERA_FOV        = 90
OUTPUT_GIF        = "intersection_behavior.gif"


# ── helpers ───────────────────────────────────────────────────────────────────
def save_frame(image, frame_list):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))   # BGRA
    frame_list.append(Image.fromarray(array[:, :, [2, 1, 0]], mode="RGB"))


def dist2d(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def find_traffic_light_for_waypoint(world, waypoint):
    """Return the traffic light associated with the given waypoint, if present."""
    traffic_lights = world.get_actors().filter("traffic.traffic_light")
    for traffic_light in traffic_lights:
        for stop_wp in traffic_light.get_stop_waypoints():
            if stop_wp and dist2d(stop_wp.transform.location, waypoint.transform.location) < 8.0:
                return traffic_light
    return None


def try_spawn_actor_with_fallback(world, blueprint, transform, max_attempts=6):
    """Try a few nearby spawn transforms to avoid failures from blocked spawn points."""
    offsets = [0.0, 2.0, -2.0, 4.0, -4.0, 6.0]
    for offset in offsets[:max_attempts]:
        candidate = carla.Transform(
            carla.Location(
                x=transform.location.x + offset,
                y=transform.location.y,
                z=transform.location.z + 0.3,
            ),
            transform.rotation,
        )
        actor = world.try_spawn_actor(blueprint, candidate)
        if actor is not None:
            return actor
    raise RuntimeError(f"Failed to spawn {blueprint.id} after {max_attempts} attempts")


def spawn_intersection_traffic(world, tm, bp_lib, junction, actor_list, reference_location, max_vehicles=4):
    """Spawn extra traffic from the requested Town03 entry points while leaving the ADS path unchanged."""
    spawned = 0
    spawn_points = world.get_map().get_spawn_points()
    preferred_indices = [10, 11, 60]

    def try_spawn_from_waypoint(wp, lane_offset=0.0, speed=30.0):
        nonlocal spawned
        if spawned >= max_vehicles:
            return
        candidate_tf = wp.transform
        forward = candidate_tf.get_forward_vector()
        right = carla.Vector3D(x=-forward.y, y=forward.x, z=0.0)
        candidate_tf.location.x += right.x * lane_offset
        candidate_tf.location.y += right.y * lane_offset
        candidate_tf.location.z += 0.3
        bg_bp = random.choice(bp_lib.filter("vehicle.*.*"))
        if bg_bp.get_attribute("number_of_wheels").as_int() != 4:
            return
        try:
            bg_veh = world.try_spawn_actor(bg_bp, candidate_tf)
        except Exception:
            bg_veh = None
        if bg_veh:
            actor_list.append(bg_veh)
            bg_veh.set_simulate_physics(True)
            bg_veh.set_autopilot(True, tm.get_port())
            tm.set_desired_speed(bg_veh, speed)
            spawned += 1

    for idx in preferred_indices:
        if spawned >= max_vehicles:
            break
        if not 0 <= idx < len(spawn_points):
            continue
        sp = spawn_points[idx]
        wp = world.get_map().get_waypoint(sp.location)
        if not wp:
            continue
        for offset in [-3.0, 0.0, 3.0]:
            if spawned >= max_vehicles:
                break
            try_spawn_from_waypoint(wp, lane_offset=offset, speed=random.uniform(22.0, 38.0))

    # Fall back to the existing junction-side sampling if needed.
    if spawned < max_vehicles:
        junction_wps = junction.get_waypoints(carla.LaneType.Driving)
        for lane_wps in junction_wps:
            if spawned >= max_vehicles:
                break
            entry_wp = lane_wps[0]
            if dist2d(entry_wp.transform.location, reference_location) < 25.0:
                continue
            prev_wps = entry_wp.previous(20.0)
            if not prev_wps:
                continue
            try_spawn_from_waypoint(prev_wps[0], lane_offset=random.choice([-3.0, 3.0]), speed=random.uniform(22.0, 38.0))

    return spawned


def find_complex_intersection(world_map):
    """
    Locates a four-way intersection in Town03 suitable for complex turns.
    Returns the target junction actor/object and approach waypoints.
    """
    spawn_points = world_map.get_spawn_points()
    
    # In Town03, Spawn point 43 and surrounding areas provide excellent 4-way configurations
    # We will search for approach waypoints that enter a valid junction.
    approach_wps = []
    for sp in spawn_points:
        wp = world_map.get_waypoint(sp.location)
        # Look for waypoints close to entering a junction
        next_wps = wp.next(15.0)
        if next_wps and next_wps[0].is_junction:
            junction = next_wps[0].get_junction()
            # We want a junction with a high amount of intersecting lanes (4-way)
            if len(junction.get_waypoints(carla.LaneType.Driving)) > 8:
                return junction, wp
                
    # Direct fallback if programmatic search shifts across CARLA map revisions
    fallback_wp = world_map.get_waypoint(spawn_points[43].location)
    return fallback_wp.get_junction(), fallback_wp


# ── main ──────────────────────────────────────────────────────────────────────
def main(args):

    random.seed(SEED)
    np.random.seed(SEED)

    frames     = []
    actor_list = []

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    print("[INFO] Loading Town03…")
    world = client.load_world("Town03")
    original_settings = world.get_settings()

    try:
        # ── synchronous fixed-step ────────────────────────────────────────────
        settings = world.get_settings()
        settings.synchronous_mode    = True
        settings.fixed_delta_seconds = FIXED_DELTA_T
        settings.no_rendering_mode   = False
        world.apply_settings(settings)

        tm = client.get_trafficmanager(args.tm_port)
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(SEED)

        bp_lib    = world.get_blueprint_library()
        world_map = world.get_map()

        # ── Setup Intersection and Routes ─────────────────────────────────────
        junction, ads_approach_wp = find_complex_intersection(world_map)
        junction_centre = junction.bounding_box.location
        
        # Pull alternate approach paths belonging to the same junction for ambient traffic
        junction_wps = junction.get_waypoints(carla.LaneType.Driving)
        
        # ── Spawn ADS Vehicle ─────────────────────────────────────────────────
        ads_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        ads_bp.set_attribute("color", "30,144,255")  # Blue
        ads_bp.set_attribute("role_name", "ADS")
        
        traffic_light = find_traffic_light_for_waypoint(world, ads_approach_wp)
        ads_spawn_wp = ads_approach_wp
        if traffic_light:
            stop_waypoints = traffic_light.get_stop_waypoints()
            if stop_waypoints:
                stop_wp = min(stop_waypoints, key=lambda wp: dist2d(wp.transform.location, ads_approach_wp.transform.location))
                prev_wps = stop_wp.previous(6.0)
                if prev_wps:
                    ads_spawn_wp = prev_wps[0]

        ads_spawn_tf = ads_spawn_wp.transform
        ads_spawn_tf.location.z += 0.3

        ads = try_spawn_actor_with_fallback(world, ads_bp, ads_spawn_tf)
        actor_list.append(ads)
        
        ads.set_simulate_physics(True)
        ads.set_autopilot(True, tm.get_port())
        tm.set_desired_speed(ads, 35.0)  # Safe city approach speed in km/h
        tm.set_route(ads, ["Left"])

        # ── Spawn Oncoming and Ambient Vehicles ──────────────────────────────
        spawned_ambient = spawn_intersection_traffic(
            world,
            tm,
            bp_lib,
            junction,
            actor_list,
            ads_spawn_wp.transform.location,
            max_vehicles=TRAFFIC_DENSITY,
        )

        print(f"[INFO] Spawned ADS and {spawned_ambient} background conflict vehicles.")

        # ── Camera Setup (High-back Chase view) ───────────────────────────────
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMG_W))
        cam_bp.set_attribute("image_size_y", str(IMG_H))
        cam_bp.set_attribute("fov",          str(CAMERA_FOV))
        
        camera = world.spawn_actor(
            cam_bp,
            carla.Transform(
                carla.Location(x=-9.0, z=5.0),
                carla.Rotation(pitch=-15.0),
            ),
            attach_to=ads,
        )
        actor_list.append(camera)
        camera.listen(lambda img: save_frame(img, frames))

        # ── Simulation Loop ───────────────────────────────────────────────────
        total_ticks = int(RECORD_SECONDS / FIXED_DELTA_T)
        print(f"[INFO] Running Intersection Behavioral Test for {RECORD_SECONDS} seconds…")

        light_switched = False

        if traffic_light:
            traffic_light.set_state(carla.TrafficLightState.Red)
            traffic_light.freeze(True)
            print("[INFO] Traffic light initialized to RED")

        for tick_i in range(total_ticks):
            world.tick()
            
            t = tick_i * FIXED_DELTA_T
            ads_loc = ads.get_location()
            ads_vel = ads.get_velocity()
            ads_speed_kmh = 3.6 * math.sqrt(ads_vel.x**2 + ads_vel.y**2 + ads_vel.z**2)
            
            if traffic_light and not light_switched and t >= 2.0:
                traffic_light.freeze(False)
                traffic_light.set_state(carla.TrafficLightState.Green)
                traffic_light.freeze(True)
                light_switched = True
                print("[INFO] Traffic light switched to GREEN")

            if traffic_light:
                traffic_light_state = traffic_light.get_state()
            else:
                traffic_light_state = "No Light Detected"

            # Log system state every second
            if tick_i % int(1.0 / FIXED_DELTA_T) == 0:
                print(f"  t={t:5.1f}s | ADS Speed: {ads_speed_kmh:4.1f} km/h | Light State: {traffic_light_state} | Dist to Intersection: {dist2d(ads_loc, junction_centre):.1f}m")

        # ── Encode Output ─────────────────────────────────────────────────────
        if frames:
            print(f"[INFO] Compiling frames into output video → {args.output}…")
            frames[0].save(
                args.output,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                duration=int(1000 / GIF_FPS),
                loop=0,
                optimize=False,
            )
            print(f"[INFO] Done — {args.output}")

    finally:
        print("[INFO] Cleaning up scenario actors…")
        for actor in reversed(actor_list):
            try:
                if actor.is_alive:
                    actor.destroy()
            except Exception:
                pass
        world.apply_settings(original_settings)
        print("[INFO] World settings restored safely.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CARLA Verification — Intersection Turning and Yielding Scenarios"
    )
    parser.add_argument("--host",    default="127.0.0.1")
    parser.add_argument("--port",    default=2000, type=int)
    parser.add_argument("--tm_port", default=8000, type=int)
    parser.add_argument("--output",  default=OUTPUT_GIF)
    main(parser.parse_args())
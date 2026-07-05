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

import numpy as np
from PIL import Image

# ── tuneable constants ────────────────────────────────────────────────────────
SEED              = 42
FIXED_DELTA_T     = 0.05          # s per tick (20 fps)
GIF_FPS           = 20
RECORD_SECONDS    = 15.0          # Longer duration to capture approach, wait, and turn

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
    import carla

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
        
        # Move back roughly 45 meters from the junction entry to simulate the approach phase
        ads_spawn_wp = ads_approach_wp.previous(45.0)[0]
        ads = world.spawn_actor(ads_bp, ads_spawn_wp.transform)
        actor_list.append(ads)
        
        # Configure Traffic Manager autopilot behavior for ADS
        ads.set_simulate_physics(True)
        ads.set_autopilot(True, tm.get_port())
        
        # Force a left turn at the intersection to create a right-of-way conflict
        # TM handles choices via down-road waypoint intentions
        next_choices = ads_approach_wp.next(10.0)
        # Filter choices to select a path turning left across oncoming traffic
        tm.set_route(ads, ["Left"]) 
        tm.set_desired_speed(ads, 35.0)  # Safe city approach speed in km/h

        # ── Spawn Oncoming and Ambient Vehicles ──────────────────────────────
        # We find lanes within the intersection that are opposing the ADS direction
        spawned_ambient = 0
        for j_wp in junction_wps:
            # Entry points to the junction
            entry_wp = j_wp[0]
            # Avoid placing directly on top of the ADS approach lane
            if dist2d(entry_wp.transform.location, ads_spawn_wp.transform.location) > 20.0:
                if spawned_ambient >= 3: # Limit density to avoid total deadlock gridlocks
                    break
                    
                # Walk backwards along the other legs of the intersection to place ambient traffic
                bg_wps = entry_wp.previous(20.0)
                if bg_wps:
                    bg_bp = random.choice(bp_lib.filter("vehicle.*.*"))
                    # Don't pick bikes/trucks for stability here
                    if bg_bp.get_attribute("number_of_wheels").as_int() == 4:
                        bg_veh = world.try_spawn_actor(bg_bp, bg_wps[0].transform)
                        if bg_veh:
                            actor_list.append(bg_veh)
                            bg_veh.set_simulate_physics(True)
                            bg_veh.set_autopilot(True, tm.get_port())
                            # Make oncoming traffic aggressive enough to challenge the ADS gap choices
                            tm.set_desired_speed(bg_veh, 40.0)
                            tm.distance_to_leading_vehicle(bg_veh, 3.0)
                            spawned_ambient += 1

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

        for tick_i in range(total_ticks):
            world.tick()
            
            t = tick_i * FIXED_DELTA_T
            ads_loc = ads.get_location()
            ads_vel = ads.get_velocity()
            ads_speed_kmh = 3.6 * math.sqrt(ads_vel.x**2 + ads_vel.y**2 + ads_vel.z**2)
            
            # Check light state affecting ADS natively
            if ads.is_at_traffic_light():
                traffic_light = ads.get_traffic_light()
                state = traffic_light.get_state()
            else:
                state = "No Light Detected"

            # Log system state every second
            if tick_i % int(1.0 / FIXED_DELTA_T) == 0:
                print(f"  t={t:5.1f}s | ADS Speed: {ads_speed_kmh:4.1f} km/h | Light State: {state} | Dist to Intersection: {dist2d(ads_loc, junction_centre):.1f}m")

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
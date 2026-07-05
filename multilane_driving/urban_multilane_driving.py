"""
Scenario: Urban Multilane Driving
==================================
Town10HD — Busy multi-lane urban avenue.

Description:
  - ADS spawns on a high-density, multi-lane avenue in Town10.
  - The ADS is commanded to maintain a consistent cruise target speed (~50 km/h).
  - Ambient background traffic is spawned across multiple adjacent lanes.
  - The ADS must safely negotiate traffic density, manage vehicle following gaps,
    and adapt to lane constraints.

Usage:
    python urban_multilane_driving.py [--host HOST] [--port PORT] [--output OUTPUT]
"""

import argparse
import math
import random

import carla
import numpy as np
from PIL import Image

# ── tuneable constants ────────────────────────────────────────────────────────
SEED              = 101
FIXED_DELTA_T     = 0.05          # s per tick (20 fps)
GIF_FPS           = 20
RECORD_SECONDS    = 12.0          # Time window to witness multilane interactions

ADS_TARGET_SPEED_KMH = 50.0       # Enforced cruising speed for the ADS
TRAFFIC_DENSITY   = 100           # Number of surrounding background vehicles

IMG_W, IMG_H      = 1280, 720
CAMERA_FOV        = 95
OUTPUT_GIF        = "urban_multilane_driving.gif"


# ── helpers ───────────────────────────────────────────────────────────────────
def save_frame(image, frame_list):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))   # BGRA
    frame_list.append(Image.fromarray(array[:, :, [2, 1, 0]], mode="RGB"))


def dist2d(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def pick_multilane_spawn(world_map, spawn_points):
    """Choose a straight, multi-lane road segment for the ADS spawn."""
    best_point = None
    best_score = -1.0

    for sp in spawn_points:
        wp = world_map.get_waypoint(sp.location)
        if not wp:
            continue

        next_wps = wp.next(35.0)
        if not next_wps:
            continue

        forward = wp.transform.get_forward_vector()
        right = carla.Vector3D(x=-forward.y, y=forward.x, z=0.0)

        lane_count = 1
        left_wp = wp.get_left_lane()
        while left_wp and lane_count < 6:
            lane_count += 1
            left_wp = left_wp.get_left_lane()

        right_wp = wp.get_right_lane()
        while right_wp and lane_count < 6:
            lane_count += 1
            right_wp = right_wp.get_right_lane()

        score = lane_count * 4.0
        if not next_wps[0].is_junction:
            score += 6.0
        if wp.lane_id != 0:
            score -= 2.0
        if wp.road_id % 2 == 0:
            score += 1.0

        if score > best_score:
            best_score = score
            best_point = sp

    return best_point if best_point else spawn_points[0]


def spawn_surrounding_traffic(world, tm, bp_lib, actor_list, reference_wp, target_count):
    """Spawn vehicles on the main corridor and nearby connecting roads."""
    spawned = 0
    candidate_wps = []

    for dist in [15.0, 30.0, 45.0, 60.0, 80.0, 100.0]:
        next_wps = reference_wp.next(dist)
        if next_wps:
            candidate_wps.append(next_wps[0])

    for wp in candidate_wps:
        if spawned >= target_count:
            break

        for lane_offset in [-3.5, 0.0, 3.5]:
            if spawned >= target_count:
                break

            candidate_tf = wp.transform
            forward = wp.transform.get_forward_vector()
            right = carla.Vector3D(x=-forward.y, y=forward.x, z=0.0)
            candidate_tf.location.x += right.x * lane_offset
            candidate_tf.location.y += right.y * lane_offset
            candidate_tf.location.z += 0.3

            bg_bp = random.choice(bp_lib.filter("vehicle.*.*"))
            if bg_bp.get_attribute("number_of_wheels").as_int() != 4:
                continue

            bg_veh = world.try_spawn_actor(bg_bp, candidate_tf)
            if bg_veh:
                actor_list.append(bg_veh)
                bg_veh.set_simulate_physics(True)
                bg_veh.set_autopilot(True, tm.get_port())
                tm.set_desired_speed(bg_veh, random.uniform(28.0, 45.0))
                tm.auto_lane_change(bg_veh, True)
                spawned += 1

    # Add a few vehicles on nearby side roads by sampling junction-connected waypoints.
    if spawned < target_count:
        for _ in range(20):
            if spawned >= target_count:
                break

            sample_wp = random.choice(candidate_wps) if candidate_wps else reference_wp
            side_wps = sample_wp.get_junction().get_waypoints(carla.LaneType.Driving) if sample_wp.get_junction() else []
            if not side_wps:
                continue

            side_wp = random.choice(side_wps)[0]
            side_tf = side_wp.transform
            side_tf.location.z += 0.3
            bg_bp = random.choice(bp_lib.filter("vehicle.*.*"))
            if bg_bp.get_attribute("number_of_wheels").as_int() != 4:
                continue

            bg_veh = world.try_spawn_actor(bg_bp, side_tf)
            if bg_veh:
                actor_list.append(bg_veh)
                bg_veh.set_simulate_physics(True)
                bg_veh.set_autopilot(True, tm.get_port())
                tm.set_desired_speed(bg_veh, random.uniform(20.0, 35.0))
                spawned += 1

    return spawned


# ── main ──────────────────────────────────────────────────────────────────────
def main(args):

    random.seed(SEED)
    np.random.seed(SEED)

    frames     = []
    actor_list = []

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    print("[INFO] Loading Town10HD…")
    world = client.load_world("Town10HD")
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
        spawn_points = world_map.get_spawn_points()

        # ── Pick a Multi-Lane Stretch ─────────────────────────────────────────
        ads_spawn_point = pick_multilane_spawn(world_map, spawn_points)
        
        # ── Spawn ADS Vehicle ─────────────────────────────────────────────────
        ads_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        ads_bp.set_attribute("color", "30,144,255")  # Blue
        ads_bp.set_attribute("role_name", "ADS")
        
        ads = world.spawn_actor(ads_bp, ads_spawn_point)
        actor_list.append(ads)
        
        ads.set_simulate_physics(True)
        ads.set_autopilot(True, tm.get_port())
        
        # Configure Traffic Manager constraints for the ADS
        tm.set_desired_speed(ads, ADS_TARGET_SPEED_KMH)
        tm.distance_to_leading_vehicle(ads, 4.0)  # Safe following buffer
        tm.auto_lane_change(ads, True)            # Allow ADS to overtake slower traffic

        print(f"[INFO] ADS spawned at {ads_spawn_point.location}. Target speed: {ADS_TARGET_SPEED_KMH} km/h")

        # ── Populate Busy Multi-lane Traffic ──────────────────────────────────
        ads_wp = world_map.get_waypoint(ads_spawn_point.location)
        ambient_count = spawn_surrounding_traffic(
            world,
            tm,
            bp_lib,
            actor_list,
            ads_wp,
            TRAFFIC_DENSITY,
        )

        print(f"[INFO] Spawned {ambient_count} background vehicles to simulate a busy road environment.")

        # ── Camera Setup (Wide Third-Person Chase View) ───────────────────────
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMG_W))
        cam_bp.set_attribute("image_size_y", str(IMG_H))
        cam_bp.set_attribute("fov",          str(CAMERA_FOV))
        
        camera = world.spawn_actor(
            cam_bp,
            carla.Transform(
                carla.Location(x=-10.0, z=5.5),
                carla.Rotation(pitch=-14.0),
            ),
            attach_to=ads,
        )
        actor_list.append(camera)
        camera.listen(lambda img: save_frame(img, frames))

        # ── Warm-up ticks ─────────────────────────────────────────────────────
        for _ in range(20):
            world.tick()

        # ── Simulation Loop ───────────────────────────────────────────────────
        total_ticks = int(RECORD_SECONDS / FIXED_DELTA_T)
        print(f"[INFO] Running Urban Multilane Evaluation for {RECORD_SECONDS} seconds…")

        for tick_i in range(total_ticks):
            world.tick()
            
            t = tick_i * FIXED_DELTA_T
            ads_loc = ads.get_location()
            ads_vel = ads.get_velocity()
            ads_speed_kmh = 3.6 * math.sqrt(ads_vel.x**2 + ads_vel.y**2 + ads_vel.z**2)

            # Log execution progress every second
            if tick_i % int(1.0 / FIXED_DELTA_T) == 0:
                # Find current lane index (rough assessment from current waypoint)
                current_wp = world_map.get_waypoint(ads_loc)
                print(f"  t={t:5.1f}s | ADS Speed: {ads_speed_kmh:4.1f} km/h (Target: {ADS_TARGET_SPEED_KMH}) | Road ID: {current_wp.road_id} | Lane ID: {current_wp.lane_id}")

        # ── Encode Output ─────────────────────────────────────────────────────
        if frames:
            print(f"[INFO] Writing output animation frames → {args.output}…")
            frames[0].save(
                args.output,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                duration=int(1000 / GIF_FPS),
                loop=0,
                optimize=False,
            )
            print(f"[INFO] Success — Scenario written to {args.output}")

    finally:
        print("[INFO] Cleaning up scenario execution actors…")
        for actor in reversed(actor_list):
            try:
                if actor.is_alive:
                    actor.destroy()
            except Exception:
                pass
        world.apply_settings(original_settings)
        print("[INFO] World settings restored.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CARLA Verification — Urban Multilane Driving Scenarios"
    )
    parser.add_argument("--host",    default="127.0.0.1")
    parser.add_argument("--port",    default=2000, type=int)
    parser.add_argument("--tm_port", default=8000, type=int)
    parser.add_argument("--output",  default=OUTPUT_GIF)
    main(parser.parse_args())
"""
Multi-Vehicle Straight-Drive Scenario — CARLA Simulation
========================================================
Simulates an ADS vehicle tracking a double-lane street containing
7 other surrounding vehicles driving straight at varying speeds and spaces.
Records a GIF of the simulation.

Determinism: synchronous mode + fixed delta + fixed seeds.
Color fix:   CARLA raw_data is BGRA → reorder to RGB via [:,:,[2,1,0]].

Usage:
    python multi_car_straight.py [--host HOST] [--port PORT] [--output OUTPUT]
"""

import argparse
import math
import random

import numpy as np
from PIL import Image

# ── tuneable constants ────────────────────────────────────────────────────────
SEED            = 42
FIXED_DELTA_T   = 0.05        # seconds per tick (20 fps physics)
GIF_FPS         = 20
RECORD_SECONDS  = 9.0
ADS_SPEED_KMH   = 55.0
CAMERA_FOV      = 90
IMG_W, IMG_H    = 1280, 720
OUTPUT_GIF      = "multi_car_straight.gif"

# Configuration for the 7 other cars
NUM_OTHER_CARS  = 7


# ── helpers ───────────────────────────────────────────────────────────────────
def kmh_to_ms(v): return v / 3.6


def save_frame(image, frame_list):
    """BGRA → RGB (correct channel order) then store as PIL frame."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # BGRA
    frame_list.append(Image.fromarray(array[:, :, [2, 1, 0]], mode="RGB"))


def set_velocity(actor, speed_ms, direction=None):
    """Force actor velocity; direction defaults to actor's current forward."""
    import carla
    if direction is None:
        direction = actor.get_transform().get_forward_vector()
    actor.set_target_velocity(carla.Vector3D(
        direction.x * speed_ms,
        direction.y * speed_ms,
        direction.z * speed_ms,
    ))


def find_multilane_spawn(world_map):
    """
    Return (ads_wp, inner_lane_anchor, outer_lane_anchor) on a road that has
    genuine adjacent driving lanes (-1 and -2 or +1 and +2) with a straight run.
    """
    import carla

    all_wps = world_map.generate_waypoints(4.0)

    # Group by road+section
    from collections import defaultdict
    buckets = defaultdict(dict)
    for wp in all_wps:
        if wp.lane_type != carla.LaneType.Driving:
            continue
        key = (wp.road_id, wp.section_id)
        buckets[key][wp.lane_id] = wp

    candidates = []
    for (road_id, sec_id), lane_map in buckets.items():
        has_neg = (-1 in lane_map and -2 in lane_map)
        has_pos = ( 1 in lane_map and  2 in lane_map)
        if not (has_neg or has_pos):
            continue

        inner_lane_id = -1 if has_neg else  1
        outer_lane_id = -2 if has_neg else  2
        wp_inner = lane_map[inner_lane_id]
        wp_outer = lane_map[outer_lane_id]

        # Check straight run: walk 150 m ahead
        chain = [wp_inner]
        cur   = wp_inner
        for _ in range(40):
            nxt = cur.next(4.0)
            if not nxt:
                break
            cur = nxt[0]
            chain.append(cur)

        if len(chain) < 30:
            continue

        h0 = chain[0].transform.rotation.yaw
        h1 = chain[-1].transform.rotation.yaw
        dev = abs((h1 - h0 + 180) % 360 - 180)
        candidates.append((dev, wp_inner, wp_outer))

    if not candidates:
        raise RuntimeError("Could not find a multi-lane straight road.")

    # Pick straightest road
    candidates.sort(key=lambda x: x[0])
    _, wp_inner, wp_outer = candidates[0]

    print(f"[INFO] Selected road={wp_inner.road_id} section={wp_inner.section_id}")

    # Spawn ADS vehicle initially
    ads_wp_list = wp_outer.next(10.0)
    if not ads_wp_list:
        raise RuntimeError("No waypoint found for ADS anchor")
    ads_wp = ads_wp_list[0]

    return ads_wp, wp_inner, wp_outer


# ── main ──────────────────────────────────────────────────────────────────────
def main(args):
    import carla

    random.seed(SEED)
    np.random.seed(SEED)

    frames     = []
    actor_list = []

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    world = client.load_world("Town10HD_Opt")
    original_settings = world.get_settings()

    try:
        # ── synchronous fixed-step ────────────────────────────────────────────
        settings = world.get_settings()
        settings.synchronous_mode    = True
        settings.fixed_delta_seconds = FIXED_DELTA_T
        settings.no_rendering_mode   = False
        world.apply_settings(settings)

        tm = client.get_trafficmanager()
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(SEED)

        bp_lib    = world.get_blueprint_library()
        world_map = world.get_map()

        # ── find a proper multi-lane road ─────────────────────────────────────
        ads_wp, inner_anchor, outer_anchor = find_multilane_spawn(world_map)

        ads_spawn = ads_wp.transform
        ads_spawn.location.z += 0.3

        # ── spawn ADS vehicle ─────────────────────────────────────────────────
        ads_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        ads_bp.set_attribute("color", "30,144,255")
        ads_bp.set_attribute("role_name", "ADS")
        ads = world.spawn_actor(ads_bp, ads_spawn)
        actor_list.append(ads)
        ads.set_simulate_physics(True)

        # ── spawn 7 other vehicles surrounding the road ───────────────────────
        other_vehicles_data = []
        available_blueprints = bp_lib.filter("vehicle.*")
        
        # We drop a few problematic or giant vehicle models manually to keep it looking clean
        clean_blueprints = [b for b in available_blueprints if not any(
            x in b.id for x in ["colola", "carlacola", "firetruck", "ambulance", "sprinter"]
        )]

        # Procedurally define spacing & speeds for 7 vehicles relative to the ADS position
        # Using varied distances (some behind ADS, some ahead, staggered across both lanes)
        spawn_configurations = [
            {"distance": -20.0, "lane": "outer", "speed_kmh": 45.0},
            {"distance": 15.0,  "lane": "inner", "speed_kmh": 32.0},
            {"distance": 30.0,  "lane": "outer", "speed_kmh": 40.0},
            {"distance": 45.0,  "lane": "inner", "speed_kmh": 48.0},
            {"distance": 65.0,  "lane": "outer", "speed_kmh": 35.0},
            {"distance": 80.0,  "lane": "inner", "speed_kmh": 50.0},
            {"distance": 100.0, "lane": "outer", "speed_kmh": 38.0},
        ]

        print(f"[INFO] Spawning {NUM_OTHER_CARS} traffic vehicles driving straight...")
        for i, config in enumerate(spawn_configurations[:NUM_OTHER_CARS]):
            dist = config["distance"]
            
            # Find base tracking waypoint on the target lane archetype
            if dist >= 0:
                target_lane_wp = ads_wp.next(dist)[0]
            else:
                # To look backward for negative values
                target_lane_wp = ads_wp.previous(abs(dist))[0]
            
            # Snap to specific inner or outer lane
            target_lane_id = inner_anchor.lane_id if config["lane"] == "inner" else outer_anchor.lane_id
            
            actual_wp = world_map.get_waypoint(
                target_lane_wp.transform.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
            
            # Re-verify lane matching orientation
            left = actual_wp.get_left_lane()
            right = actual_wp.get_right_lane()
            if left and left.lane_type == carla.LaneType.Driving and left.lane_id == target_lane_id:
                actual_wp = left
            elif right and right.lane_type == carla.LaneType.Driving and right.lane_id == target_lane_id:
                actual_wp = right

            spawn_tf = actual_wp.transform
            spawn_tf.location.z += 0.4
            
            # Assign random models and unique muted colors
            veh_bp = random.choice(clean_blueprints)
            if veh_bp.has_attribute("color"):
                veh_bp.set_attribute("color", f"{random.randint(50,200)},{random.randint(50,200)},{random.randint(50,200)}")
            
            traffic_actor = world.try_spawn_actor(veh_bp, spawn_tf)
            if traffic_actor is not None:
                traffic_actor.set_simulate_physics(True)
                actor_list.append(traffic_actor)
                
                other_vehicles_data.append({
                    "actor": traffic_actor,
                    "speed_ms": kmh_to_ms(config["speed_kmh"])
                })
            else:
                print(f"[WARN] Failed to spawn vehicle {i} at distance {dist}m")

        # ── camera ────────────────────────────────────────────────────────────
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMG_W))
        cam_bp.set_attribute("image_size_y", str(IMG_H))
        cam_bp.set_attribute("fov",          str(CAMERA_FOV))
        cam_bp.set_attribute("sensor_tick",  "0.0")

        camera = world.spawn_actor(
            cam_bp,
            carla.Transform(carla.Location(x=-10.0, z=4.5),
                            carla.Rotation(pitch=-12.0)),
            attach_to=ads,
        )
        actor_list.append(camera)
        camera.listen(lambda img: save_frame(img, frames))

        # Warm-up ticks
        for _ in range(15):
            world.tick()

        # ── simulation loop ───────────────────────────────────────────────────
        ads_speed_ms = kmh_to_ms(ADS_SPEED_KMH)
        total_ticks  = int(RECORD_SECONDS / FIXED_DELTA_T)

        print(f"[INFO] Simulating {RECORD_SECONDS}s ({total_ticks} ticks)…")

        for tick_i in range(total_ticks):
            t = tick_i * FIXED_DELTA_T

            # ADS: Constant cruise speed forward
            set_velocity(ads, ads_speed_ms)

            # All other cars: Drive straight down their road direction at unique speeds
            for item in other_vehicles_data:
                actor = item["actor"]
                speed = item["speed_ms"]
                
                cur_wp = world_map.get_waypoint(
                    actor.get_location(),
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
                
                # Align straight along current road waypoint heading vector
                road_fwd = cur_wp.transform.get_forward_vector() if cur_wp else actor.get_transform().get_forward_vector()
                set_velocity(actor, speed, road_fwd)

            world.tick()

            # Logging status every 1 second
            if tick_i % int(1.0 / FIXED_DELTA_T) == 0:
                print(f"  t={t:5.1f}s | Active traffic vehicles moving: {len(other_vehicles_data)}")

        print(f"\n[INFO] Captured {len(frames)} frames.")

        # ── encode GIF ────────────────────────────────────────────────────────
        if not frames:
            print("[ERROR] No frames captured.")
            return

        print(f"[INFO] Writing GIF → {args.output} ({len(frames)} frames @ {GIF_FPS} fps)…")
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
        print("[INFO] Cleaning up…")
        for actor in reversed(actor_list):
            try:
                if actor.is_alive:
                    actor.destroy()
            except Exception:
                pass
        world.apply_settings(original_settings)
        print("[INFO] World settings restored.")


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CARLA Straight Multi-Vehicle Flow")
    parser.add_argument("--host",   default="127.0.0.1")
    parser.add_argument("--port",   default=2000, type=int)
    parser.add_argument("--output", default=OUTPUT_GIF)
    main(parser.parse_args())

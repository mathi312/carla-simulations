"""
TTC Lane-Cut Scenario — CARLA Simulation
=========================================
Simulates a car cutting into the ADS's lane 20 m ahead at 36 km/h
while the ADS travels at 55 km/h. Records a GIF of the simulation.

Fixes vs prior version
-----------------------
* Road selection: scans all roads for a segment that genuinely has
  two adjacent driving lanes (lane -1 AND lane -2) instead of
  hardcoding road 37 (which is single-lane in Town10).
* Merge steering: the other car follows its own lane waypoints before
  the merge, then steers laterally using the road's perpendicular
  vector — not toward a stale fixed point — so it never drives into
  a wall.  After merging it locks to the road forward direction
  queried from the map at its current position.

Determinism: synchronous mode + fixed delta + fixed seeds.
Color fix:   CARLA raw_data is BGRA → reorder to RGB via [:,:,[2,1,0]].

Usage:
    python ttc_lane_cut.py [--host HOST] [--port PORT] [--output OUTPUT]
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
SPAWN_GAP_M     = 30.0        # other car starts this far ahead of ADS
ADS_SPEED_KMH   = 55.0
OTHER_SPEED_KMH = 40.0
CAMERA_FOV      = 90
IMG_W, IMG_H    = 1280, 720
OUTPUT_GIF      = "ttc_lane_cut.gif"

# Merge timing
MERGE_START_S   = 2.0         # other car begins lane-change at t=2 s
MERGE_END_S     = 3.8         # merge complete by t=3.8 s


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


def vec2d_len(v):
    return math.sqrt(v.x ** 2 + v.y ** 2)


def normalize2d(v):
    import carla
    l = vec2d_len(v) + 1e-9
    return carla.Vector3D(v.x / l, v.y / l, 0.0)


def compute_ttc(ads_tf, ads_spd, other_tf, other_spd):
    """TTC for two vehicles; returns None if not closing."""
    dx   = other_tf.location.x - ads_tf.location.x
    dy   = other_tf.location.y - ads_tf.location.y
    dist = math.sqrt(dx*dx + dy*dy) + 1e-9
    ux, uy = dx/dist, dy/dist

    fwd_a = ads_tf.get_forward_vector()
    fwd_o = other_tf.get_forward_vector()
    v_rel = (ads_spd   * (fwd_a.x*ux + fwd_a.y*uy) -
             other_spd * (fwd_o.x*ux + fwd_o.y*uy))
    return (dist / v_rel) if v_rel > 0 else None


def find_multilane_spawn(world_map):
    """
    Return (ads_wp, other_wp) on a road that has genuine adjacent
    driving lanes (-1 and -2 or +1 and +2) with a straight run of
    at least 120 m ahead.

    Strategy: bucket all driving waypoints by (road_id, section_id),
    keep only buckets that contain both lane -1 and lane -2 (or +1/+2),
    then pick the bucket whose road has the longest straight run.
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
        # Need two negative-direction lanes OR two positive-direction lanes
        has_neg = (-1 in lane_map and -2 in lane_map)
        has_pos = ( 1 in lane_map and  2 in lane_map)
        if not (has_neg or has_pos):
            continue

        inner_lane_id = -1 if has_neg else  1
        outer_lane_id = -2 if has_neg else  2
        wp_inner = lane_map[inner_lane_id]
        wp_outer = lane_map[outer_lane_id]

        # Check straight run: walk 120 m ahead and measure heading deviation
        chain = [wp_inner]
        cur   = wp_inner
        for _ in range(30):          # 30 × 4 m = 120 m
            nxt = cur.next(4.0)
            if not nxt:
                break
            cur = nxt[0]
            chain.append(cur)

        if len(chain) < 20:          # need at least 80 m
            continue

        # Heading deviation over the run
        h0 = chain[0].transform.rotation.yaw
        h1 = chain[-1].transform.rotation.yaw
        dev = abs((h1 - h0 + 180) % 360 - 180)
        candidates.append((dev, wp_inner, wp_outer))

    if not candidates:
        raise RuntimeError(
            "Could not find a multi-lane straight road. "
            "Check that Town10HD_Opt is loaded."
        )

    # Pick straightest road
    candidates.sort(key=lambda x: x[0])
    _, wp_inner, wp_outer = candidates[0]

    print(f"[INFO] Selected road={wp_inner.road_id} section={wp_inner.section_id} "
          f"lane_ADS={wp_outer.lane_id} lane_other={wp_inner.lane_id} "
          f"heading_dev={candidates[0][0]:.1f}°")

    # ADS spawns on the OUTER lane (further from centre).
    # Other car spawns SPAWN_GAP_M ahead on the INNER lane (closer to centre),
    # then merges outward into the ADS lane — i.e. cuts in from the right.
    ads_wp_list = wp_outer.next(10.0)
    if not ads_wp_list:
        raise RuntimeError("No waypoint 10 m ahead of outer-lane anchor")
    ads_wp = ads_wp_list[0]

    other_ahead = ads_wp.next(SPAWN_GAP_M)
    if not other_ahead:
        raise RuntimeError("No waypoint ahead for other car spawn")
    other_ahead_wp = other_ahead[0]

    # Snap other car to the inner lane at the same s-position
    other_wp = world_map.get_waypoint(
        other_ahead_wp.transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    # Walk to inner lane
    left  = other_wp.get_left_lane()
    right = other_wp.get_right_lane()
    if left  and left.lane_type  == carla.LaneType.Driving and left.lane_id  == wp_inner.lane_id:
        other_wp = left
    elif right and right.lane_type == carla.LaneType.Driving and right.lane_id == wp_inner.lane_id:
        other_wp = right

    return ads_wp, other_wp


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
        ads_wp, other_wp = find_multilane_spawn(world_map)

        ads_spawn       = ads_wp.transform
        ads_spawn.location.z += 0.3

        other_spawn     = other_wp.transform
        other_spawn.location.z += 0.3

        # Pre-compute the road's lateral (perpendicular) direction at the
        # other car's spawn.  We use this during the merge to shift the
        # other car sideways into the ADS lane — keeping it aligned with
        # the road rather than aiming at a fixed world point.
        other_fwd_world = other_wp.transform.get_forward_vector()
        # Left-perpendicular (cross product with Z-up):  (-fy, fx, 0)
        perp_to_ads = carla.Vector3D(
            -other_fwd_world.y,
             other_fwd_world.x,
             0.0,
        )
        # Decide sign: should point from other lane toward ADS lane
        ads_loc_ref = ads_wp.transform.location
        oth_loc_ref = other_wp.transform.location
        diff = carla.Vector3D(
            ads_loc_ref.x - oth_loc_ref.x,
            ads_loc_ref.y - oth_loc_ref.y,
            0.0,
        )
        if (perp_to_ads.x * diff.x + perp_to_ads.y * diff.y) < 0:
            perp_to_ads = carla.Vector3D(-perp_to_ads.x, -perp_to_ads.y, 0.0)
        perp_to_ads = normalize2d(perp_to_ads)

        # ── spawn vehicles ────────────────────────────────────────────────────
        ads_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        ads_bp.set_attribute("color",     "30,144,255")
        ads_bp.set_attribute("role_name", "ADS")

        other_bp = bp_lib.filter("vehicle.dodge.charger_2020")[0]
        other_bp.set_attribute("color",     "220,50,50")
        other_bp.set_attribute("role_name", "other")

        ads   = world.spawn_actor(ads_bp,   ads_spawn)
        other = world.spawn_actor(other_bp, other_spawn)
        actor_list.extend([ads, other])

        ads.set_simulate_physics(True)
        other.set_simulate_physics(True)

        # ── camera ────────────────────────────────────────────────────────────
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMG_W))
        cam_bp.set_attribute("image_size_y", str(IMG_H))
        cam_bp.set_attribute("fov",          str(CAMERA_FOV))
        cam_bp.set_attribute("sensor_tick",  "0.0")

        camera = world.spawn_actor(
            cam_bp,
            carla.Transform(carla.Location(x=-8.0, z=3.5),
                            carla.Rotation(pitch=-10.0)),
            attach_to=ads,
        )
        actor_list.append(camera)
        camera.listen(lambda img: save_frame(img, frames))

        # Warm-up
        for _ in range(15):
            world.tick()

        # ── simulation loop ───────────────────────────────────────────────────
        ads_speed_ms   = kmh_to_ms(ADS_SPEED_KMH)
        other_speed_ms = kmh_to_ms(OTHER_SPEED_KMH)
        total_ticks    = int(RECORD_SECONDS / FIXED_DELTA_T)
        merge_start_tk = int(MERGE_START_S   / FIXED_DELTA_T)
        merge_end_tk   = int(MERGE_END_S     / FIXED_DELTA_T)

        # Lane width for lateral blend (typical CARLA lane = 3.5 m)
        lane_width = other_wp.lane_width if other_wp.lane_width else 3.5

        print(f"[INFO] Simulating {RECORD_SECONDS}s ({total_ticks} ticks)…")
        print(f"[INFO] Merge window: t={MERGE_START_S}s → t={MERGE_END_S}s")

        for tick_i in range(total_ticks):
            t = tick_i * FIXED_DELTA_T

            # ── ADS: constant cruise speed ────────────────────────────────────
            set_velocity(ads, ads_speed_ms)

            # ── Other car movement ────────────────────────────────────────────
            if tick_i < merge_start_tk:
                # Phase 1: travel straight in own lane
                # Direction from the map at current position keeps it road-aligned
                cur_wp = world_map.get_waypoint(
                    other.get_location(),
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
                road_fwd = cur_wp.transform.get_forward_vector() if cur_wp else other_fwd_world
                set_velocity(other, other_speed_ms, road_fwd)

            elif tick_i < merge_end_tk:
                # Phase 2: blend forward direction with lateral shift direction
                # alpha goes 0→1 over the merge window
                alpha = (tick_i - merge_start_tk) / (merge_end_tk - merge_start_tk)

                # Road forward at current position
                cur_wp = world_map.get_waypoint(
                    other.get_location(),
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
                road_fwd = cur_wp.transform.get_forward_vector() if cur_wp else other_fwd_world

                # Blend: mostly forward, increasingly lateral
                # At alpha=0: pure forward.  At alpha=0.5: peak lateral blend (~35°).
                # At alpha=1: mostly forward again (just completing the lane change).
                lateral_weight = math.sin(alpha * math.pi) * 0.5   # peak 0.5 at mid-merge
                merge_dir = carla.Vector3D(
                    road_fwd.x + perp_to_ads.x * lateral_weight,
                    road_fwd.y + perp_to_ads.y * lateral_weight,
                    0.0,
                )
                set_velocity(other, other_speed_ms, normalize2d(merge_dir))

            else:
                # Phase 3: merged — follow road forward in the ADS lane
                cur_wp = world_map.get_waypoint(
                    other.get_location(),
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
                if cur_wp:
                    road_fwd = cur_wp.transform.get_forward_vector()
                else:
                    road_fwd = ads.get_transform().get_forward_vector()
                set_velocity(other, other_speed_ms, road_fwd)

            world.tick()

            # Console TTC every second
            if tick_i % int(1.0 / FIXED_DELTA_T) == 0:
                ttc = compute_ttc(
                    ads.get_transform(),   ads_speed_ms,
                    other.get_transform(), other_speed_ms,
                )
                gap = ads.get_location().distance(other.get_location())
                ttc_str = f"{ttc:.2f} s" if ttc is not None else "—"
                phase = ("cruise" if tick_i < merge_start_tk else
                         "MERGING" if tick_i < merge_end_tk else "merged")
                print(f"  t={t:5.1f}s  [{phase:8s}]  "
                      f"TTC={ttc_str:8s}  gap={gap:.1f} m")

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
    parser = argparse.ArgumentParser(description="CARLA TTC lane-cut scenario")
    parser.add_argument("--host",   default="127.0.0.1")
    parser.add_argument("--port",   default=2000, type=int)
    parser.add_argument("--output", default=OUTPUT_GIF)
    main(parser.parse_args())

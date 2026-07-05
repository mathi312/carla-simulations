"""
Conflict Severity (CS) — Pedestrian Crossing Scenario
======================================================
Town03 — pedestrian crossing scenario.

Scenario:
  - ADS drives toward an intersection on green at ~50 km/h
  - A pedestrian starts crossing at a designated crossing (zebra crossing)
  - ADS detects the pedestrian and brakes at maximum deceleration
  - A collision may occur; residual speed at impact is recorded
  - CS is computed as:
        CS = v_residual × (m_ADS / (m_ADS + m_pedestrian))
  - With m_ADS=1800 kg, m_pedestrian=80 kg, v_residual≈X m/s → CS varies

Why Town03:
  Town03 has marked pedestrian crossings (zebra crossings) integrated into
  traffic intersections. We find a suitable crossing and spawn a pedestrian
  to cross while the ADS approaches.

Determinism: synchronous mode + fixed delta + fixed seeds.
Color fix:   CARLA raw_data is BGRA → reorder to RGB via [:,:,[2,1,0]].

Usage:
    python cs_pedestrian_crossing.py [--host HOST] [--port PORT] [--output OUTPUT]
"""

import argparse
import math
import random

import numpy as np
from PIL import Image

# ── tuneable constants ────────────────────────────────────────────────────────
SEED              = 42
FIXED_DELTA_T     = 0.05          # s per tick  (20 fps physics)
GIF_FPS           = 20
RECORD_SECONDS    = 10.0

ADS_SPEED_KMH     = 45.0          # cruise speed approaching intersection
ADS_BRAKE_DECEL   = 8.0           # m/s²  maximum emergency braking

# Pedestrian masses (kg)
ADS_MASS_KG       = 1800.0
PEDESTRIAN_MASS_KG = 80.0

# ADS detects the pedestrian when they are this many metres away
DETECTION_DIST_M  = 25.0

# How far before the crossing the ADS spawns
ADS_APPROACH_M    = 60.0

# Pedestrian crossing speed (m/s)
PEDESTRIAN_SPEED_MS = 1.5

IMG_W, IMG_H      = 1280, 720
CAMERA_FOV        = 90
OUTPUT_GIF        = "cs_pedestrian_crossing.gif"


# ── helpers ───────────────────────────────────────────────────────────────────
def kmh_to_ms(v): return v / 3.6
def ms_to_kmh(v): return v * 3.6


def save_frame(image, frame_list):
    """
    CARLA → PIL with correct channel order.
    CARLA stores BGRA; naive [:,:,:3] yields BGR which PIL reads as RGB
    → red/blue swap.  Explicit channel reorder fixes this.
    """
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))   # BGRA
    frame_list.append(Image.fromarray(array[:, :, [2, 1, 0]], mode="RGB"))


def set_velocity(actor, speed_ms, direction=None):
    """Force actor velocity; direction defaults to actor's current forward."""
    import carla
    if direction is None:
        direction = actor.get_transform().get_forward_vector()
    actor.set_target_velocity(carla.Vector3D(
        x=direction.x * speed_ms,
        y=direction.y * speed_ms,
        z=direction.z * speed_ms,
    ))


def dist2d(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def normalize2d(v):
    import carla
    l = math.sqrt(v.x ** 2 + v.y ** 2) + 1e-9
    return carla.Vector3D(v.x / l, v.y / l, 0.0)


def compute_cs(v_residual_ms, mass_ads, mass_pedestrian):
    """
    CS = v_residual × (m_ADS / (m_ADS + m_pedestrian))
    Adapted for pedestrian collisions.
    """
    mass_ratio = mass_ads / (mass_ads + mass_pedestrian)
    return v_residual_ms * mass_ratio


def find_crossing_and_approach(world, world_map):
    """
    Find a pedestrian crossing in Town03 and suitable approach waypoint for ADS.

    Strategy:
      1. Iterate through all waypoints looking for crossings.
      2. For each crossing, find a road segment that approaches it perpendicularly.
      3. Return the crossing location, the ADS approach waypoint, and the
         pedestrian's crossing path.

    Returns (crossing_location, ads_approach_wp, ped_start_pos, ped_end_pos)
    """
    import carla

    all_wps = world_map.generate_waypoints(distance=2.0)
    
    # Find crossings by looking for junction waypoints adjacent to roads
    crossing_candidates = []
    
    for wp in all_wps:
        # Check if this waypoint is near a junction (potential crossing area)
        if wp.is_junction:
            # Look at nearby waypoints to find the road that approaches this junction
            next_wps = wp.next(5.0)
            prev_wps = wp.previous(5.0)
            
            if next_wps and prev_wps:
                next_wp = next_wps[0]
                prev_wp = prev_wps[0]
                
                # Calculate if roads are perpendicular enough
                curr_yaw = wp.transform.rotation.yaw
                prev_yaw = prev_wp.transform.rotation.yaw
                
                yaw_diff = abs((curr_yaw - prev_yaw + 180) % 360 - 180)
                
                # We want roughly perpendicular approach (around 90 degrees)
                if 45 < yaw_diff < 135:
                    crossing_candidates.append((wp, prev_wp))
    
    if not crossing_candidates:
        print("[WARNING] No suitable crossings found; using first junction area")
        # Fallback: use any junction waypoint
        for wp in all_wps:
            if wp.is_junction:
                prev_wps = wp.previous(5.0)
                if prev_wps:
                    crossing_candidates.append((wp, prev_wps[0]))
                    break
    
    if not crossing_candidates:
        raise RuntimeError("Could not find any crossing waypoint")
    
    # Pick the first suitable candidate
    crossing_wp, ads_approach_wp = crossing_candidates[0]
    
    # Move ADS approach waypoint back
    ads_prev = ads_approach_wp.previous(ADS_APPROACH_M)
    if ads_prev:
        ads_approach_wp = ads_prev[0]
    
    # Pedestrian crossing: compute start and end positions perpendicular to road
    crossing_loc = crossing_wp.transform.location
    
    # Get the perpendicular direction (90 degrees to the road)
    road_direction = normalize2d(ads_approach_wp.transform.get_forward_vector())
    # Perpendicular = rotate 90 degrees
    ped_direction = carla.Vector3D(-road_direction.y, road_direction.x, 0.0)
    
    # Pedestrian starts on one side of the road
    ped_start_offset = 4.0  # offset from road center
    ped_end_offset = -4.0
    
    ped_start_pos = carla.Location(
        x=crossing_loc.x + ped_direction.x * ped_start_offset,
        y=crossing_loc.y + ped_direction.y * ped_start_offset,
        z=crossing_loc.z,
    )
    
    ped_end_pos = carla.Location(
        x=crossing_loc.x + ped_direction.x * ped_end_offset,
        y=crossing_loc.y + ped_direction.y * ped_end_offset,
        z=crossing_loc.z,
    )
    
    print(f"[INFO] Crossing location: ({crossing_loc.x:.1f}, {crossing_loc.y:.1f})")
    print(f"[INFO] ADS approach road yaw: {ads_approach_wp.transform.rotation.yaw:.0f}°")
    
    return crossing_loc, ads_approach_wp, ped_start_pos, ped_end_pos


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

        tm = client.get_trafficmanager()
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(SEED)

        bp_lib    = world.get_blueprint_library()
        world_map = world.get_map()

        # ── find crossing ─────────────────────────────────────────────────────
        crossing_loc, ads_approach_wp, ped_start_pos, ped_end_pos = \
            find_crossing_and_approach(world, world_map)

        # ADS spawns back along its approach road
        ads_spawn_wp = ads_approach_wp
        ads_spawn_tf = ads_spawn_wp.transform
        ads_spawn_tf.location.z += 0.3

        # Pre-compute ADS direction
        ads_dir = normalize2d(ads_spawn_wp.transform.get_forward_vector())

        # ── spawn ADS vehicle ─────────────────────────────────────────────────
        ads_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        ads_bp.set_attribute("color",     "30,144,255")   # blue
        ads_bp.set_attribute("role_name", "ADS")

        ads = world.spawn_actor(ads_bp, ads_spawn_tf)
        actor_list.append(ads)
        ads.set_simulate_physics(True)

        # Set vehicle mass
        ads_physics       = ads.get_physics_control()
        ads_physics.mass  = ADS_MASS_KG
        ads.apply_physics_control(ads_physics)

        # ── spawn pedestrian ──────────────────────────────────────────────────
        # Find a pedestrian walker blueprint
        walker_bps = bp_lib.filter("walker.pedestrian.*")
        if not walker_bps:
            print("[ERROR] No pedestrian blueprints found")
            return

        walker_bp = random.choice(walker_bps)
        
        # Spawn pedestrian at start position
        ped_spawn_tf = carla.Transform(ped_start_pos)
        ped_spawn_tf.location.z += 0.5  # Adjust height
        
        pedestrian = world.spawn_actor(walker_bp, ped_spawn_tf)
        actor_list.append(pedestrian)
        
        print(f"[INFO] Spawned pedestrian: {walker_bp.id}")

        # ── compute pedestrian crossing vector ─────────────────────────────────
        ped_cross_vec = carla.Location(
            x=ped_end_pos.x - ped_start_pos.x,
            y=ped_end_pos.y - ped_start_pos.y,
            z=0.0,
        )
        ped_cross_dist = math.sqrt(ped_cross_vec.x ** 2 + ped_cross_vec.y ** 2)
        if ped_cross_dist > 0:
            ped_cross_dir = carla.Vector3D(
                ped_cross_vec.x / ped_cross_dist,
                ped_cross_vec.y / ped_cross_dist,
                0.0
            )
        else:
            ped_cross_dir = carla.Vector3D(1, 0, 0)

        # ── camera — chase behind ADS ────────────────────────────────────────
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMG_W))
        cam_bp.set_attribute("image_size_y", str(IMG_H))
        cam_bp.set_attribute("fov",          str(CAMERA_FOV))
        cam_bp.set_attribute("sensor_tick",  "0.0")

        camera = world.spawn_actor(
            cam_bp,
            carla.Transform(
                carla.Location(x=-8.0, z=4.5),
                carla.Rotation(pitch=-12.0),
            ),
            attach_to=ads,
        )
        actor_list.append(camera)
        camera.listen(lambda img: save_frame(img, frames))

        # ── warm-up ticks ─────────────────────────────────────────────────────
        for _ in range(20):
            world.tick()

        # ── simulation state ──────────────────────────────────────────────────
        ads_speed_ms   = kmh_to_ms(ADS_SPEED_KMH)
        braking        = False
        collision_tick = None
        v_at_collision = None
        cs_computed    = False
        ped_started_crossing = False

        total_ticks = int(RECORD_SECONDS / FIXED_DELTA_T)
        print(f"[INFO] Simulating {RECORD_SECONDS}s ({total_ticks} ticks)…")
        print(f"[INFO] ADS mass={ADS_MASS_KG:.0f} kg  |  "
              f"Pedestrian mass={PEDESTRIAN_MASS_KG:.0f} kg")

        for tick_i in range(total_ticks):
            t = tick_i * FIXED_DELTA_T

            ads_loc = ads.get_location()
            ped_loc = pedestrian.get_location()
            gap = dist2d(ads_loc, ped_loc)
            dist_to_crossing = dist2d(ads_loc, crossing_loc)

            # ── pedestrian crossing logic ─────────────────────────────────────
            # Start crossing when ADS is about 30m away
            if not ped_started_crossing and dist_to_crossing < 35.0:
                ped_started_crossing = True
                print(f"  [t={t:.2f}s] Pedestrian starts crossing…")

            if ped_started_crossing:
                # Move pedestrian across the road
                set_velocity(pedestrian, PEDESTRIAN_SPEED_MS, ped_cross_dir)
            else:
                # Pedestrian waiting
                set_velocity(pedestrian, 0.0, ped_cross_dir)

            # ── detect conflict: pedestrian in collision path ──────────────────
            if not braking and gap < DETECTION_DIST_M and ped_started_crossing:
                braking = True
                print(f"  [t={t:.2f}s] Pedestrian detected — ADS braking hard  "
                      f"(gap={gap:.1f}m, ADS speed={ms_to_kmh(ads_speed_ms):.1f} km/h)")

            # ── ADS speed update ──────────────────────────────────────────────
            if braking:
                ads_speed_ms = max(0.0, ads_speed_ms - ADS_BRAKE_DECEL * FIXED_DELTA_T)

            # ── detect collision (within ~2 m for pedestrian) ──────────────────
            if collision_tick is None and gap < 2.0 and tick_i > 10:
                collision_tick = tick_i
                v_at_collision = ads_speed_ms
                cs = compute_cs(v_at_collision, ADS_MASS_KG, PEDESTRIAN_MASS_KG)
                print(f"\n  *** COLLISION at t={t:.2f}s ***")
                print(f"      ADS residual speed : {v_at_collision:.2f} m/s  "
                      f"({ms_to_kmh(v_at_collision):.1f} km/h)")
                print(f"      Mass ratio (ADS)   : {ADS_MASS_KG:.0f} / "
                      f"({ADS_MASS_KG:.0f} + {PEDESTRIAN_MASS_KG:.0f}) = "
                      f"{ADS_MASS_KG/(ADS_MASS_KG+PEDESTRIAN_MASS_KG):.3f}")
                print(f"      CS = {v_at_collision:.2f} × "
                      f"{ADS_MASS_KG/(ADS_MASS_KG+PEDESTRIAN_MASS_KG):.3f} = {cs:.2f}")
                cs_computed = True

            # ── apply velocities ──────────────────────────────────────────────
            # After collision let physics engine take over
            if collision_tick is None or tick_i < collision_tick + 5:
                set_velocity(ads, ads_speed_ms, ads_dir)

            world.tick()

            # Console every second
            if tick_i % int(1.0 / FIXED_DELTA_T) == 0:
                print(f"  t={t:5.1f}s  "
                      f"ADS {ms_to_kmh(ads_speed_ms):5.1f} km/h  "
                      f"ped_gap={gap:5.1f} m  "
                      f"dist_to_crossing={dist_to_crossing:5.1f} m  "
                      f"{'[CROSSING]' if ped_started_crossing else '[WAITING]':12s}  "
                      f"{'[BRAKING]' if braking else '':9s}  "
                      f"{'[COLLISION]' if collision_tick and tick_i >= collision_tick else ''}")

        # ── CS summary ────────────────────────────────────────────────────────
        print("\n── CS Summary (Pedestrian) ──────────────────────────────────")
        if cs_computed:
            cs_final = compute_cs(v_at_collision, ADS_MASS_KG, PEDESTRIAN_MASS_KG)
            ratio    = ADS_MASS_KG / (ADS_MASS_KG + PEDESTRIAN_MASS_KG)
            print(f"  v_residual          = {v_at_collision:.2f} m/s")
            print(f"  m_ADS               = {ADS_MASS_KG:.0f} kg")
            print(f"  m_pedestrian        = {PEDESTRIAN_MASS_KG:.0f} kg")
            print(f"  mass ratio          = {ADS_MASS_KG:.0f} / "
                  f"{ADS_MASS_KG + PEDESTRIAN_MASS_KG:.0f} = {ratio:.3f}")
            print(f"  CS = {v_at_collision:.2f} × {ratio:.3f} = {cs_final:.2f}")
        else:
            print("  No collision detected — pedestrian successfully crossed.")
            print("  (You can try reducing ADS_APPROACH_M or increasing PEDESTRIAN_SPEED_MS)")
        print("─────────────────────────────────────────────────────────────\n")

        # ── encode GIF ────────────────────────────────────────────────────────
        if not frames:
            print("[ERROR] No frames captured.")
            return

        print(f"[INFO] Writing GIF → {args.output}  "
              f"({len(frames)} frames @ {GIF_FPS} fps)…")
        frames[0].save(
            args.output,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / GIF_FPS),
            loop=0,
            optimize=False,   # no palette reduction — preserve CARLA colours
        )
        print(f"[INFO] Done — {args.output}")

    finally:
        print("[INFO] Cleaning up actors…")
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
    parser = argparse.ArgumentParser(
        description="CARLA Conflict Severity — pedestrian crossing scenario"
    )
    parser.add_argument("--host",   default="127.0.0.1")
    parser.add_argument("--port",   default=2000, type=int)
    parser.add_argument("--output", default=OUTPUT_GIF)
    main(parser.parse_args())

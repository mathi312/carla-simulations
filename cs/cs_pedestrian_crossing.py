"""
Conflict Severity (CS) — Pedestrian Mid-Block Crossing Scenario
==============================================================
Town03 — Straight road segment (no intersection).

Scenario:
  - ADS drives down a straight road at ~45 km/h
  - A pedestrian walks out from the side of the road to cross it
  - ADS detects the conflict and brakes at maximum deceleration
  - A collision still occurs; residual speed at impact is recorded
  - CS is computed as:
        CS = v_residual × (m_ADS / (m_ADS + m_pedestrian))
  - With m_ADS=1800 kg, m_pedestrian=75 kg, v_residual ≈ 4 m/s → CS ≈ 3.84

Usage:
    python cs_pedestrian.py [--host HOST] [--port PORT] [--output OUTPUT]
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
RECORD_SECONDS    = 8.0

ADS_SPEED_KMH     = 45.0          # cruise speed approaching pedestrian
ADS_BRAKE_DECEL   = 8.0           # m/s² maximum emergency braking
PED_SPEED_MS      = 2.5           # Walking/jogging speed across the street

# Masses (kg)
ADS_MASS_KG       = 1800.0
PED_MASS_KG       = 75.0          # Average human weight for CS equation

# ADS detects the conflict when the pedestrian is within this radius
DETECTION_DIST_M  = 16.0

# Spawning parameters along the chosen road
ADS_APPROACH_M    = 50.0          # How far back the ADS spawns from the crossing spot
IMG_W, IMG_H      = 1280, 720
CAMERA_FOV        = 90
OUTPUT_GIF        = "cs_pedestrian.gif"


# ── helpers ───────────────────────────────────────────────────────────────────
def kmh_to_ms(v): return v / 3.6
def ms_to_kmh(v): return v * 3.6


def save_frame(image, frame_list):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))   # BGRA
    frame_list.append(Image.fromarray(array[:, :, [2, 1, 0]], mode="RGB"))


def set_velocity(actor, speed_ms, direction=None):
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


def compute_cs(v_residual_ms, mass_ads, mass_ped):
    """
    CS = v_residual × (m_ADS / (m_ADS + m_pedestrian))
    Since m_ADS >> m_ped, the mass ratio approaches 1.0, 
    meaning almost all kinetic energy shifts directly into the pedestrian.
    """
    mass_ratio = mass_ads / (mass_ads + mass_ped)
    return v_residual_ms * mass_ratio


def find_straight_road_segment(world_map):
    """
    Finds a straight road section away from intersections.
    Returns a waypoint right in the middle of a straight line segment.
    """
    topology = world_map.get_topology()
    
    # Let's find a long edge segment
    for wp_start, wp_end in topology:
        # Avoid roads that transition directly into junctions
        if wp_start.is_junction or wp_end.is_junction:
            continue
            
        # Check if the road is long enough and relatively straight
        dist = dist2d(wp_start.transform.location, wp_end.transform.location)
        if dist > 120.0:
            # Let's verify intermediate direction stability
            mid_wp = wp_start.next(dist / 2.0)[0]
            if not mid_wp.is_junction:
                return mid_wp
                
    # Fallback to map spawn points if topology parsing misses a clean stretch
    spawn_points = world_map.get_spawn_points()
    return world_map.get_waypoint(spawn_points[0].location)


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

        # ── Find Straight Road and Calculate Points ───────────────────────────
        crossing_wp = find_straight_road_segment(world_map)
        crossing_loc = crossing_wp.transform.location
        
        # ADS spawns back along the road
        ads_prev = crossing_wp.previous(ADS_APPROACH_M)
        if not ads_prev:
            raise RuntimeError("Could not find approach stretch for ADS")
        ads_spawn_wp = ads_prev[0]
        
        ads_spawn_tf = ads_spawn_wp.transform
        ads_spawn_tf.location.z += 0.3
        ads_dir = normalize2d(ads_spawn_wp.transform.get_forward_vector())
        
        # Calculate Pedestrian crossing angle (perpendicular to road direction)
        # Vector rotated 90 degrees: (x, y) -> (-y, x)
        ped_cross_dir = carla.Vector3D(-ads_dir.y, ads_dir.x, 0.0)
        
        # Spawn pedestrian on the right shoulder (~5 meters to the side)
        ped_spawn_loc = crossing_loc - (ped_cross_dir * 5.0)
        ped_spawn_loc.z += 1.0  # Spawning clearance
        ped_spawn_rot = carla.Rotation(yaw=math.degrees(math.atan2(ped_cross_dir.y, ped_cross_dir.x)))
        ped_spawn_tf  = carla.Transform(ped_spawn_loc, ped_spawn_rot)

        print(f"[INFO] Chosen Straight Road ID: {crossing_wp.road_id}")
        print(f"[INFO] Crossing Point: ({crossing_loc.x:.1f}, {crossing_loc.y:.1f})")

        # ── Spawn ADS Vehicle ─────────────────────────────────────────────────
        ads_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        ads_bp.set_attribute("color",     "30,144,255")   # Blue
        ads_bp.set_attribute("role_name", "ADS")
        ads = world.spawn_actor(ads_bp, ads_spawn_tf)
        actor_list.append(ads)
        
        ads.set_simulate_physics(True)
        ads_physics = ads.get_physics_control()
        ads_physics.mass = ADS_MASS_KG
        ads.apply_physics_control(ads_physics)

        # ── Spawn Pedestrian ──────────────────────────────────────────────────
        ped_bp = bp_lib.filter("walker.pedestrian.*")[0]
        pedestrian = world.spawn_actor(ped_bp, ped_spawn_tf)
        actor_list.append(pedestrian)
        
        # ── Camera Setup ──────────────────────────────────────────────────────
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMG_W))
        cam_bp.set_attribute("image_size_y", str(IMG_H))
        cam_bp.set_attribute("fov",          str(CAMERA_FOV))
        
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

        # ── Warm-up ticks ─────────────────────────────────────────────────────
        for _ in range(20):
            world.tick()

        # ── Simulation State ──────────────────────────────────────────────────
        ads_speed_ms   = kmh_to_ms(ADS_SPEED_KMH)
        braking        = False
        collision_tick = None
        v_at_collision = None
        cs_computed    = False

        total_ticks = int(RECORD_SECONDS / FIXED_DELTA_T)
        print(f"[INFO] Simulating {RECORD_SECONDS}s ({total_ticks} ticks)…")
        
        # Explicit controller handle for CARLA walker movement
        ped_control = carla.WalkerControl()
        ped_control.speed = PED_SPEED_MS
        ped_control.direction = ped_cross_dir

        for tick_i in range(total_ticks):
            t = tick_i * FIXED_DELTA_T

            ads_loc = ads.get_location()
            ped_loc = pedestrian.get_location()
            gap     = dist2d(ads_loc, ped_loc)

            # ── Detect Conflict ───────────────────────────────────────────────
            if not braking and gap < DETECTION_DIST_M:
                braking = True
                print(f"  [t={t:.2f}s] Conflict detected! Pedestrian crossing! ADS Braking hard.")

            # ── ADS Deceleration Logic ────────────────────────────────────────
            if braking:
                ads_speed_ms = max(0.0, ads_speed_ms - ADS_BRAKE_DECEL * FIXED_DELTA_T)

            # ── Detect Collision (Vehicles vs Pedestrian bounding check) ──────
            if collision_tick is None and gap < 2.0 and tick_i > 10:
                collision_tick = tick_i
                v_at_collision = ads_speed_ms
                cs = compute_cs(v_at_collision, ADS_MASS_KG, PED_MASS_KG)
                
                print(f"\n  *** PEDESTRIAN IMPACT at t={t:.2f}s ***")
                print(f"      ADS residual speed : {v_at_collision:.2f} m/s ({ms_to_kmh(v_at_collision):.1f} km/h)")
                print(f"      Mass ratio (ADS)   : {ADS_MASS_KG:.0f} / ({ADS_MASS_KG:.0f} + {PED_MASS_KG:.0f}) = {ADS_MASS_KG/(ADS_MASS_KG+PED_MASS_KG):.3f}")
                print(f"      CS = {v_at_collision:.2f} × {ADS_MASS_KG/(ADS_MASS_KG+PED_MASS_KG):.3f} = {cs:.2f}")
                cs_computed = True

            # ── Apply Velocities / Direct Control ─────────────────────────────
            if collision_tick is None or tick_i < collision_tick + 3:
                set_velocity(ads, ads_speed_ms, ads_dir)
                pedestrian.apply_control(ped_control)

            world.tick()

            # Console tracking log
            if tick_i % int(1.0 / FIXED_DELTA_T) == 0:
                print(f"  t={t:5.1f}s  "
                      f"ADS {ms_to_kmh(ads_speed_ms):5.1f} km/h  "
                      f"gap={gap:5.1f} m  "
                      f"{'[BRAKING]' if braking else ''}  "
                      f"{'[IMPACT]' if collision_tick and tick_i >= collision_tick else ''}")

        # ── CS summary ────────────────────────────────────────────────────────
        print("\n── CS Summary ───────────────────────────────────────────────")
        if cs_computed:
            cs_final = compute_cs(v_at_collision, ADS_MASS_KG, PED_MASS_KG)
            ratio    = ADS_MASS_KG / (ADS_MASS_KG + PED_MASS_KG)
            print(f"  v_residual          = {v_at_collision:.2f} m/s")
            print(f"  m_ADS               = {ADS_MASS_KG:.0f} kg")
            print(f"  m_pedestrian        = {PED_MASS_KG:.0f} kg")
            print(f"  mass ratio          = {ratio:.3f}")
            print(f"  Final CS            = {cs_final:.2f}")
        else:
            print("  No impact detected. Check DETECTION_DIST_M or tuning speeds.")
        print("─────────────────────────────────────────────────────────────\n")

        # ── Encode GIF ────────────────────────────────────────────────────────
        if frames:
            print(f"[INFO] Writing GIF → {args.output}…")
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
        print("[INFO] Cleaning up actors…")
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
        description="CARLA Conflict Severity — Pedestrian Mid-Block Crossing Scenario"
    )
    parser.add_argument("--host",   default="127.0.0.1")
    parser.add_argument("--port",   default=2000, type=int)
    parser.add_argument("--output", default=OUTPUT_GIF)
    main(parser.parse_args())
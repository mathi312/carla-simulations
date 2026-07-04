"""
Conflict Severity (CS) — Red-Light Runner Scenario
====================================================
Town03 — signalised intersection.

Scenario:
  - ADS drives toward an intersection on green at ~50 km/h
  - Another vehicle runs the red light and enters from the side road
  - ADS detects the conflict and brakes at maximum deceleration
  - A collision still occurs; residual speed at impact is recorded
  - CS is computed as:
        CS = v_residual × (m_ADS / (m_ADS + m_other))
  - With m_ADS=1800 kg, m_other=1500 kg, v_residual≈4 m/s → CS≈2.2

Why Town03:
  Town03 has signalised four-way intersections on straight road segments.
  We pick one programmatically by finding a junction waypoint that has
  traffic lights and straight approach roads on at least two axes.

Determinism: synchronous mode + fixed delta + fixed seeds.
Color fix:   CARLA raw_data is BGRA → reorder to RGB via [:,:,[2,1,0]].

Usage:
    python cs_red_light.py [--host HOST] [--port PORT] [--output OUTPUT]
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

ADS_SPEED_KMH     = 50.0          # cruise speed approaching intersection
ADS_BRAKE_DECEL   = 8.0           # m/s²  maximum emergency braking
OTHER_SPEED_KMH   = 45.0          # red-light runner approach speed

# Vehicle masses (kg) — match the paper example
ADS_MASS_KG       = 1800.0
OTHER_MASS_KG     = 1500.0

# ADS detects the conflict when the other car is this many metres into
# the intersection path (gives time to brake but not enough to stop)
DETECTION_DIST_M  = 18.0

# How far before the intersection centre the ADS spawns
ADS_APPROACH_M    = 60.0
# How far to the side of the intersection the other car spawns
OTHER_APPROACH_M  = 50.0

IMG_W, IMG_H      = 1280, 720
CAMERA_FOV        = 90
OUTPUT_GIF        = "cs_red_light.gif"


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


def compute_cs(v_residual_ms, mass_ads, mass_other):
    """
    CS = v_residual × (m_ADS / (m_ADS + m_other))
    Higher mass ratio → ADS dominates → more energy transferred to other agent.
    """
    mass_ratio = mass_ads / (mass_ads + mass_other)
    return v_residual_ms * mass_ratio


def find_signalised_intersection(world, world_map):
    """
    Find a four-way signalised intersection in Town03 with straight
    approach roads in (roughly) two perpendicular axes.

    Strategy:
      1. Collect all traffic lights in the map.
      2. For each light, get its stop waypoints to find the junction.
      3. Pick the junction whose approach roads are most perpendicular
         and have the longest straight runs (good camera framing).

    Returns (junction_centre, ads_approach_wp, other_approach_wp)
    where ads_approach_wp is on the main axis and other_approach_wp
    is on the perpendicular axis (the red-light runner's road).
    """
    import carla

    traffic_lights = world.get_actors().filter("traffic.traffic_light")
    if not traffic_lights:
        raise RuntimeError("No traffic lights found — is Town03 loaded?")

    # Collect unique junction IDs and their approach waypoints
    junctions = {}   # junction_id → list of (wp, traffic_light)
    for tl in traffic_lights:
        for wp in tl.get_stop_waypoints():
            jid = wp.get_junction().id if wp.is_junction else None
            # The stop waypoint is just before the junction; get the junction via next()
            nxt = wp.next(2.0)
            if nxt and nxt[0].is_junction:
                jid = nxt[0].get_junction().id
            if jid is None:
                # Try directly
                jid = id(tl)   # group by traffic light if no junction id
            if jid not in junctions:
                junctions[jid] = []
            junctions[jid].append(wp)

    best = None
    best_score = -1

    for jid, wps in junctions.items():
        if len(wps) < 2:
            continue

        # Compute all pairwise heading differences to find ~90° pairs
        for i in range(len(wps)):
            for j in range(i + 1, len(wps)):
                h1 = wps[i].transform.rotation.yaw
                h2 = wps[j].transform.rotation.yaw
                diff = abs((h1 - h2 + 180) % 360 - 180)
                # Want close to 90°
                perp_score = 1.0 - abs(diff - 90) / 90.0

                # Prefer long straight approaches
                def straight_run(wp):
                    chain, cur = 0, wp
                    for _ in range(20):
                        nxt = cur.previous(3.0)
                        if not nxt:
                            break
                        h0 = wp.transform.rotation.yaw
                        hn = nxt[0].transform.rotation.yaw
                        if abs((hn - h0 + 180) % 360 - 180) > 15:
                            break
                        cur = nxt[0]
                        chain += 1
                    return chain

                score = perp_score * (straight_run(wps[i]) + straight_run(wps[j]))
                if score > best_score:
                    best_score = score
                    best = (wps[i], wps[j])

    if best is None:
        raise RuntimeError("Could not find a suitable signalised intersection.")

    wp_a, wp_b = best

    # Compute intersection centre: average of the two stop-line locations
    # projected forward into the junction
    def junction_entry(wp):
        nxt = wp.next(6.0)
        return nxt[0].transform.location if nxt else wp.transform.location

    loc_a = junction_entry(wp_a)
    loc_b = junction_entry(wp_b)
    import carla
    centre = carla.Location(
        x=(loc_a.x + loc_b.x) / 2,
        y=(loc_a.y + loc_b.y) / 2,
        z=(loc_a.z + loc_b.z) / 2,
    )

    print(f"[INFO] Intersection centre: ({centre.x:.1f}, {centre.y:.1f})  "
          f"approach_A road={wp_a.road_id} yaw={wp_a.transform.rotation.yaw:.0f}°  "
          f"approach_B road={wp_b.road_id} yaw={wp_b.transform.rotation.yaw:.0f}°")

    return centre, wp_a, wp_b


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

        # ── find intersection ─────────────────────────────────────────────────
        centre, wp_ads_stop, wp_other_stop = find_signalised_intersection(world, world_map)

        # ADS spawns ADS_APPROACH_M back along its approach road
        ads_prev = wp_ads_stop.previous(ADS_APPROACH_M)
        if not ads_prev:
            raise RuntimeError("Cannot walk back along ADS approach road")
        ads_spawn_wp = ads_prev[0]

        # Other car spawns OTHER_APPROACH_M back along the perpendicular road
        other_prev = wp_other_stop.previous(OTHER_APPROACH_M)
        if not other_prev:
            raise RuntimeError("Cannot walk back along other car approach road")
        other_spawn_wp = other_prev[0]

        ads_spawn_tf       = ads_spawn_wp.transform
        ads_spawn_tf.location.z += 0.3
        other_spawn_tf     = other_spawn_wp.transform
        other_spawn_tf.location.z += 0.3

        # Pre-compute constant travel directions (road forward at spawn)
        ads_dir   = normalize2d(ads_spawn_wp.transform.get_forward_vector())
        other_dir = normalize2d(other_spawn_wp.transform.get_forward_vector())

        # ── spawn vehicles ────────────────────────────────────────────────────
        ads_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        ads_bp.set_attribute("color",     "30,144,255")   # blue
        ads_bp.set_attribute("role_name", "ADS")

        # Use a heavier-looking vehicle for the red-light runner
        runner_bps = bp_lib.filter("vehicle.dodge.charger_2020")
        if not runner_bps:
            runner_bps = bp_lib.filter("vehicle.audi.tt")
        other_bp = runner_bps[0]
        other_bp.set_attribute("color",     "220,50,50")   # red
        other_bp.set_attribute("role_name", "runner")

        ads   = world.spawn_actor(ads_bp,   ads_spawn_tf)
        other = world.spawn_actor(other_bp, other_spawn_tf)
        actor_list.extend([ads, other])
        ads.set_simulate_physics(True)
        other.set_simulate_physics(True)

        # Set vehicle masses via physics control to match the paper values
        ads_physics         = ads.get_physics_control()
        ads_physics.mass    = ADS_MASS_KG
        ads.apply_physics_control(ads_physics)

        other_physics       = other.get_physics_control()
        other_physics.mass  = OTHER_MASS_KG
        other.apply_physics_control(other_physics)

        # ── freeze all traffic lights at this intersection ────────────────────
        # ADS gets green (frozen), runner's light is red (frozen) — this is the
        # red-light violation setup.  We freeze them so they never cycle during
        # the short scenario window.
        tl_actors = world.get_actors().filter("traffic.traffic_light")
        ads_tl    = None   # traffic light on ADS approach
        other_tl  = None   # traffic light on other car's approach

        for tl in tl_actors:
            for sw in tl.get_stop_waypoints():
                if dist2d(sw.transform.location, wp_ads_stop.transform.location) < 8.0:
                    ads_tl = tl
                if dist2d(sw.transform.location, wp_other_stop.transform.location) < 8.0:
                    other_tl = tl

        if ads_tl:
            ads_tl.set_state(carla.TrafficLightState.Green)
            ads_tl.freeze(True)
            print("[INFO] ADS traffic light → GREEN (frozen)")
        if other_tl:
            other_tl.set_state(carla.TrafficLightState.Red)
            other_tl.freeze(True)
            print("[INFO] Runner traffic light → RED (frozen)")

        # ── camera — chase behind ADS, wide enough to see the intersection ────
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
        other_speed_ms = kmh_to_ms(OTHER_SPEED_KMH)
        braking        = False
        collision_tick = None
        v_at_collision = None
        cs_computed    = False

        total_ticks = int(RECORD_SECONDS / FIXED_DELTA_T)
        print(f"[INFO] Simulating {RECORD_SECONDS}s ({total_ticks} ticks)…")
        print(f"[INFO] ADS mass={ADS_MASS_KG:.0f} kg  |  "
              f"Runner mass={OTHER_MASS_KG:.0f} kg")

        for tick_i in range(total_ticks):
            t = tick_i * FIXED_DELTA_T

            ads_loc   = ads.get_location()
            other_loc = other.get_location()
            gap       = dist2d(ads_loc, other_loc)
            dist_to_centre = dist2d(ads_loc, centre)

            # ── detect conflict: other car within DETECTION_DIST_M and in path ─
            if not braking and gap < DETECTION_DIST_M:
                braking = True
                print(f"  [t={t:.2f}s] Conflict detected — ADS braking hard  "
                      f"(gap={gap:.1f}m, ADS speed={ms_to_kmh(ads_speed_ms):.1f} km/h)")

            # ── ADS speed update ──────────────────────────────────────────────
            if braking:
                ads_speed_ms = max(0.0, ads_speed_ms - ADS_BRAKE_DECEL * FIXED_DELTA_T)

            # ── detect collision (vehicles within ~3 m of each other) ─────────
            if collision_tick is None and gap < 3.0 and tick_i > 10:
                collision_tick = tick_i
                v_at_collision = ads_speed_ms
                cs = compute_cs(v_at_collision, ADS_MASS_KG, OTHER_MASS_KG)
                print(f"\n  *** COLLISION at t={t:.2f}s ***")
                print(f"      ADS residual speed : {v_at_collision:.2f} m/s  "
                      f"({ms_to_kmh(v_at_collision):.1f} km/h)")
                print(f"      Mass ratio (ADS)   : {ADS_MASS_KG:.0f} / "
                      f"({ADS_MASS_KG:.0f} + {OTHER_MASS_KG:.0f}) = "
                      f"{ADS_MASS_KG/(ADS_MASS_KG+OTHER_MASS_KG):.3f}")
                print(f"      CS = {v_at_collision:.2f} × "
                      f"{ADS_MASS_KG/(ADS_MASS_KG+OTHER_MASS_KG):.3f} = {cs:.2f}")
                cs_computed = True

            # ── apply velocities ──────────────────────────────────────────────
            # After collision let physics engine take over (stop forcing velocity)
            if collision_tick is None or tick_i < collision_tick + 5:
                set_velocity(ads,   ads_speed_ms,   ads_dir)
                set_velocity(other, other_speed_ms, other_dir)

            world.tick()

            # Console every second
            if tick_i % int(1.0 / FIXED_DELTA_T) == 0:
                print(f"  t={t:5.1f}s  "
                      f"ADS {ms_to_kmh(ads_speed_ms):5.1f} km/h  "
                      f"gap={gap:5.1f} m  "
                      f"dist_to_centre={dist_to_centre:5.1f} m  "
                      f"{'[BRAKING]' if braking else ''}  "
                      f"{'[COLLISION]' if collision_tick and tick_i >= collision_tick else ''}")

        # ── CS summary ────────────────────────────────────────────────────────
        print("\n── CS Summary ───────────────────────────────────────────────")
        if cs_computed:
            cs_final = compute_cs(v_at_collision, ADS_MASS_KG, OTHER_MASS_KG)
            ratio    = ADS_MASS_KG / (ADS_MASS_KG + OTHER_MASS_KG)
            print(f"  v_residual          = {v_at_collision:.2f} m/s")
            print(f"  m_ADS               = {ADS_MASS_KG:.0f} kg")
            print(f"  m_other             = {OTHER_MASS_KG:.0f} kg")
            print(f"  mass ratio          = {ADS_MASS_KG:.0f} / "
                  f"{ADS_MASS_KG + OTHER_MASS_KG:.0f} = {ratio:.3f}")
            print(f"  CS = {v_at_collision:.2f} × {ratio:.3f} = {cs_final:.2f}")
            print(f"  (Paper target: v_residual=4 m/s → CS≈2.2)")
        else:
            print("  No collision detected — try reducing ADS_APPROACH_M "
                  "or increasing OTHER_SPEED_KMH so the paths overlap.")
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
        description="CARLA Conflict Severity — red-light runner scenario"
    )
    parser.add_argument("--host",   default="127.0.0.1")
    parser.add_argument("--port",   default=2000, type=int)
    parser.add_argument("--output", default=OUTPUT_GIF)
    main(parser.parse_args())
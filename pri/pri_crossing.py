"""
PRI Pedestrian Crossing Scenario — CARLA Simulation
=====================================================
Town03 — uses a mid-block zebra crossing on a straight road segment.

Scenario:
  - ADS approaches a zebra crossing at ~55 km/h
  - A pedestrian steps onto the crossing just as the ADS closes in
  - ADS applies emergency braking when pedestrian is detected
  - PRI is sampled at t_tz = 3.0 s, 1.5 s, 0.5 s before the crossing

PRI formula (summed over crossing event):
  PRI = Σ  v_impact(t) / TTZ(t)

Color fix:
  CARLA raw_data is BGRA. Slicing [:,:,:3] gives BGR which PIL reads as
  RGB → red/blue swap. We reorder explicitly to RGB via [:,:,[2,1,0]].

Determinism:
  Synchronous mode + fixed delta + fixed seeds.

Usage:
    python pri_crossing.py [--host HOST] [--port PORT] [--output OUTPUT]
"""

import argparse
import math
import random
import sys

import numpy as np
from PIL import Image

# ── tuneable constants ────────────────────────────────────────────────────────
SEED             = 42
FIXED_DELTA_T    = 0.05          # seconds per tick (20 fps physics)
GIF_FPS          = 20
RECORD_SECONDS   = 10.0
ADS_SPEED_KMH    = 45          # initial cruise speed
BRAKE_DECEL      = 7           # m/s² deceleration when braking
PED_WALK_SPEED   = 1.6           # m/s pedestrian crossing speed
DETECTION_TTZ    = 3          # ADS detects pedestrian at this TTZ (s)
STOP_BUFFER_M    = 10.0          # stop the ADS a bit earlier before the crossing
IMG_W, IMG_H     = 1280, 720
CAMERA_FOV       = 90
OUTPUT_GIF       = "pri_crossing.gif"

# PRI sample points (TTZ in seconds at which we record v_impact)
PRI_SAMPLE_TTZ   = [3.0, 1.5, 0.5]


# ── helpers ───────────────────────────────────────────────────────────────────
def kmh_to_ms(v): return v / 3.6


def save_frame(image, frame_list):
    """
    CARLA → PIL, with correct channel order.
    CARLA stores BGRA; reorder to RGB so PIL colours are faithful.
    Naive [:,:,:3] gives BGR which PIL misreads as RGB (red↔blue swap).
    """
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # BGRA
    rgb   = array[:, :, [2, 1, 0]]                         # → RGB
    frame_list.append(Image.fromarray(rgb, mode="RGB"))


def set_speed(actor, speed_ms, direction=None):
    """Force an actor to a given speed along `direction` (default: forward)."""
    import carla
    if direction is None:
        direction = actor.get_transform().get_forward_vector()
    actor.set_target_velocity(carla.Vector3D(
        x=direction.x * speed_ms,
        y=direction.y * speed_ms,
        z=direction.z * speed_ms,
    ))


def dist2d(a, b):
    """XY distance between two carla.Location objects."""
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)


def find_midblock_crossing(world_map):
    """
    Locate a genuine mid-block zebra crossing in Town03 — one that sits
    on a straight road segment well away from any junction.

    get_crosswalks() returns a flat list of polygon vertices, with each
    polygon closed by repeating its first point.  We split them, compute
    centroids, project onto the road, then reject anything inside or
    within 40 m of a junction.  First survivor wins.

    Returns (road_waypoint, crossing_centre_Location).
    """
    import carla

    def centroid(pts):
        return carla.Location(
            x=sum(p.x for p in pts) / len(pts),
            y=sum(p.y for p in pts) / len(pts),
            z=sum(p.z for p in pts) / len(pts),
        )

    def near_junction(wp, look_m=40):
        for step in range(1, int(look_m / 2) + 1):
            for w in (wp.next(2.0 * step) or []) + (wp.previous(2.0 * step) or []):
                if w.is_junction:
                    return True
        return False

    raw = world_map.get_crosswalks()   # flat list of carla.Location

    # Split into polygons: each ring is closed by repeating its first vertex
    polygons, current = [], []
    for loc in raw:
        if current and abs(loc.x - current[0].x) < 0.05 and abs(loc.y - current[0].y) < 0.05:
            polygons.append(current)
            current = []
        else:
            current.append(loc)
    if current:
        polygons.append(current)

    for poly in polygons:
        if len(poly) < 3:
            continue
        c  = centroid(poly)
        wp = world_map.get_waypoint(c, project_to_road=True,
                                     lane_type=carla.LaneType.Driving)
        if wp is None or wp.is_junction:
            continue
        if near_junction(wp, look_m=40):
            continue
        print(f"[INFO] Mid-block crossing: road={wp.road_id} s={wp.s:.1f} "
              f"loc=({c.x:.1f}, {c.y:.1f})")
        return wp, c

    # Hard-coded fallback — known mid-block crossing in Town03
    print("[WARN] No mid-block crossing found via API — using hard-coded fallback.")
    c  = carla.Location(x=150.5, y=55.0, z=0.0)
    wp = world_map.get_waypoint(c, project_to_road=True,
                                  lane_type=carla.LaneType.Driving)
    return wp, c


# ── PRI accumulator ───────────────────────────────────────────────────────────
class PRITracker:
    def __init__(self, sample_ttz_list):
        self.targets  = sorted(sample_ttz_list, reverse=True)  # descending
        self.samples  = []   # list of (ttz, v_impact, pri_term)
        self.recorded = set()

    def record_if_due(self, ttz, v_impact):
        """Record a sample when TTZ crosses a target value."""
        for target in self.targets:
            if target not in self.recorded and ttz <= target + 0.12:
                term = v_impact / max(ttz, 0.01)
                self.samples.append((target, v_impact, term))
                self.recorded.add(target)
                print(f"  [PRI sample]  TTZ={target:.1f}s  "
                      f"v_impact={v_impact:.2f} m/s  "
                      f"term={term:.3f}")
                break

    @property
    def total(self):
        return sum(s[2] for s in self.samples)


# ── main ──────────────────────────────────────────────────────────────────────
def main(args):
    import carla

    random.seed(SEED)
    np.random.seed(SEED)

    frames     = []
    actor_list = []

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    print("[INFO] Loading Town03 (mid-block zebra crossings)…")
    world = client.load_world("Town03")
    original_settings = world.get_settings()

    try:
        # ── synchronous fixed-step mode ───────────────────────────────────────
        settings = world.get_settings()
        settings.synchronous_mode    = True
        settings.fixed_delta_seconds = FIXED_DELTA_T
        settings.no_rendering_mode   = False
        world.apply_settings(settings)

        tm = client.get_trafficmanager()
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(SEED)

        bp_lib  = world.get_blueprint_library()
        world_map = world.get_map()

        # ── locate a mid-block crossing ───────────────────────────────────────
        crossing_wp, crossing_centre = find_midblock_crossing(world_map)
        print(f"[INFO] Crossing waypoint: road={crossing_wp.road_id}  "
              f"lane={crossing_wp.lane_id}  s={crossing_wp.s:.1f}  "
              f"loc={crossing_wp.transform.location}")

        # ADS spawns APPROACH_DIST metres before the crossing
        APPROACH_DIST = 55.0          # gives ~3.6 s at 55 km/h to reach crossing
        prev_wps = crossing_wp.previous(APPROACH_DIST)
        if not prev_wps:
            raise RuntimeError("Could not find approach waypoint — try adjusting APPROACH_DIST")
        ads_wp = prev_wps[0]

        ads_spawn = ads_wp.transform
        ads_spawn.location.z += 0.3

        # ── spawn ADS vehicle ─────────────────────────────────────────────────
        ads_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        ads_bp.set_attribute("color",     "30,144,255")   # dodger blue
        ads_bp.set_attribute("role_name", "ADS")
        ads = world.spawn_actor(ads_bp, ads_spawn)
        actor_list.append(ads)
        ads.set_simulate_physics(True)

        # ── spawn pedestrian ──────────────────────────────────────────────────
        # Use the actual crossing polygon centroid so the pedestrian is placed
        # exactly on the zebra marking, not just the nearest road waypoint.
        crossing_loc = crossing_centre

        # Perpendicular direction across the road (right-hand side of road fwd)
        fwd   = crossing_wp.transform.get_forward_vector()
        right = carla.Vector3D(x=-fwd.y, y=fwd.x, z=0.0)   # 90° CW in XY plane

        # Start close to the crossing but walk across a longer span so the
        # pedestrian clearly traverses the scene instead of stopping mid-way.
        SPAWN_OFFSET = 1.6   # spawn just off the crossing edge
        TARGET_OFFSET = 7.0  # walk across the road to the far side

        ped_start = carla.Location(
            x=crossing_loc.x + right.x * SPAWN_OFFSET,
            y=crossing_loc.y + right.y * SPAWN_OFFSET,
            z=crossing_loc.z + 0.93,   # standard walker spawn height
        )
        # Walk toward the opposite side of the road
        ped_target_loc = carla.Location(
            x=crossing_loc.x - right.x * TARGET_OFFSET,
            y=crossing_loc.y - right.y * TARGET_OFFSET,
            z=crossing_loc.z,
        )
        # Direction vector the pedestrian will walk (constant, perpendicular to road)
        ped_walk_dir = carla.Vector3D(
            x=-right.x,
            y=-right.y,
            z=0.0,
        )
        ped_walk_len = math.sqrt(ped_walk_dir.x**2 + ped_walk_dir.y**2) + 1e-9
        ped_walk_dir = carla.Vector3D(
            ped_walk_dir.x / ped_walk_len,
            ped_walk_dir.y / ped_walk_len,
            0.0,
        )

        ped_bps = bp_lib.filter("walker.pedestrian.*")
        random.seed(SEED)
        ped_bp  = random.choice(ped_bps)
        ped_bp.set_attribute("is_invincible", "false")

        # Spawn pedestrian facing the direction they will walk
        ped_yaw = math.degrees(math.atan2(ped_walk_dir.y, ped_walk_dir.x))
        ped_spawn_tf = carla.Transform(
            ped_start,
            carla.Rotation(yaw=ped_yaw),
        )
        pedestrian = world.spawn_actor(ped_bp, ped_spawn_tf)
        actor_list.append(pedestrian)
        # No AI controller — we drive the pedestrian manually each tick
        # so they cross exactly the right zebra regardless of navmesh routing.

        # ── camera — attached to ADS ──────────────────────────────────────────
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMG_W))
        cam_bp.set_attribute("image_size_y", str(IMG_H))
        cam_bp.set_attribute("fov",          str(CAMERA_FOV))
        cam_bp.set_attribute("sensor_tick",  "0.0")

        cam_tf = carla.Transform(
            carla.Location(x=-7.0, z=3.2),
            carla.Rotation(pitch=-8.0)
        )
        camera = world.spawn_actor(cam_bp, cam_tf, attach_to=ads)
        actor_list.append(camera)
        camera.listen(lambda img: save_frame(img, frames))

        # ── warm-up ticks ─────────────────────────────────────────────────────
        for _ in range(15):
            world.tick()

        # ── simulation loop ───────────────────────────────────────────────────
        ads_speed_ms  = kmh_to_ms(ADS_SPEED_KMH)
        braking       = False
        pri_tracker   = PRITracker(PRI_SAMPLE_TTZ)
        ped_spawn_z   = ped_start.z
        ped_yaw_deg   = ped_yaw

        total_ticks   = int(RECORD_SECONDS / FIXED_DELTA_T)
        print(f"[INFO] Simulating {RECORD_SECONDS}s  ({total_ticks} ticks) …")
        print(f"[INFO] Pedestrian walks manually — no AI navmesh routing.")

        for tick_i in range(total_ticks):
            t = tick_i * FIXED_DELTA_T

            # ── pedestrian: use walker controls so it actually walks ──
            ped_loc  = pedestrian.get_location()
            ped_dist = dist2d(ped_loc, ped_target_loc)
            if ped_dist > 0.3:
                control = carla.WalkerControl()
                control.speed = PED_WALK_SPEED
                control.direction = carla.Vector3D(
                    x=ped_walk_dir.x,
                    y=ped_walk_dir.y,
                    z=0.0,
                )
                control.jump = False
                pedestrian.apply_control(control)
            else:
                control = carla.WalkerControl()
                control.speed = 0.0
                control.direction = carla.Vector3D(0.0, 0.0, 0.0)
                control.jump = False
                pedestrian.apply_control(control)
                pedestrian.set_transform(carla.Transform(
                    carla.Location(x=ped_target_loc.x, y=ped_target_loc.y, z=ped_spawn_z),
                    carla.Rotation(yaw=ped_yaw_deg),
                ))

            ads_loc     = ads.get_location()
            crossing_d  = dist2d(ads_loc, crossing_loc)

            # Time-to-Zebra (TTZ): how long until ADS reaches the crossing
            # at current speed (>0 guard)
            ttz = crossing_d / max(ads_speed_ms, 0.1)

            # Trigger braking earlier so the ADS comes to rest before the
            # crossing rather than at or after it.
            required_stop_dist = max(2.0, (ads_speed_ms ** 2) / (2 * max(BRAKE_DECEL, 0.1)) + STOP_BUFFER_M)
            if (not braking and ttz <= DETECTION_TTZ and crossing_d <= required_stop_dist + 0.5):
                braking = True
                print(f"  [t={t:.2f}s] Pedestrian detected — ADS braking  "
                      f"(TTZ={ttz:.2f}s, gap={crossing_d:.1f}m, speed={ads_speed_ms*3.6:.1f} km/h)")

            # Update ADS speed
            if braking:
                ads_speed_ms = max(0.0, ads_speed_ms - BRAKE_DECEL * FIXED_DELTA_T)

            set_speed(ads, ads_speed_ms)

            # PRI: predicted impact speed ≈ current speed (worst-case, no further braking)
            # This matches the metric definition: speed at crossing if no change
            v_impact = ads_speed_ms

            # Record PRI sample at defined TTZ milestones (only while ADS is approaching)
            if crossing_d > 0.5:   # ADS hasn't passed the crossing yet
                pri_tracker.record_if_due(ttz, v_impact)

            world.tick()

            # Console summary every second
            if tick_i % int(1.0 / FIXED_DELTA_T) == 0:
                print(f"  t={t:5.1f}s  TTZ={ttz:.2f}s  "
                      f"v={ads_speed_ms*3.6:.1f} km/h  "
                      f"gap_to_crossing={crossing_d:.1f}m  "
                      f"{'[BRAKING]' if braking else ''}")

        # ── PRI summary ───────────────────────────────────────────────────────
        print("\n── PRI Summary ──────────────────────────────────────────")
        print(f"  {'TTZ (s)':<10} {'v_impact (m/s)':<18} {'term (v/TTZ)':<15}")
        print(f"  {'-'*10} {'-'*18} {'-'*15}")
        for (ttz_s, v_imp, term) in pri_tracker.samples:
            print(f"  {ttz_s:<10.1f} {v_imp:<18.2f} {term:<15.3f}")
        print(f"  {'─'*43}")
        print(f"  PRI  =  {pri_tracker.total:.2f}")
        print(f"  (Example target from paper: ≈12.7)")
        print("─────────────────────────────────────────────────────────\n")

        # ── encode GIF ────────────────────────────────────────────────────────
        if not frames:
            print("[ERROR] No frames captured — check camera sensor setup.")
            return

        duration_ms = int(1000 / GIF_FPS)
        print(f"[INFO] Writing GIF → {args.output}  "
              f"({len(frames)} frames @ {GIF_FPS} fps) …")
        frames[0].save(
            args.output,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,      # preserve all colours; no palette reduction
        )
        print(f"[INFO] Saved: {args.output}")

    finally:
        # ── teardown ──────────────────────────────────────────────────────────
        print("[INFO] Cleaning up actors …")
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
    parser = argparse.ArgumentParser(description="CARLA PRI pedestrian crossing scenario")
    parser.add_argument("--host",   default="127.0.0.1")
    parser.add_argument("--port",   default=2000, type=int)
    parser.add_argument("--output", default=OUTPUT_GIF)
    args = parser.parse_args()
    main(args)

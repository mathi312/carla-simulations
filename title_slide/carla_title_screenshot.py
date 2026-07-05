"""
carla_title_screenshot.py
─────────────────────────
Generates a cinematic title-slide screenshot from a running CARLA server.

What it produces
────────────────
  • Spawns a BMW 3-Series (or closest available) in Town10 (urban)
  • Attaches an RGB camera with a low, slightly angled cinematic FOV
  • Overlays in post-process (via PIL / OpenCV, no CARLA UI):
      - Blue planned trajectory arc on the road
      - Coloured bounding boxes around nearby vehicles & pedestrians
      - Subtle LiDAR point-cloud dots (rendered from a LiDAR sensor)
  • Lots of empty upper-left headroom for slide title text
  • Saves carla_title.png  (3840 × 2160 or 1920 × 1080 depending on --res)

Requirements
────────────
  pip install carla pygame numpy pillow opencv-python
  A running CARLA server:  ./CarlaUE4.sh  (or CarlaUE4.exe -windowed)

Usage
─────
  python carla_title_screenshot.py
  python carla_title_screenshot.py --host 127.0.0.1 --port 2000 --res 4k
  python carla_title_screenshot.py --res 1080p --town Town10 --out my_title.png
"""

import argparse
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

# ── CARLA import ──────────────────────────────────────────────────────────────
try:
    import carla
except ImportError:
    sys.exit(
        "carla package not found.\n"
        "Install it with:  pip install carla\n"
        "Make sure the version matches your CARLA server."
    )

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
RESOLUTIONS = {
    "4k":    (3840, 2160),
    "1080p": (1920, 1080),
    "720p":  (1280, 720),
}

# Colours (BGR for OpenCV, RGB for PIL)
BBOX_VEHICLE_COLOR   = (0, 200, 255)   # cyan-ish
BBOX_PEDESTRIAN_COLOR = (50, 255, 120) # green
TRAJ_COLOR           = (80, 160, 255)  # blue  (BGR)
LIDAR_COLOR          = (200, 230, 255) # pale blue-white

# Camera mount: low and slightly behind the ego, angled forward-down
CAM_X, CAM_Y, CAM_Z      = -6.0, 0.0, 2.8    # behind, centred, up
CAM_PITCH, CAM_YAW        = -8.0, 0.0         # slight downward tilt
CAM_FOV                   = 75

# LiDAR mount
LIDAR_Z = 2.4

# How many NPC vehicles & walkers to spawn
N_VEHICLES  = 25
N_WALKERS   = 15

# Trajectory: how many waypoints ahead, spacing in metres
TRAJ_STEPS  = 30
TRAJ_STEP_M = 2.5


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

from typing import List

def find_blueprint(bp_lib, preferred: List[str], fallback_filter: str) -> carla.ActorBlueprint:
    """Return the first preferred blueprint found, else a random fallback."""
    for name in preferred:
        bp = bp_lib.find(name)
        if bp:
            return bp
    candidates = bp_lib.filter(fallback_filter)
    return random.choice(list(candidates))


def get_camera_intrinsics(w: int, h: int, fov: float) -> np.ndarray:
    f = w / (2.0 * math.tan(math.radians(fov) / 2.0))
    cx, cy = w / 2.0, h / 2.0
    return np.array([[f, 0, cx],
                     [0, f, cy],
                     [0, 0,  1]], dtype=np.float64)


def world_to_image(world_pt: np.ndarray, K: np.ndarray,
                   cam_transform: carla.Transform, w: int, h: int):
    """
    Project a 3-D world point onto the image plane.
    Returns (u, v) pixel coords, or None if behind the camera.
    """
    # World → camera frame
    cam_mat = np.array(cam_transform.get_matrix())   # 4×4
    world_h = np.array([world_pt[0], world_pt[1], world_pt[2], 1.0])
    cam_h   = np.linalg.inv(cam_mat) @ world_h       # camera coords

    # CARLA camera axes: X=right, Y=down, Z=forward
    x_cam, y_cam, z_cam = cam_h[1], -cam_h[2], cam_h[0]

    if z_cam <= 0:          # behind camera
        return None

    u = int(K[0, 0] * x_cam / z_cam + K[0, 2])
    v = int(K[1, 1] * y_cam / z_cam + K[1, 2])

    if 0 <= u < w and 0 <= v < h:
        return (u, v)
    return None


def draw_bbox_3d(img: np.ndarray, actor: carla.Actor,
                 cam_transform: carla.Transform,
                 K: np.ndarray, color: tuple, w: int, h: int):
    """Draw a 3-D bounding box projected onto the image."""
    bb    = actor.bounding_box
    verts = bb.get_world_vertices(actor.get_transform())

    pts = []
    for v in verts:
        p = world_to_image(np.array([v.x, v.y, v.z]), K, cam_transform, w, h)
        pts.append(p)

    # 12 edges of the box
    edges = [
        (0,1),(1,3),(3,2),(2,0),   # bottom face
        (4,5),(5,7),(7,6),(6,4),   # top face
        (0,4),(1,5),(2,6),(3,7),   # verticals
    ]
    for a, b in edges:
        if pts[a] and pts[b]:
            cv2.line(img, pts[a], pts[b], color, 2, cv2.LINE_AA)


def draw_trajectory(img: np.ndarray, waypoints: list,
                    cam_transform: carla.Transform,
                    K: np.ndarray, color: tuple, w: int, h: int):
    """Draw a smooth trajectory arc on the road."""
    prev = None
    for wp in waypoints:
        loc = wp.transform.location
        # Raise slightly off ground so it's visible
        pt = world_to_image(np.array([loc.x, loc.y, loc.z + 0.15]),
                            K, cam_transform, w, h)
        if pt and prev:
            cv2.line(img, prev, pt, color, 4, cv2.LINE_AA)
            # Glow pass (thicker, lower opacity via blend)
            overlay = img.copy()
            cv2.line(overlay, prev, pt, color, 12, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
        prev = pt


def draw_lidar_dots(img: np.ndarray, lidar_data,
                    cam_transform: carla.Transform,
                    K: np.ndarray, color: tuple, w: int, h: int,
                    lidar_transform: carla.Transform):
    """Project LiDAR points onto the image as small dots."""
    pts = np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape(-1, 4)

    # LiDAR frame → world
    lidar_mat = np.array(lidar_transform.get_matrix())

    for pt in pts[::4]:                     # sample every 4th point for speed
        x, y, z, _ = pt
        world_h = lidar_mat @ np.array([x, y, z, 1.0])
        img_pt = world_to_image(world_h[:3], K, cam_transform, w, h)
        if img_pt:
            cv2.circle(img, img_pt, 2, color, -1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    W, H   = RESOLUTIONS[args.res]
    out_path = Path(args.out)

    # ── Connect ───────────────────────────────────────────────────────────────
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    print(f"[INFO] Connected to CARLA {client.get_server_version()}")
    world  = client.load_world(args.town)
    bp_lib = world.get_blueprint_library()

    # ── Weather: golden-hour cinematic ────────────────────────────────────────
    weather = carla.WeatherParameters(
        cloudiness            = 20.0,
        precipitation         = 0.0,
        sun_altitude_angle    = 28.0,   # low sun → long shadows
        sun_azimuth_angle     = 220.0,
        fog_density           = 5.0,
        fog_distance          = 80.0,
        wetness               = 10.0,
        wind_intensity        = 0.2,
    )
    world.set_weather(weather)

    # ── Spectator out of the way ──────────────────────────────────────────────
    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(
        carla.Location(x=0, y=0, z=200),
        carla.Rotation(pitch=-90)))

    actors_spawned = []

    try:
        # ── Spawn ego vehicle ─────────────────────────────────────────────────
        ego_bp = find_blueprint(
            bp_lib,
            ["vehicle.bmw.grandtourer",      # BMW closest in CARLA
             "vehicle.mercedes.coupe_2020",
             "vehicle.audi.a2"],
            "vehicle.tesla.model3",
        )
        ego_bp.set_attribute("role_name", "hero")
        if ego_bp.has_attribute("color"):
            ego_bp.set_attribute("color", "10,60,120")  # dark navy blue

        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        # Pick a straight-ish spawn point
        ego_transform = spawn_points[0]
        ego = world.try_spawn_actor(ego_bp, ego_transform)
        if ego is None:
            ego_transform = spawn_points[1]
            ego = world.spawn_actor(ego_bp, ego_transform)
        actors_spawned.append(ego)
        print(f"[INFO] Ego vehicle: {ego.type_id}")

        # ── RGB Camera ────────────────────────────────────────────────────────
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x",  str(W))
        cam_bp.set_attribute("image_size_y",  str(H))
        cam_bp.set_attribute("fov",           str(CAM_FOV))
        cam_bp.set_attribute("motion_blur_intensity", "0.45")
        cam_bp.set_attribute("motion_blur_max_distortion", "0.2")

        cam_transform = carla.Transform(
            carla.Location(x=CAM_X, y=CAM_Y, z=CAM_Z),
            carla.Rotation(pitch=CAM_PITCH, yaw=CAM_YAW))
        camera = world.spawn_actor(cam_bp, cam_transform, attach_to=ego)
        actors_spawned.append(camera)

        # ── LiDAR sensor ──────────────────────────────────────────────────────
        lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels",         "64")
        lidar_bp.set_attribute("range",            "80")
        lidar_bp.set_attribute("points_per_second","560000")
        lidar_bp.set_attribute("rotation_frequency","20")

        lidar_transform = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=LIDAR_Z))
        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego)
        actors_spawned.append(lidar)

        # ── NPC traffic ───────────────────────────────────────────────────────
        tm = client.get_trafficmanager(8000)
        tm.set_global_distance_to_leading_vehicle(2.5)
        tm.set_synchronous_mode(False)

        vehicle_bps  = bp_lib.filter("vehicle.*")
        walker_bps   = bp_lib.filter("walker.pedestrian.*")
        npc_vehicles = []
        npc_walkers  = []

        for sp in spawn_points[2:2 + N_VEHICLES]:
            v_bp = random.choice(list(vehicle_bps))
            v = world.try_spawn_actor(v_bp, sp)
            if v:
                v.set_autopilot(True, 8000)
                npc_vehicles.append(v)
                actors_spawned.append(v)

        walker_spawn_bps = []
        walker_spawns    = []
        for _ in range(N_WALKERS):
            loc = world.get_random_location_from_navigation()
            if loc:
                w_bp = random.choice(list(walker_bps))
                walker_spawn_bps.append(w_bp)
                walker_spawns.append(carla.Transform(loc))

        batch = [carla.command.SpawnActor(bp, sp)
                 for bp, sp in zip(walker_spawn_bps, walker_spawns)]
        results = client.apply_batch_sync(batch, True)
        for res in results:
            if not res.error:
                walker = world.get_actor(res.actor_id)
                if walker:
                    npc_walkers.append(walker)
                    actors_spawned.append(walker)

        # Add walker AI controllers
        ctrl_bp = bp_lib.find("controller.ai.walker")
        for w_actor in npc_walkers:
            ctrl = world.try_spawn_actor(ctrl_bp, carla.Transform(),
                                         attach_to=w_actor)
            if ctrl:
                actors_spawned.append(ctrl)
                ctrl.start()
                ctrl.go_to_location(world.get_random_location_from_navigation())
                ctrl.set_max_speed(1.4)

        # ── Let world settle ──────────────────────────────────────────────────
        print("[INFO] Letting scene settle (5 s) …")
        ego.set_autopilot(True, 8000)
        time.sleep(5)

        # ── Collect one camera frame ──────────────────────────────────────────
        cam_image   = [None]
        lidar_data  = [None]

        def on_cam(img):
            cam_image[0] = img

        def on_lidar(data):
            lidar_data[0] = data

        camera.listen(on_cam)
        lidar.listen(on_lidar)

        print("[INFO] Waiting for sensor frames …")
        timeout = time.time() + 15
        while (cam_image[0] is None or lidar_data[0] is None) \
                and time.time() < timeout:
            time.sleep(0.05)

        camera.stop()
        lidar.stop()

        if cam_image[0] is None:
            sys.exit("[ERROR] No camera frame received — is CARLA running?")

        # ── Convert raw image ─────────────────────────────────────────────────
        raw = np.frombuffer(cam_image[0].raw_data, dtype=np.uint8)
        raw = raw.reshape((H, W, 4))         # BGRA
        img = raw[:, :, :3].copy()           # BGR

        # ── Camera transform in world coords ──────────────────────────────────
        world_cam_tf = camera.get_transform()
        K = get_camera_intrinsics(W, H, CAM_FOV)

        # ── Draw LiDAR ────────────────────────────────────────────────────────
        if lidar_data[0] is not None:
            draw_lidar_dots(img, lidar_data[0], world_cam_tf, K,
                            LIDAR_COLOR, W, H, lidar.get_transform())

        # ── Get trajectory waypoints ──────────────────────────────────────────
        carla_map  = world.get_map()
        ego_wp     = carla_map.get_waypoint(ego.get_location())
        traj_wps   = [ego_wp]
        cur        = ego_wp
        for _ in range(TRAJ_STEPS):
            nexts = cur.next(TRAJ_STEP_M)
            if nexts:
                cur = nexts[0]
                traj_wps.append(cur)

        draw_trajectory(img, traj_wps, world_cam_tf, K, TRAJ_COLOR, W, H)

        # ── Draw bounding boxes ───────────────────────────────────────────────
        ego_loc = ego.get_location()

        for v in npc_vehicles:
            if v.get_location().distance(ego_loc) < 60:
                draw_bbox_3d(img, v, world_cam_tf, K,
                             BBOX_VEHICLE_COLOR, W, H)

        for w_actor in npc_walkers:
            if w_actor.get_location().distance(ego_loc) < 40:
                draw_bbox_3d(img, w_actor, world_cam_tf, K,
                             BBOX_PEDESTRIAN_COLOR, W, H)

        # ── Cinematic post-process (vignette + slight cool grade) ─────────────
        # Vignette
        rows, cols = img.shape[:2]
        X = np.linspace(-1, 1, cols)[np.newaxis, :]
        Y = np.linspace(-1, 1, rows)[:, np.newaxis]
        vignette = 1.0 - 0.55 * (X**2 + Y**2)
        vignette = np.clip(vignette, 0, 1)
        img = (img * vignette[:, :, np.newaxis]).astype(np.uint8)

        # Slight blue-teal colour grade
        img = img.astype(np.float32)
        img[:, :, 0] = np.clip(img[:, :, 0] * 1.04, 0, 255)  # boost blue
        img[:, :, 2] = np.clip(img[:, :, 2] * 0.94, 0, 255)  # reduce red
        img = img.astype(np.uint8)

        # ── Upper-left empty space: darken to make title text readable ────────
        # Gradient darkening — top-left quadrant only
        mask = np.zeros((H, W), dtype=np.float32)
        # Gradient: fully dark at (0,0), fades to 0 at (W/2, H/2)
        for row in range(H // 2):
            for col in range(W // 2):
                fade = 1.0 - (row / (H / 2)) * (col / (W / 2))
                mask[row, col] = fade * 0.55      # max 55% darkening

        img = img.astype(np.float32)
        img[:, :, 0] -= mask * img[:, :, 0]
        img[:, :, 1] -= mask * img[:, :, 1]
        img[:, :, 2] -= mask * img[:, :, 2]
        img = np.clip(img, 0, 255).astype(np.uint8)

        # ── Save ──────────────────────────────────────────────────────────────
        cv2.imwrite(str(out_path), img)
        print(f"[DONE] Saved → {out_path.resolve()}  ({W}×{H})")

    finally:
        print("[INFO] Cleaning up actors …")
        client.apply_batch([carla.command.DestroyActor(a)
                            for a in actors_spawned])
        time.sleep(1.0)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a cinematic CARLA title-slide screenshot.")
    parser.add_argument("--host",  default="127.0.0.1",
                        help="CARLA server host (default: 127.0.0.1)")
    parser.add_argument("--port",  default=2000, type=int,
                        help="CARLA server port (default: 2000)")
    parser.add_argument("--town",  default="Town10HD",
                        help="CARLA map to load (default: Town10HD)")
    parser.add_argument("--res",   default="1080p",
                        choices=RESOLUTIONS.keys(),
                        help="Output resolution (default: 1080p)")
    parser.add_argument("--out",   default="carla_title.png",
                        help="Output file path (default: carla_title.png)")
    args = parser.parse_args()
    main(args)

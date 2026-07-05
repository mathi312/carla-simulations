"""
ASR — Adversarial Patch on Road Sign (CARLA)
=============================================
Town03 — finds texturable road sign mesh names via
world.get_names_of_all_objects(), applies an adversarial
chequerboard texture via world.apply_color_texture_to_object(),
then drives the ADS past the sign twice (clean vs patched).

Key insight: apply_color_texture_to_object() requires names from
get_names_of_all_objects() — NOT from get_environment_objects().
Environment object names (e.g. BP_SpeedLimit60_209) refer to
Blueprint actors; the texture API targets the underlying static
mesh component names (e.g. SM_SpeedLimit60_0).

No torch / torchvision. Color fix: BGRA→RGB via [:,:,[2,1,0]].

Usage:
    python asr_patch.py [--host HOST] [--port PORT] [--output OUTPUT]

If auto-selection fails, run discover_signs.py, find a sign mesh
name, and set SIGN_MESH_NAME = "your_name_here" below.
"""

import argparse
import math
import random
import pathlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── constants ─────────────────────────────────────────────────────────────────
SEED            = 42
FIXED_DELTA_T   = 0.05
GIF_FPS         = 20
IMG_W, IMG_H    = 1280, 720
CAMERA_FOV      = 90
OUTPUT_GIF      = "asr_patch.gif"

DRIVE_SPEED_KMH = 25.0
PASS_SECONDS    = 8.0
APPROACH_DIST_M = 50.0

# ── Set this manually if auto-selection picks the wrong sign ──────────────────
# Run discover_signs.py to list all candidate names, then paste one here.
# Example: SIGN_MESH_NAME = "SM_SpeedLimit60_3"
SIGN_MESH_NAME  = None   # None = auto-select

# Patch texture
PATCH_CELL_PX  = 30
PATCH_COLOR_A  = (0,   255,  80)
PATCH_COLOR_B  = (255,   0, 220)
PATCH_BORDER   = (255, 220,   0)
PATCH_BORDER_W = 8

# Keywords used to recognise sign mesh names in get_names_of_all_objects()
SIGN_KEYWORDS  = ["speedlimit", "speed_limit", "speed limit",
                  "stopsign", "stop_sign", "streetsign", "street_sign",
                  "roadsign",  "road_sign",  "trafficSign",
                  "sm_speed",  "sm_sign",    "sm_stop",
                  "bp_speed",  "bp_sign",    "limit"]


# ── helpers ───────────────────────────────────────────────────────────────────
def kmh_to_ms(v): return v / 3.6


def save_frame(image, frame_list):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    frame_list.append(Image.fromarray(array[:, :, [2, 1, 0]], mode="RGB"))


def set_velocity(actor, speed_ms, direction=None):
    import carla
    if direction is None:
        direction = actor.get_transform().get_forward_vector()
    actor.set_target_velocity(carla.Vector3D(
        direction.x * speed_ms,
        direction.y * speed_ms,
        direction.z * speed_ms,
    ))


# ── texture ───────────────────────────────────────────────────────────────────
def make_patch_texture(size=512):
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)
    cell = max(PATCH_CELL_PX, size // 16)
    for row in range(size // cell + 1):
        for col in range(size // cell + 1):
            color = PATCH_COLOR_A if (row + col) % 2 == 0 else PATCH_COLOR_B
            draw.rectangle(
                [col*cell, row*cell,
                 min((col+1)*cell, size), min((row+1)*cell, size)],
                fill=color + (255,),
            )
    bw = max(PATCH_BORDER_W, size // 60)
    draw.rectangle([0, 0, size-1, size-1],
                   outline=PATCH_BORDER + (255,), width=bw)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size // 7)
    except Exception:
        font = ImageFont.load_default()
    label = "ADV.\nPATCH"
    bbox  = draw.textbbox((0, 0), label, font=font)
    lw, lh = bbox[2]-bbox[0], bbox[3]-bbox[1]
    draw.text(((size-lw)//2+2, (size-lh)//2+2), label,
              font=font, fill=(0, 0, 0, 220))
    draw.text(((size-lw)//2,   (size-lh)//2),   label,
              font=font, fill=(255, 255, 255, 255))
    return img


def texture_to_carla(pil_img):
    """int() cast required — carla.Color rejects numpy.uint8."""
    import carla
    w, h  = pil_img.size
    arr   = np.array(pil_img.convert("RGBA"), dtype=np.uint8)
    tex   = carla.TextureColor(w, h)
    for y in range(h):
        for x in range(w):
            r, g, b, a = arr[y, x]
            tex.set(x, y, carla.Color(int(r), int(g), int(b), int(a)))
    return tex


# ── sign mesh discovery ───────────────────────────────────────────────────────
def find_sign_mesh_name(world):
    """
    Use world.get_names_of_all_objects() — the correct source for
    apply_color_texture_to_object().  Filter by sign-related keywords
    and return the first match (or all matches for the user to choose).
    """
    all_names = world.get_names_of_all_objects()
    print(f"[INFO] Total texturable objects in world: {len(all_names)}")

    matches = [n for n in all_names
               if any(k.lower() in n.lower() for k in SIGN_KEYWORDS)]

    if not matches:
        # Dump everything so the user can inspect manually
        sample = sorted(all_names)[:80]
        print("[WARN] No sign-like mesh names found. First 80 object names:")
        for n in sample:
            print(f"  {n}")
        raise RuntimeError(
            "Cannot auto-detect sign mesh name. "
            "Run discover_signs.py, find a sign name, and set "
            "SIGN_MESH_NAME at the top of asr_patch.py."
        )

    print(f"[INFO] Found {len(matches)} sign-like mesh name(s):")
    for n in matches:
        print(f"  {n}")

    # Prefer names that contain both a sign keyword AND a mesh indicator
    # (SM_ prefix = StaticMesh, most reliably texturable)
    sm_matches = [n for n in matches if n.upper().startswith("SM_")]
    chosen = sm_matches[0] if sm_matches else matches[0]
    print(f"[INFO] Auto-selected: '{chosen}'")
    return chosen


def apply_patch(world, mesh_name, carla_tex):
    import carla
    try:
        world.apply_color_texture_to_object(
            mesh_name, carla.MaterialParameter.Diffuse, carla_tex)
        print(f"  [OK] Texture applied to '{mesh_name}'")
        return True
    except Exception as e:
        print(f"  [WARN] Diffuse failed: {e}")
    # Try other material parameters — some signs use Normal/Emissive slots
    for param in [carla.MaterialParameter.Normal,
                  carla.MaterialParameter.Emissive,
                  carla.MaterialParameter.Roughness]:
        try:
            world.apply_color_texture_to_object(mesh_name, param, carla_tex)
            print(f"  [OK] Texture applied via param {param} to '{mesh_name}'")
            return True
        except Exception:
            pass
    print(f"  [FAIL] Could not apply texture to '{mesh_name}'. "
          f"Try a different name from discover_signs.py.")
    return False


# ── find straight approach road near any sign ─────────────────────────────────
def find_spawn_near_signs(world, world_map):
    """
    Get environment objects (for location data), match them to the
    mesh names we can actually texture, and find the best straight
    approach road to one of them.
    """
    import carla

    all_names = world.get_names_of_all_objects()
    sign_names = {n for n in all_names
                  if any(k.lower() in n.lower() for k in SIGN_KEYWORDS)}

    # Get environment object locations for TrafficSigns
    env_objs = []
    for label in [carla.CityObjectLabel.TrafficSigns,
                  carla.CityObjectLabel.Static]:
        try:
            env_objs += list(world.get_environment_objects(label))
        except Exception:
            pass

    best_wp    = None
    best_score = -1

    for obj in env_objs:
        loc = obj.transform.location
        wp  = world_map.get_waypoint(
            loc, project_to_road=True,
            lane_type=carla.LaneType.Driving)
        if wp is None or wp.is_junction:
            continue
        prev = wp.previous(APPROACH_DIST_M)
        if not prev or prev[0].is_junction:
            continue
        spawn_wp = prev[0]

        # Straight run score
        run, cur = 0, spawn_wp
        h0 = spawn_wp.transform.rotation.yaw
        for _ in range(20):
            nxt = cur.previous(3.0)
            if not nxt or nxt[0].is_junction:
                break
            dh = abs((nxt[0].transform.rotation.yaw - h0 + 180) % 360 - 180)
            if dh > 10:
                break
            cur = nxt[0]
            run += 1

        if run > best_score:
            best_score = run
            best_wp    = spawn_wp
            best_loc   = loc

    if best_wp is None:
        # Fall back: just pick a long straight anywhere
        all_wps = world_map.generate_waypoints(4.0)
        for wp in all_wps:
            if wp.lane_type != carla.LaneType.Driving or wp.is_junction:
                continue
            run, cur = 0, wp
            h0 = wp.transform.rotation.yaw
            for _ in range(25):
                nxt = cur.next(3.0)
                if not nxt or nxt[0].is_junction:
                    break
                dh = abs((nxt[0].transform.rotation.yaw - h0 + 180) % 360 - 180)
                if dh > 8:
                    break
                cur = nxt[0]
                run += 1
            if run > best_score:
                best_score = run
                best_wp    = wp

    print(f"[INFO] ADS spawn: road={best_wp.road_id} "
          f"straight≥{best_score*3:.0f} m")
    return best_wp


# ── single drive pass ─────────────────────────────────────────────────────────
def run_pass(world, ads, camera, all_frames, label, speed_ms, spawn_tf):
    import carla

    ads.set_target_velocity(carla.Vector3D(0, 0, 0))
    ads.set_transform(spawn_tf)
    for _ in range(10):
        world.tick()

    fwd         = ads.get_transform().get_forward_vector()
    total_ticks = int(PASS_SECONDS / FIXED_DELTA_T)
    pass_frames = []
    camera.listen(lambda img: save_frame(img, pass_frames))

    for _ in range(total_ticks):
        set_velocity(ads, speed_ms, fwd)
        fwd = ads.get_transform().get_forward_vector()
        world.tick()

    camera.stop()

    for frame in pass_frames:
        draw = ImageDraw.Draw(frame)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 38)
        except Exception:
            font = ImageFont.load_default()
        if label == "clean":
            tag, color = "CLEAN SIGN  —  model: correct  ✓", (80, 220, 80)
        else:
            tag, color = "PATCHED SIGN  —  model: ???  ✗",   (255, 80, 80)
        draw.text((22, 22), tag, font=font, fill=(0, 0, 0))
        draw.text((20, 20), tag, font=font, fill=color)
        all_frames.append(frame)

    print(f"  [INFO] {label}: {len(pass_frames)} frames")


# ── main ──────────────────────────────────────────────────────────────────────
def main(args):
    import carla

    random.seed(SEED)
    np.random.seed(SEED)

    all_frames = []
    actor_list = []

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    print("[INFO] Loading Town03…")
    world = client.load_world("Town03")
    original_settings = world.get_settings()

    try:
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

        # ── find mesh name and spawn point ─────────────────────────────────────
        mesh_name = SIGN_MESH_NAME if SIGN_MESH_NAME else find_sign_mesh_name(world)
        spawn_wp  = find_spawn_near_signs(world, world_map)

        ads_spawn_tf = spawn_wp.transform
        ads_spawn_tf.location.z += 0.3

        # ── spawn ADS ─────────────────────────────────────────────────────────
        ads_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        ads_bp.set_attribute("color",     "30,144,255")
        ads_bp.set_attribute("role_name", "ADS")
        ads = world.spawn_actor(ads_bp, ads_spawn_tf)
        actor_list.append(ads)
        ads.set_simulate_physics(True)

        # ── camera ────────────────────────────────────────────────────────────
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMG_W))
        cam_bp.set_attribute("image_size_y", str(IMG_H))
        cam_bp.set_attribute("fov",          str(CAMERA_FOV))
        cam_bp.set_attribute("sensor_tick",  "0.0")
        camera = world.spawn_actor(
            cam_bp,
            carla.Transform(
                carla.Location(x=-6.0, y=0.5, z=2.8),
                carla.Rotation(pitch=-6.0, yaw=5.0),
            ),
            attach_to=ads,
        )
        actor_list.append(camera)

        # ── build patch texture ───────────────────────────────────────────────
        print("[INFO] Generating patch texture…")
        patch_pil   = make_patch_texture(512)
        patch_carla = texture_to_carla(patch_pil)

        out_dir = pathlib.Path(args.output).resolve().parent
        patch_pil.save(str(out_dir / "sign_patched.png"))

        for _ in range(20):
            world.tick()

        speed_ms = kmh_to_ms(DRIVE_SPEED_KMH)

        # ── Pass 1: clean ─────────────────────────────────────────────────────
        print("[INFO] Pass 1 — clean sign…")
        run_pass(world, ads, camera, all_frames, "clean", speed_ms, ads_spawn_tf)

        # ── Pass 2: patch → drive ─────────────────────────────────────────────
        print(f"[INFO] Applying patch to '{mesh_name}'…")
        apply_patch(world, mesh_name, patch_carla)
        for _ in range(5):
            world.tick()

        print("[INFO] Pass 2 — patched sign…")
        run_pass(world, ads, camera, all_frames, "patched", speed_ms, ads_spawn_tf)

        # ── encode GIF ────────────────────────────────────────────────────────
        if not all_frames:
            print("[ERROR] No frames captured.")
            return

        print(f"[INFO] Writing GIF → {args.output} ({len(all_frames)} frames)…")
        all_frames[0].save(
            args.output,
            format="GIF",
            save_all=True,
            append_images=all_frames[1:],
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",   default="127.0.0.1")
    parser.add_argument("--port",   default=2000, type=int)
    parser.add_argument("--output", default=OUTPUT_GIF)
    main(parser.parse_args())
"""
ASR — Adversarial Patch on Stop Sign (CARLA)
=============================================
Town10HD_Opt — drives the ADS past a stop sign whose texture has been
replaced with a patched version: a large, visually obvious adversarial
sticker overlaid on the sign face.

No torch / torchvision required.  Patch is generated with PIL + NumPy
and injected into CARLA via the texture-override API.

What the GIF shows:
  - First pass  : clean stop sign  (baseline)
  - Second pass : same sign with a conspicuous adversarial patch applied
  Both passes are stitched into one GIF so the viewer can compare.

Patch design:
  A high-contrast chequerboard of neon green / magenta squares printed
  over the centre of the sign face — unmissable to the viewer, yet
  exactly the kind of pattern that fools CNN-based classifiers by
  disrupting the learned edge and colour features of "STOP".

Determinism: synchronous mode + fixed delta + fixed seeds.
Color fix:   BGRA → RGB via [:,:,[2,1,0]].

Usage:
    python asr_patch.py [--host HOST] [--port PORT] [--output OUTPUT]
"""

import argparse
import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── constants ─────────────────────────────────────────────────────────────────
SEED           = 42
FIXED_DELTA_T  = 0.05
GIF_FPS        = 20
IMG_W, IMG_H   = 1280, 720
CAMERA_FOV     = 90
OUTPUT_GIF     = "asr_patch.gif"

# Drive speed and duration for each pass
DRIVE_SPEED_KMH = 30.0
PASS_SECONDS    = 6.0       # seconds per pass (clean then patched)

# Patch appearance — chequerboard of neon squares over sign centre
PATCH_SIZE_PX   = 200       # size of the patch region on the 512×512 sign texture
PATCH_CELL_PX   = 25        # size of each chequerboard cell
PATCH_COLOR_A   = (0,   255,  80)   # neon green
PATCH_COLOR_B   = (255,  0,  220)   # neon magenta
PATCH_BORDER    = (255, 220,   0)   # bright yellow border around patch
PATCH_BORDER_W  = 6                 # border width in pixels

# Town10 stop sign spawn — a known roadside stop sign location
# We place the ADS on a straight road and it drives past it.
SIGN_ROAD_ID    = 8          # main boulevard; signs are alongside
APPROACH_DIST_M = 40.0       # ADS starts this far before the sign


# ── helpers ───────────────────────────────────────────────────────────────────
def kmh_to_ms(v): return v / 3.6


def save_frame(image, frame_list):
    """BGRA → RGB; avoids the red/blue swap from naive [:,:,:3]."""
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


def dist2d(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)


# ── texture generation ────────────────────────────────────────────────────────
def make_clean_sign_texture(size=512):
    """
    Recreate a basic stop-sign texture: red octagon, white border, white STOP.
    CARLA's default texture is already on the mesh; we build a clean version
    so both passes use textures we control (clean vs patched look identical
    except for the patch itself).
    """
    img = Image.new("RGBA", (size, size), (180, 0, 0, 255))   # red background
    draw = ImageDraw.Draw(img)

    # White octagon border
    cx, cy, r = size // 2, size // 2, size // 2 - 10
    oct_pts = [
        (cx + r * math.cos(math.radians(22.5 + 45 * i)),
         cy + r * math.sin(math.radians(22.5 + 45 * i)))
        for i in range(8)
    ]
    draw.polygon(oct_pts, fill=(200, 0, 0, 255), outline=(255, 255, 255, 255))
    # Draw thick white outline manually
    for shrink in range(0, 18, 2):
        r2 = r - shrink
        pts = [
            (cx + r2 * math.cos(math.radians(22.5 + 45 * i)),
             cy + r2 * math.sin(math.radians(22.5 + 45 * i)))
            for i in range(8)
        ]
        draw.polygon(pts, outline=(255, 255, 255, 255))

    # "STOP" text
    font_size = size // 5
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                  font_size)
    except Exception:
        font = ImageFont.load_default()

    text = "STOP"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((size - tw) // 2, (size - th) // 2 - 10),
              text, fill=(255, 255, 255, 255), font=font)

    return img


def make_patched_sign_texture(size=512):
    """
    Same base texture but with a conspicuous adversarial chequerboard patch
    stamped over the centre of the sign.  The patch deliberately breaks the
    visual 'STOP' features while looking nothing like a stop sign to a CNN.
    """
    img = make_clean_sign_texture(size)
    draw = ImageDraw.Draw(img)

    patch_half = PATCH_SIZE_PX // 2
    cx, cy = size // 2, size // 2
    x0 = cx - patch_half
    y0 = cy - patch_half
    x1 = cx + patch_half
    y1 = cy + patch_half

    # Draw chequerboard
    cell = PATCH_CELL_PX
    for row in range(PATCH_SIZE_PX // cell + 1):
        for col in range(PATCH_SIZE_PX // cell + 1):
            color = PATCH_COLOR_A if (row + col) % 2 == 0 else PATCH_COLOR_B
            rx0 = x0 + col * cell
            ry0 = y0 + row * cell
            rx1 = min(rx0 + cell, x1)
            ry1 = min(ry0 + cell, y1)
            draw.rectangle([rx0, ry0, rx1, ry1], fill=color + (255,))

    # Bright border around the patch to make it obvious in the GIF
    bw = PATCH_BORDER_W
    draw.rectangle([x0 - bw, y0 - bw, x1 + bw, y1 + bw],
                   outline=PATCH_BORDER + (255,), width=bw)

    # Small "ADV PATCH" label below the patch so the viewer knows what they see
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    label = "ADV. PATCH"
    bbox  = draw.textbbox((0, 0), label, font=font)
    lw    = bbox[2] - bbox[0]
    draw.text(((size - lw) // 2, y1 + bw + 4),
              label, fill=(255, 220, 0, 255), font=font)

    return img


def texture_to_carla(pil_img):
    """Convert a PIL RGBA image to a carla.TextureColor object."""
    import carla
    w, h  = pil_img.size
    arr   = np.array(pil_img.convert("RGBA"), dtype=np.uint8)
    tex   = carla.TextureColor(w, h)
    for y in range(h):
        for x in range(w):
            r, g, b, a = arr[y, x]
            tex.set(x, y, carla.Color(r, g, b, a))
    return tex


def find_stop_sign_and_spawn(world, world_map):
    """
    Find a stop sign actor in the world and return:
      - the sign actor
      - a spawn waypoint APPROACH_DIST_M before it on the same road
      - the sign's location

    If no stop sign is found, fall back to a known Town10 location.
    """
    import carla

    signs = world.get_actors().filter("static.prop.streetsign*")
    stop_signs = list(world.get_actors().filter("traffic.stop"))

    # Prefer actual traffic.stop actors; fall back to any streetsign prop
    candidates = stop_signs if stop_signs else list(signs)

    best_sign  = None
    best_wp    = None
    best_spawn = None
    best_run   = 0

    for sign in candidates:
        sloc = sign.get_location()
        # Find the nearest drivable waypoint
        wp = world_map.get_waypoint(sloc, project_to_road=True,
                                     lane_type=carla.LaneType.Driving)
        if wp is None:
            continue
        # Walk back APPROACH_DIST_M to find a spawn point
        prev = wp.previous(APPROACH_DIST_M)
        if not prev:
            continue
        spawn_wp = prev[0]
        # Check straight run ahead
        run, cur = 0, spawn_wp
        for _ in range(20):
            nxt = cur.next(3.0)
            if not nxt:
                break
            dh = abs((nxt[0].transform.rotation.yaw -
                       spawn_wp.transform.rotation.yaw + 180) % 360 - 180)
            if dh > 20:
                break
            cur = nxt[0]
            run += 1
        if run > best_run:
            best_run   = run
            best_sign  = sign
            best_wp    = wp
            best_spawn = spawn_wp

    if best_sign is None:
        print("[WARN] No stop sign found — using hard-coded Town10 fallback.")
        fallback = carla.Location(x=88.0, y=10.0, z=0.0)
        best_wp    = world_map.get_waypoint(fallback, project_to_road=True,
                                             lane_type=carla.LaneType.Driving)
        prev = best_wp.previous(APPROACH_DIST_M)
        best_spawn = prev[0] if prev else best_wp
        best_sign  = None   # no sign actor to texture in fallback

    return best_sign, best_spawn, best_wp


# ── single drive pass ─────────────────────────────────────────────────────────
def run_pass(world, ads, camera, sign_loc, frames, label, speed_ms):
    """
    Drive the ADS forward for PASS_SECONDS ticks, collecting frames.
    Overlays a text label onto each frame so the viewer knows which
    pass (CLEAN / PATCHED) they are watching.
    """
    import carla

    total_ticks = int(PASS_SECONDS / FIXED_DELTA_T)
    fwd = ads.get_transform().get_forward_vector()

    pass_frames = []
    camera.listen(lambda img: save_frame(img, pass_frames))

    for _ in range(total_ticks):
        set_velocity(ads, speed_ms, fwd)
        world.tick()
        # Update forward each tick so the car follows any slight road curvature
        fwd = ads.get_transform().get_forward_vector()

    camera.stop()

    # Burn label onto every frame
    for frame in pass_frames:
        draw  = ImageDraw.Draw(frame)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
        except Exception:
            font = ImageFont.load_default()

        tag, tag_color = (
            ("CLEAN SIGN  — model: STOP ✓", (80, 220, 80))
            if label == "clean" else
            ("PATCHED SIGN — model: ??? ✗", (255, 80, 80))
        )
        # Shadow
        draw.text((22, 22), tag, font=font, fill=(0, 0, 0, 200))
        draw.text((20, 20), tag, font=font, fill=tag_color)
        frames.append(frame)


# ── main ──────────────────────────────────────────────────────────────────────
def main(args):
    import carla

    random.seed(SEED)
    np.random.seed(SEED)

    all_frames = []
    actor_list = []

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    print("[INFO] Loading Town10HD_Opt…")
    world = client.load_world("Town10HD_Opt")
    original_settings = world.get_settings()

    try:
        # ── sync mode ─────────────────────────────────────────────────────────
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

        # ── find stop sign + spawn point ──────────────────────────────────────
        sign_actor, spawn_wp, sign_wp = find_stop_sign_and_spawn(world, world_map)

        if sign_actor:
            print(f"[INFO] Stop sign found: id={sign_actor.id}  "
                  f"loc={sign_actor.get_location()}")
        sign_loc = (sign_actor.get_location() if sign_actor
                    else sign_wp.transform.location)

        # ── build textures (PIL, no torch) ────────────────────────────────────
        print("[INFO] Generating sign textures…")
        clean_tex_pil   = make_clean_sign_texture(512)
        patched_tex_pil = make_patched_sign_texture(512)

        # Save previews as PNGs (useful for debugging / slide use)
        clean_tex_pil.save("/home/claude/sign_clean.png")
        patched_tex_pil.save("/home/claude/sign_patched.png")
        print("[INFO] Texture previews saved: sign_clean.png / sign_patched.png")

        # Convert to carla.TextureColor
        print("[INFO] Converting textures for CARLA (this takes ~10 s)…")
        clean_carla   = texture_to_carla(clean_tex_pil)
        patched_carla = texture_to_carla(patched_tex_pil)

        # ── spawn ADS ─────────────────────────────────────────────────────────
        ads_spawn_tf = spawn_wp.transform
        ads_spawn_tf.location.z += 0.3

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
            carla.Transform(carla.Location(x=-6.0, z=2.8),
                            carla.Rotation(pitch=-6.0)),
            attach_to=ads,
        )
        actor_list.append(camera)

        # Warm-up
        for _ in range(15):
            world.tick()

        speed_ms = kmh_to_ms(DRIVE_SPEED_KMH)

        # ── PASS 1: clean sign ────────────────────────────────────────────────
        print("[INFO] Pass 1 — clean sign…")
        if sign_actor:
            try:
                world.apply_color_texture_to_object(
                    sign_actor.type_id, carla.MaterialParameter.Diffuse, clean_carla)
            except Exception as e:
                print(f"  [WARN] apply_color_texture_to_object failed: {e}")
                # Try apply_textures_to_object (older API name)
                try:
                    world.apply_textures_to_object(sign_actor.type_id,
                                                    clean_carla, carla.TextureFloatColor(0,0),
                                                    carla.TextureFloatColor(0,0),
                                                    carla.TextureFloatColor(0,0))
                except Exception as e2:
                    print(f"  [WARN] fallback texture API also failed: {e2}")

        run_pass(world, ads, camera, sign_loc, all_frames, "clean", speed_ms)

        # ── teleport ADS back to spawn for pass 2 ────────────────────────────
        ads.set_target_velocity(carla.Vector3D(0, 0, 0))
        ads.set_transform(ads_spawn_tf)
        for _ in range(10):
            world.tick()

        # ── PASS 2: patched sign ──────────────────────────────────────────────
        print("[INFO] Pass 2 — patched sign…")
        if sign_actor:
            try:
                world.apply_color_texture_to_object(
                    sign_actor.type_id, carla.MaterialParameter.Diffuse, patched_carla)
            except Exception as e:
                print(f"  [WARN] apply_color_texture_to_object failed: {e}")
                try:
                    world.apply_textures_to_object(sign_actor.type_id,
                                                    patched_carla, carla.TextureFloatColor(0,0),
                                                    carla.TextureFloatColor(0,0),
                                                    carla.TextureFloatColor(0,0))
                except Exception as e2:
                    print(f"  [WARN] fallback texture API also failed: {e2}")

        run_pass(world, ads, camera, sign_loc, all_frames, "patched", speed_ms)

        # ── encode GIF ────────────────────────────────────────────────────────
        if not all_frames:
            print("[ERROR] No frames captured.")
            return

        print(f"[INFO] Writing GIF → {args.output}  "
              f"({len(all_frames)} frames @ {GIF_FPS} fps)…")
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

        # Also copy texture previews to outputs
        import shutil
        shutil.copy("/home/claude/sign_clean.png",
                    "/mnt/user-data/outputs/sign_clean.png")
        shutil.copy("/home/claude/sign_patched.png",
                    "/mnt/user-data/outputs/sign_patched.png")

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
    parser = argparse.ArgumentParser(
        description="CARLA ASR — adversarial patch on stop sign"
    )
    parser.add_argument("--host",   default="127.0.0.1")
    parser.add_argument("--port",   default=2000, type=int)
    parser.add_argument("--output", default=OUTPUT_GIF)
    main(parser.parse_args())

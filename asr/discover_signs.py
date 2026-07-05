"""
Discover texturable road sign mesh names in CARLA.

Uses world.get_names_of_all_objects() — the correct source of names
for world.apply_color_texture_to_object(). Environment object names
from get_environment_objects() do NOT work with the texture API.

Run this first, look for names containing "speed", "sign", "limit",
"stop" etc., then set SIGN_MESH_NAME in asr_patch.py.

Usage:
    python discover_signs.py [--host HOST] [--port PORT] [--town TOWN]
"""
import argparse
import carla

parser = argparse.ArgumentParser()
parser.add_argument("--host",   default="127.0.0.1")
parser.add_argument("--port",   default=2000, type=int)
parser.add_argument("--town",   default="Town03")
parser.add_argument("--filter", default="", help="Filter substring (e.g. 'speed', 'sign')")
args = parser.parse_args()

client = carla.Client(args.host, args.port)
client.set_timeout(20.0)
world  = client.load_world(args.town)

print(f"\n── Texturable object names in {args.town} ─────────────────────────")
names = world.get_names_of_all_objects()
print(f"Total objects: {len(names)}")

keywords = args.filter.lower().split(",") if args.filter else \
           ["speed","sign","stop","limit","traffic","road","street","km","mph"]

matches = [n for n in names if any(k in n.lower() for k in keywords)]
print(f"Matches for {keywords}: {len(matches)}\n")
for n in sorted(matches):
    print(f"  {n}")

if not matches:
    print("No matches — printing first 60 names to help narrow down:")
    for n in sorted(names)[:60]:
        print(f"  {n}")

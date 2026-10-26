import os
import glob

base = "S:/edited vid/viral shorts"
print(f"{'Folder Name':<32} | {'Shorts (MP4)':<14} | {'JSON Metadata':<14} | {'SUMMARY.md'}")
print("-" * 75)

total_shorts = 0
for d in sorted(os.listdir(base)):
    p = os.path.join(base, d)
    if os.path.isdir(p) and not d.startswith((".", "__", "frames", "visual", "vsm")):
        mp4s = glob.glob(os.path.join(p, "*.mp4"))
        jsons = glob.glob(os.path.join(p, "*.json"))
        has_sum = os.path.exists(os.path.join(p, "SUMMARY.md"))
        total_shorts += len(mp4s)
        print(f"{d:<32} | {len(mp4s):<14} | {len(jsons):<14} | {'YES' if has_sum else 'NO'}")

print("-" * 75)
print(f"TOTAL VIRAL SHORTS GENERATED: {total_shorts}")

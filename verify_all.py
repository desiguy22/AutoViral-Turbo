import os
import glob

base = "S:/edited vid/viral shorts"
print("================================================================================")
print(f"{'Folder Name':<28} | {'Shorts':<6} | {'Total Size':<10} | {'Sample Short Titles'}")
print("================================================================================")

total_shorts = 0
total_bytes = 0

for d in sorted(os.listdir(base)):
    p = os.path.join(base, d)
    if os.path.isdir(p) and not d.startswith((".", "__", "frames", "visual", "vsm")):
        mp4s = sorted(glob.glob(os.path.join(p, "*.mp4")))
        size = sum(os.path.getsize(f) for f in mp4s)
        total_shorts += len(mp4s)
        total_bytes += size
        samples = ", ".join(f'"{os.path.splitext(os.path.basename(f))[0]}"' for f in mp4s[:2])
        if len(mp4s) > 2:
            samples += f", +{len(mp4s)-2} more"
        print(f"{d:<28} | {len(mp4s):<6} | {size/(1024*1024):<7.1f} MB | {samples}")

print("================================================================================")
print(f"TOTAL: {total_shorts} Viral Shorts | {total_bytes/(1024*1024*1024):.2f} GB rendered")
print("================================================================================")

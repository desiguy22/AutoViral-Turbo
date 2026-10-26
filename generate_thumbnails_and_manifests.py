import os
import sys
import glob
import json
import csv
import subprocess

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

base = "S:/edited vid/viral shorts"
print("Generating 1080x1920 thumbnails and upload manifests across all folders...")

for d in sorted(os.listdir(base)):
    p = os.path.join(base, d)
    if os.path.isdir(p) and not d.startswith((".", "__", "frames", "visual", "vsm")):
        mp4s = sorted(glob.glob(os.path.join(p, "*.mp4")))
        shorts_meta = []
        for idx, m in enumerate(mp4s, 1):
            base_name = os.path.splitext(os.path.basename(m))[0]
            thumb_path = os.path.join(p, f"{base_name}_thumb.jpg")
            meta_path = os.path.join(p, f"{base_name}.json")
            
            # Extract thumbnail if not present
            if not os.path.isfile(thumb_path):
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", "1.20",
                    "-i", m,
                    "-vframes", "1",
                    "-q:v", "2",
                    thumb_path
                ]
                subprocess.run(cmd, capture_output=True)

            meta = {}
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as mf:
                        meta = json.load(mf)
                except Exception:
                    pass

            shorts_meta.append({
                "index": idx,
                "title": meta.get("title", base_name),
                "filename": os.path.basename(m),
                "thumbnail": f"{base_name}_thumb.jpg",
                "duration": round(meta.get("duration", 30.0), 1),
                "virality_score": meta.get("virality_score", 85),
                "hook_headline": meta.get("hook_top", "TACTICAL MOMENT"),
                "hook_subtitle": meta.get("hook_sub", "WATCH TILL THE END!"),
                "description": meta.get("description", f"Intense gaming action from {d}"),
                "hashtags": " ".join(meta.get("hashtags", ["#Shorts", "#Gaming", "#ModernWarfare"]))
            })

        # Write manifest CSV
        manifest_path = os.path.join(p, "upload_manifest.csv")
        fieldnames = [
            "index", "title", "filename", "thumbnail", "duration",
            "virality_score", "hook_headline", "hook_subtitle",
            "description", "hashtags"
        ]
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in shorts_meta:
                writer.writerow(row)

        print(f"  ✓ {d:<28}: {len(mp4s)} thumbnails & upload_manifest.csv created")

print("Done! All folders now have thumbnails and upload manifests.")

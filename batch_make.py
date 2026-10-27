#!/usr/bin/env python3
"""Batch generator for AutoViral Turbo.

Iterates through source videos, extracting 10 high-retention viral shorts
per video into separate organized folders with full-bleed 9:16 smart-crop,
word-level neon karaoke subtitles, and 2s hook headline cards.
"""
import os
import sys
import time
import glob
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from vsm import pipeline

def run_batch(video_paths, num_shorts="auto", out_base=None, force=False):
    if out_base is None:
        out_base = "S:/edited vid/viral shorts"
    os.makedirs(out_base, exist_ok=True)

    print(f"================================================================")
    print(f"🚀 AUTOVIRAL TURBO - BATCH PROCESSING")
    print(f"📹 Videos queued: {len(video_paths)}")
    print(f"🎯 Target shorts per video: {num_shorts}")
    print(f"📂 Output base directory: {out_base}")
    print(f"🔄 Force regeneration: {force}")
    print(f"================================================================\n")

    overall_start = time.time()
    results = {}

    for idx, vid in enumerate(video_paths, 1):
        vname = os.path.splitext(os.path.basename(vid))[0]
        print(f"\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(f"[{idx}/{len(video_paths)}] STARTING: {vname}")
        print(f"Path: {vid}")
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")

        v_start = time.time()
        try:
            shorts = pipeline.process_video(
                src_video=vid,
                output_dir=out_base,
                num_shorts=num_shorts,
                min_len=20.0,
                max_len=45.0,
                target_len=32.0,
                tracker="auto",
                use_local_llm=True,
                force=force
            )
            v_elapsed = time.time() - v_start
            results[vname] = {"status": "SUCCESS", "count": len(shorts), "time_sec": v_elapsed}
            print(f"\n✅ Completed {vname} in {v_elapsed/60.0:.1f} minutes ({len(shorts)} shorts)")
        except Exception as e:
            v_elapsed = time.time() - v_start
            results[vname] = {"status": "FAILED", "error": str(e), "time_sec": v_elapsed}
            print(f"\n❌ FAILED {vname}: {e}")
            import traceback
            traceback.print_exc()

        # Free GPU memory between videos
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - overall_start
    print("\n================================================================")
    print("🏁 BATCH COMPLETE SUMMARY")
    print(f"Total processing time: {total_time/60.0:.1f} minutes")
    for name, r in results.items():
        status = r["status"]
        if status == "SUCCESS":
            print(f"  ✓ {name}: {r['count']} shorts ({r['time_sec']/60.0:.1f} min)")
        else:
            print(f"  ✗ {name}: ERROR ({r.get('error', 'unknown')})")
    print("================================================================\n")

if __name__ == "__main__":
    force = False
    num_shorts = "auto"
    args = []
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--force":
            force = True
        elif arg in ("--num", "-n") and i + 1 < len(sys.argv):
            val = sys.argv[i + 1]
            if val.isdigit():
                num_shorts = "auto" if int(val) <= 0 else int(val)
            else:
                num_shorts = "auto" if val.lower() in ("auto", "all") else val
            i += 1
        elif arg.startswith("--num="):
            val = arg.split("=", 1)[1]
            if val.isdigit():
                num_shorts = "auto" if int(val) <= 0 else int(val)
            else:
                num_shorts = "auto" if val.lower() in ("auto", "all") else val
        else:
            args.append(arg)
        i += 1

    if args:
        targets = args
    else:
        # Default order: finish codmw4_7 extract, then shorter ones first (4, 6), then longer (1, 2, 5)
        # Note: codmw4_3 already has 10 shorts, so it will instantly verify and skip
        source_dir = "S:/edited vid"
        priority_order = [
            "codmw4_3.mp4",
            "codmw4_7 extract.mp4",
            "codmw4_4.mp4",
            "codmw4_6 jamming station.mp4",
            "codmw4_2.mp4",
            "codmw4_1.mp4",
            "codmw4_5.mp4",
        ]
        targets = [os.path.join(source_dir, f) for f in priority_order if os.path.exists(os.path.join(source_dir, f))]

    run_batch(targets, num_shorts=num_shorts, force=force)

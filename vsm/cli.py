"""Command Line Interface for Viral Shorts Maker.

Usage:
    python -m vsm.cli "S:/edited vid/codmw4_3.mp4"
    python -m vsm.cli "S:/edited vid/codmw4_3.mp4" --tracker compare
    python -m vsm.cli "S:/edited vid/codmw4_3.mp4" --num 5 --out "S:/edited vid/viral_shorts"
"""
import argparse
import sys
import os

from . import pipeline

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    
    parser = argparse.ArgumentParser(
        description="Viral Shorts Maker - Turn any video into high-engagement Shorts/Reels"
    )
    parser.add_argument("video", help="Path to input video file (e.g. S:/edited vid/codmw4_3.mp4)")
    parser.add_argument("-n", "--num", default="auto", help="Number of viral shorts to generate or 'auto' for uncapped detection (default: auto)")
    parser.add_argument("-o", "--out", default=None, help="Output directory (default: <video_dir>/viral_shorts/<video_name>)")
    parser.add_argument("--min-len", type=float, default=20.0, help="Minimum clip duration in seconds (default: 20)")
    parser.add_argument("--max-len", type=float, default=45.0, help="Maximum clip duration in seconds (default: 45)")
    parser.add_argument("--target-len", type=float, default=32.0, help="Target clip duration in seconds (default: 32)")
    parser.add_argument("--whisper-model", default="base", help="Whisper model size (default: base)")
    parser.add_argument("--layout", choices=["auto", "blur_padding", "smart_crop"], default="auto", help="Framing layout")
    parser.add_argument("--tracker", choices=["auto", "yolo", "opencv", "compare"], default="auto",
                        help="Subject tracking engine or compare phase (default: auto)")
    parser.add_argument("--no-local-llm", action="store_true", help="Disable local GPU neural LLM (use pure heuristics)")
    parser.add_argument("--suffix", default="", help="Optional suffix to append to filenames (e.g. _new)")
    
    args = parser.parse_args()
    
    num_arg = args.num
    if isinstance(num_arg, str) and num_arg.strip().isdigit():
        num_val = int(num_arg)
        num_shorts = "auto" if num_val <= 0 else num_val
    elif str(num_arg).strip().lower() in ("auto", "none", "0"):
        num_shorts = "auto"
    else:
        num_shorts = num_arg

    forced_layout = None if args.layout == "auto" else args.layout
    
    pipeline.process_video(
        src_video=args.video,
        output_dir=args.out,
        num_shorts=num_shorts,
        min_len=args.min_len,
        max_len=args.max_len,
        target_len=args.target_len,
        whisper_model=args.whisper_model,
        forced_layout=forced_layout,
        tracker=args.tracker,
        use_local_llm=not args.no_local_llm,
        filename_suffix=args.suffix
    )

if __name__ == "__main__":
    main()

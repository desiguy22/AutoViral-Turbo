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
    parser.add_argument("-n", "--num", type=int, default=5, help="Number of viral shorts to generate (default: 5)")
    parser.add_argument("-o", "--out", default=None, help="Output directory (default: <video_dir>/viral_shorts/<video_name>)")
    parser.add_argument("--min-len", type=float, default=20.0, help="Minimum clip duration in seconds (default: 20)")
    parser.add_argument("--max-len", type=float, default=45.0, help="Maximum clip duration in seconds (default: 45)")
    parser.add_argument("--target-len", type=float, default=32.0, help="Target clip duration in seconds (default: 32)")
    parser.add_argument("--whisper-model", default="base", help="Whisper model size (default: base)")
    parser.add_argument("--layout", choices=["auto", "blur_padding", "smart_crop"], default="auto", help="Framing layout")
    parser.add_argument("--tracker", choices=["auto", "yolo", "opencv", "compare"], default="auto",
                        help="Subject tracking engine or compare phase (default: auto)")
    parser.add_argument("--no-local-llm", action="store_true", help="Disable local GPU neural LLM (use pure heuristics)")
    
    args = parser.parse_args()
    
    forced_layout = None if args.layout == "auto" else args.layout
    
    pipeline.process_video(
        src_video=args.video,
        output_dir=args.out,
        num_shorts=args.num,
        min_len=args.min_len,
        max_len=args.max_len,
        target_len=args.target_len,
        whisper_model=args.whisper_model,
        forced_layout=forced_layout,
        tracker=args.tracker,
        use_local_llm=not args.no_local_llm
    )

if __name__ == "__main__":
    main()

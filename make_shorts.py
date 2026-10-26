#!/usr/bin/env python3
"""Viral Shorts / Reel Maker - Root Launcher.

Commands:
    python make_shorts.py "S:/edited vid/codmw4_3.mp4"
    python make_shorts.py "S:/edited vid/codmw4_3.mp4" --num 5
    python make_shorts.py --web
"""
import sys
import os

# Ensure package is on sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

if __name__ == "__main__":
    if "--web" in sys.argv:
        from vsm.web import start_server
        port = 8501
        for i, arg in enumerate(sys.argv):
            if arg == "--port" and i + 1 < len(sys.argv):
                port = int(sys.argv[i + 1])
        start_server(port)
    else:
        from vsm.cli import main
        main()

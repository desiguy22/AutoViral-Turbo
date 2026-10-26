"""High-performance FFmpeg video renderer for vertical Shorts and Reels.

Supports:
- Dynamic smart crop (9:16 subject tracking) OR dual blurred top/bottom layout
- 2.0s Attention-grabbing Hook card at top (headline + urgency badge)
- Animated word-by-word karaoke subtitles burned in via libass
- Fast NVENC GPU hardware acceleration with libx264 CPU fallback
- Faststart MP4 for instant streaming on YouTube & Instagram
"""
import os
import sys
import subprocess
import time

from . import media

def escape_ass_path(path):
    """Format path for FFmpeg ass filter on Windows."""
    abs_p = os.path.abspath(path).replace("\\", "/").replace(":", r"\:").replace(",", r"\,")
    return f"filename='{abs_p}'"

def sanitize_drawtext(s):
    """Sanitize text for FFmpeg drawtext filter."""
    return s.replace("'", "’").replace(":", " - ").replace(",", " ")

def build_filter_graph(src_probe, layout, crop_x_pct, hook_top, hook_sub, ass_path):
    in_w = src_probe["width"]
    in_h = src_probe["height"]
    
    ht = sanitize_drawtext(hook_top)
    hs = sanitize_drawtext(hook_sub)
    ass_escaped = escape_ass_path(ass_path)
    
    # Modern high-contrast hook card (Urgency Pill + Bold Headline)
    hook_filters = (
        f"drawtext=text='{hs}':font='Impact':fontsize=44:fontcolor=white:"
        f"box=1:boxcolor=red@0.90:boxborderw=12:borderw=4:bordercolor=black:x=(w-text_w)/2:y=190:enable='between(t,0,2.5)',"
        f"drawtext=text='{ht}':font='Impact':fontsize=56:fontcolor=white:"
        f"borderw=6:bordercolor=black:shadowx=4:shadowy=4:x=(w-text_w)/2:y=275:enable='between(t,0,2.5)',"
        f"ass={ass_escaped}"
    )

    if layout == "smart_crop":
        crop_w = int(in_h * 9 / 16)
        if crop_w > in_w:
            crop_w = in_w
        raw_x = int(crop_x_pct * in_w - crop_w / 2.0)
        crop_x = max(0, min(in_w - crop_w, raw_x))
        
        fg = (
            f"[0:v]crop={crop_w}:{in_h}:{crop_x}:0,scale=1080:1920,setsar=1,"
            f"{hook_filters}[v]"
        )
    else:
        # Enhanced cinematic blurred top/bottom with larger 1080x720 centered video
        fg = (
            "[0:v]split=2[in_bg][in_fg];"
            "[in_bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:5,eq=brightness=0.04:saturation=1.4,setsar=1[bg];"
            "[in_fg]scale=1080:720,setsar=1[fg];"
            "[bg][fg]overlay=x=0:y=600,"
            "drawbox=y=597:color=white@0.4:width=1080:height=3:t=fill,"
            "drawbox=y=1320:color=white@0.4:width=1080:height=3:t=fill,"
            f"{hook_filters}[v]"
        )
        
    return fg


def render_short(src_video, start_t, end_t, layout, crop_x_pct, hook_top, hook_sub, ass_path, out_mp4):
    """Render a single vertical Short via FFmpeg."""
    t0 = time.time()
    probe_info = media.probe(src_video)
    dur = round(end_t - start_t, 2)
    
    fg = build_filter_graph(probe_info, layout, crop_x_pct, hook_top, hook_sub, ass_path)
    
    # Try NVENC first, fallback to libx264
    encoders_to_try = [
        ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "19", "-b:v", "14M"],
        ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    ]
    
    for v_enc in encoders_to_try:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_t:.3f}",
            "-t", f"{dur:.3f}",
            "-i", src_video,
            "-filter_complex", fg,
            "-map", "[v]",
            "-map", "0:a:0",
            *v_enc,
            "-c:a", "aac", "-b:a", "256k",
            "-movflags", "+faststart",
            out_mp4
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            elapsed = time.time() - t0
            size_mb = os.path.getsize(out_mp4) / (1024 * 1024)
            return {
                "success": True,
                "elapsed": elapsed,
                "size_mb": size_mb,
                "out_path": out_mp4
            }

    raise RuntimeError(f"FFmpeg render failed for {out_mp4}:\n{res.stderr[-1000:]}")

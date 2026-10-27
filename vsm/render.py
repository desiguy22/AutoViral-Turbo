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

def ensure_audio_assets():
    """Ensure tension drone bed and hook card bass drop sound effects exist in assets/."""
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    bass_drop_path = os.path.join(assets_dir, "bass_drop.wav")
    tension_bed_path = os.path.join(assets_dir, "tension_bed.wav")
    
    import numpy as np
    import wave
    
    rate = 44100
    if not os.path.isfile(bass_drop_path) or os.path.getsize(bass_drop_path) < 1000:
        dur = 2.0
        t = np.linspace(0, dur, int(rate * dur), endpoint=False)
        freq = 32.0 + (95.0 - 32.0) * ((dur - t) / dur) ** 2
        phase = 2 * np.pi * np.cumsum(freq) / rate
        env = np.minimum(t / 0.03, 1.0) * np.exp(-2.2 * t)
        signal = (np.sin(phase) + 0.3 * np.sin(phase * 0.5)) * env * 0.8
        audio16 = (signal * 32767).astype(np.int16)
        with wave.open(bass_drop_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(audio16.tobytes())
            
    if not os.path.isfile(tension_bed_path) or os.path.getsize(tension_bed_path) < 1000:
        dur = 60.0
        t = np.linspace(0, dur, int(rate * dur), endpoint=False)
        lfo = 0.5 + 0.5 * np.sin(2 * np.pi * 0.25 * t)
        drone = 0.45 * np.sin(2 * np.pi * 55.0 * t) + 0.25 * np.sin(2 * np.pi * 110.0 * t) * (0.8 + 0.2 * lfo) + 0.2 * np.sin(2 * np.pi * 82.41 * t)
        noise = np.random.normal(0, 0.05, len(t))
        k = 100
        smooth_noise = np.convolve(noise, np.ones(k)/k, mode="same")
        signal = (drone + smooth_noise * 1.5) * 0.4
        audio16 = (signal * 32767).astype(np.int16)
        with wave.open(tension_bed_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(audio16.tobytes())
            
    return bass_drop_path, tension_bed_path


def build_filter_graph(src_probe, layout, crop_x_pct, hook_top, hook_sub, ass_path, peak_t=None):
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
        v_base = f"[0:v]crop={crop_w}:{in_h}:{crop_x}:0,scale=1080:1920,setsar=1"
    else:
        # Enhanced cinematic blurred top/bottom with larger 1080x720 centered video
        v_base = (
            "[0:v]split=2[in_bg][in_fg];"
            "[in_bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:5,eq=brightness=0.04:saturation=1.4,setsar=1[bg];"
            "[in_fg]scale=1080:720,setsar=1[fg];"
            "[bg][fg]overlay=x=0:y=600,"
            "drawbox=y=597:color=white@0.4:width=1080:height=3:t=fill,"
            "drawbox=y=1320:color=white@0.4:width=1080:height=3:t=fill"
        )
        
    # Dynamic 1.15x reticle punch zoom centered at crosshair around combat peak
    if peak_t is not None and peak_t > 0:
        punch_s = max(0.0, peak_t - 0.25)
        punch_e = punch_s + 0.65
        fg = (
            f"{v_base}[base];"
            f"[base]split=2[m_v][z_v];"
            f"[z_v]crop=939:1669:70:125,scale=1080:1920[zoomed_v];"
            f"[m_v][zoomed_v]overlay=0:0:enable='between(t,{punch_s:.2f},{punch_e:.2f})'[punched];"
            f"[punched]{hook_filters}[v]"
        )
    else:
        fg = f"{v_base},{hook_filters}[v]"
        
    return fg


def render_short(src_video, start_t, end_t, layout, crop_x_pct, hook_top, hook_sub, ass_path, out_mp4, peak_t=None):
    """Render a single vertical Short via FFmpeg with dynamic punch-zoom and auto-ducked tension sound."""
    t0 = time.time()
    probe_info = media.probe(src_video)
    dur = round(end_t - start_t, 2)
    has_audio = probe_info.get("has_audio", False)
    
    fg = build_filter_graph(probe_info, layout, crop_x_pct, hook_top, hook_sub, ass_path, peak_t=peak_t)
    
    # Ensure audio enhancement assets exist
    bass_path, tension_path = ensure_audio_assets()
    has_audio_assets = os.path.isfile(bass_path) and os.path.isfile(tension_path)

    # Audio ducking / mixing filter complex
    if has_audio and has_audio_assets:
        # Music ducks under speech (-18 dB), surges back during pure combat. Bass drop on hook card.
        audio_filter = (
            f"[0:a:0]asplit=2[gameplay][speech_ref];"
            f"[1:a]atrim=0:{dur:.2f},asetpts=PTS-STARTPTS,volume=0.32[bgm_in];"
            f"[bgm_in][speech_ref]sidechaincompress=threshold=0.08:ratio=8:attack=25:release=350[ducked_bgm];"
            f"[2:a]atrim=0:2.0,asetpts=PTS-STARTPTS,volume=0.75[sfx];"
            f"[gameplay][ducked_bgm][sfx]amix=inputs=3:duration=first:weights=1.0 0.8 1.0,"
            f"loudnorm=I=-14:TP=-1.0:LRA=11[a]"
        )
        combined_fg = f"{fg};{audio_filter}"
        audio_inputs = ["-i", tension_path, "-i", bass_path]
        audio_maps = ["-map", "[a]"]
    elif (not has_audio) and has_audio_assets:
        # Video has no audio track: synthesize tension bed and bass drop directly
        audio_filter = (
            f"[1:a]atrim=0:{dur:.2f},asetpts=PTS-STARTPTS,volume=0.45[bgm_in];"
            f"[2:a]atrim=0:2.0,asetpts=PTS-STARTPTS,volume=0.85[sfx];"
            f"[bgm_in][sfx]amix=inputs=2:duration=first:weights=1.0 1.0,"
            f"loudnorm=I=-14:TP=-1.0:LRA=11[a]"
        )
        combined_fg = f"{fg};{audio_filter}"
        audio_inputs = ["-i", tension_path, "-i", bass_path]
        audio_maps = ["-map", "[a]"]
    elif has_audio:
        combined_fg = fg
        audio_inputs = []
        audio_maps = ["-map", "0:a:0", "-af", "loudnorm=I=-14:TP=-1.0:LRA=11"]
    else:
        combined_fg = fg
        audio_inputs = ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]
        audio_maps = ["-map", "1:a", "-t", f"{dur:.3f}"]

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
            *audio_inputs,
            "-filter_complex", combined_fg,
            "-map", "[v]",
            *audio_maps,
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

    # If sidechain audio setup failed, fallback to basic audio
    fb_inputs = [] if has_audio else ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]
    fb_maps = ["-map", "0:a:0", "-af", "loudnorm=I=-14:TP=-1.0:LRA=11"] if has_audio else ["-map", "1:a", "-t", f"{dur:.3f}"]
    for v_enc in encoders_to_try:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_t:.3f}",
            "-t", f"{dur:.3f}",
            "-i", src_video,
            *fb_inputs,
            "-filter_complex", fg,
            "-map", "[v]",
            *fb_maps,
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


def extract_thumbnail(video_path, out_thumb_path, timestamp_sec=1.2):
    """Extract a high-contrast 1080x1920 poster frame from the hook window."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{timestamp_sec:.2f}",
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "2",
        out_thumb_path
    ]
    res = subprocess.run(cmd, capture_output=True)
    return os.path.isfile(out_thumb_path)

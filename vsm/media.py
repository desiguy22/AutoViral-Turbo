"""Thin ffmpeg/ffprobe helpers. Everything heavy goes through ffmpeg so any codec/length works."""
import json
import os
import subprocess
import wave

import numpy as np


def run(cmd, cwd=None):
    """Run a command, raise with the stderr tail on failure."""
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        raise RuntimeError(f"command failed ({res.returncode}): {' '.join(map(str, cmd[:6]))} ...\n{res.stderr[-2000:]}")
    return res


def probe(path):
    """Return {duration, width, height, fps, has_audio} for a media file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    res = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path])
    info = json.loads(res.stdout)
    v = next((s for s in info["streams"] if s.get("codec_type") == "video"), None)
    if v is None:
        raise ValueError(f"no video stream in {path}")
    num, den = (v.get("avg_frame_rate") or v.get("r_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    if fps <= 0 or fps > 240:
        fps = 30.0
    w, h = int(v["width"]), int(v["height"])
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    if abs(rot) in (90, 270):
        w, h = h, w
    return {
        "duration": float(info["format"].get("duration") or v.get("duration") or 0),
        "width": w,
        "height": h,
        "fps": fps,
        "has_audio": any(s.get("codec_type") == "audio" for s in info["streams"]),
    }


def extract_audio(src, wav_path, sr=16000):
    """Mono 16 kHz PCM wav (what Whisper wants), cached."""
    if os.path.isfile(wav_path) and os.path.getsize(wav_path) > 1000:
        return wav_path
    tmp = wav_path + ".part.wav"
    run(["ffmpeg", "-y", "-v", "error", "-i", src, "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", tmp])
    os.replace(tmp, wav_path)
    return wav_path


def read_wav(wav_path):
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return data, sr


def iter_small_frames(src, fps=1.0, width=160, height=90, start=None, duration=None):
    """Yield (t, gray uint8 HxW) frames decoded by ffmpeg with hardware acceleration."""
    # Fast seek using -ss before -i if start is provided
    cmd = ["ffmpeg", "-v", "error", "-hwaccel", "cuda"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", src, "-an"]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vf", f"fps={fps},scale={width}:{height}:flags=fast_bilinear,format=gray",
            "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
            
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n = width * height
    i = 0
    t0 = start or 0.0
    try:
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            yield t0 + i / fps, np.frombuffer(buf, dtype=np.uint8).reshape(height, width)
            i += 1
    finally:
        proc.stdout.close()
        proc.wait()


def extract_frame(src, t, out_jpg, width=None):
    vf = ["-vf", f"scale={width}:-2"] if width else []
    run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", src, "-frames:v", "1", *vf, "-q:v", "3", out_jpg])
    return out_jpg

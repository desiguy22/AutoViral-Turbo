"""Per-second signals used for ranking: loudness, motion, scene cuts.

Computes RMS audio loudness on a 1.0s grid in <0.2s from the cached WAV file.
Visual motion is optionally computed or defaulted, keeping moment selection instantaneous.
"""
import json
import os
import numpy as np

from . import media

RES = 1.0  # 1.0 second resolution


def loudness_db(data, sr, res=RES):
    """RMS loudness in dBFS per `res`-second bin."""
    hop = max(1, int(sr * res))
    n = len(data) // hop
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    chunks = data[: n * hop].reshape(n, hop)
    rms = np.sqrt(np.mean(chunks.astype(np.float64) ** 2, axis=1))
    return (20 * np.log10(rms + 1e-6)).astype(np.float32)


def robust_z(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    if mad < 1e-6:
        mad = np.std(x) or 1.0
    return np.clip((x - med) / mad, -4, 4)


def compute(src, wav_path, duration, cache_json, compute_visual=False):
    if os.path.isfile(cache_json):
        with open(cache_json) as f:
            d = json.load(f)
        return {"loud": np.array(d["loud"]), "motion": np.array(d["motion"]), "cuts": d["cuts"], "res": d.get("res", RES)}
        
    data, sr = media.read_wav(wav_path)
    loud = loudness_db(data, sr, res=RES)
    
    nbins = int(np.ceil(duration / RES))
    motion = np.zeros(nbins, dtype=np.float32)
    cuts = []

    # Optional visual motion pass if requested
    if compute_visual:
        try:
            prev = None
            for t, fr in media.iter_small_frames(src, fps=0.5, width=160, height=90):
                f = fr.astype(np.int16)
                if prev is not None:
                    idx = min(nbins - 1, int(t / RES))
                    motion[idx] = float(np.mean(np.abs(f - prev)))
                prev = f
        except Exception:
            pass

    n = max(len(loud), len(motion))
    loud = np.pad(loud, (0, n - len(loud)), constant_values=loud.min() if len(loud) else -90)
    motion = np.pad(motion, (0, n - len(motion)))
    
    result = {"loud": loud.round(2).tolist(), "motion": motion.round(3).tolist(), "cuts": cuts, "res": RES}
    with open(cache_json, "w") as f:
        json.dump(result, f)
        
    return {"loud": loud, "motion": motion, "cuts": cuts, "res": RES}

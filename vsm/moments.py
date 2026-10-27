"""Find and rank Short-sized moments.

Improvements over the old score_moments.py (keyword windows) + fast_audio_peaks.py (loudness):
 * clips only start/end at clean points (between sentences, never mid-word),
 * many signals: loudness, peaks, motion, cuts, speech, emotional words, early hook, payoff, dead air,
 * non-overlapping selection.
"""
import bisect
import re

import numpy as np

from . import hookscore
from .signals import robust_z

INTENSE = set("""no stop run go now kill killed dead die dying help watch look what why how never wow oh god omg
insane crazy holy damn hell shit fuck wait down contact fire move everyone out boom move get back behind
incoming grenade cover target enemy enemies hostile hostiles shot shoot shots hit fight war yes nice let's
lets come on unbelievable impossible secret truth money win won lose lost""".split())

SENT_END = re.compile(r"[.!?…]$")


def sentences_from_segments(segments, gap=0.7):
    """Split Whisper segments into sentence units using word timings: punctuation or a long pause ends a unit."""
    words = [w for s in segments for w in s.get("words", [])]
    if not words and segments:  # no word timings: fall back to segments as units
        return [{"start": s["start"], "end": s["end"], "text": s["text"], "words": []} for s in segments]
    out, cur = [], []
    for i, w in enumerate(words):
        if cur and w["start"] - cur[-1]["end"] > gap:
            out.append(cur)
            cur = []
        cur.append(w)
        if SENT_END.search(w["w"]):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return [{"start": c[0]["start"], "end": c[-1]["end"], "text": " ".join(x["w"] for x in c), "words": c} for c in out]


def sentence_features(s):
    toks = re.findall(r"[a-z0-9']+", s["text"].lower())
    return {
        "n_words": len(toks),
        "excl": s["text"].count("!"),
        "ques": s["text"].count("?"),
        "intense": sum(1 for t in toks if t in INTENSE),
        "hook": hookscore.score(s["text"])[1] if toks else 0,
    }


def clean_points(sents, duration, cuts=(), grid=1.0, pre=0.2, post=0.35):
    """Times where a clip may start/end without slicing speech."""
    pts = set()
    bounds = [(0.0, 0.0)] + [(s["start"], s["end"]) for s in sents] + [(duration, duration)]
    for (_, prev_end), (nxt_start, _) in zip(bounds[:-1], bounds[1:]):
        lo, hi = prev_end, nxt_start
        if hi < lo:
            continue
        pts.add(round(min(hi, lo + post), 2) if prev_end > 0 else 0.0)
        pts.add(round(max(lo, hi - pre), 2))
        t = lo + grid
        while t < hi - grid * 0.5:
            pts.add(round(t, 2))
            t += grid
        for c in cuts:
            if lo + 0.05 < c < hi - 0.05:
                pts.add(round(c, 2))
    return sorted(p for p in pts if 0 <= p <= duration)


def _mean(arr, a, b):
    seg = arr[a:b]
    return float(seg.mean()) if seg.size else 0.0


def score_window(s, e, ctx):
    res = ctx["res"]
    a, b = int(s / res), max(int(s / res) + 1, int(e / res))
    zl, zm, sl = ctx["zl"], ctx["zm"], ctx["zl_smooth"]
    dur = e - s
    i0 = bisect.bisect_left(ctx["s_starts"], s - 1e-6)
    i1 = bisect.bisect_right(ctx["s_ends"], e + 1e-6)
    inside = [k for k in range(i0, min(i1, len(ctx["sents"]))) if ctx["sents"][k]["start"] >= s - 1e-6]
    feats = [ctx["feats"][k] for k in inside]
    nw = sum(f["n_words"] for f in feats)
    emo = sum(f["excl"] + f["ques"] + f["intense"] for f in feats)
    speech = ctx["speech_mask"][a:b]
    dead = float(np.mean((zl[a:b] < -0.5) & (zm[a:b] < -0.3) & ~speech)) if b > a else 0.0
    peak_seg = sl[a:b]
    peak = float(peak_seg.max()) if peak_seg.size else 0.0
    peak_pos = (int(np.argmax(peak_seg)) / max(1, len(peak_seg) - 1)) if peak_seg.size else 0.0
    early = sl[a:min(b, a + int(3 / res))]
    hook = 0.0
    if early.size:
        hook += max(0.0, float(early.max())) * 0.5
    if inside and ctx["sents"][inside[0]]["start"] - s < 2.5:
        f0 = feats[0]
        hook += f0["hook"] / 100.0 + 0.3 * min(2, f0["excl"] + f0["ques"] + f0["intense"])
    end = 0.0
    if inside:
        last = ctx["sents"][inside[-1]]
        if e - last["end"] < 3.0 and SENT_END.search(last["text"]):
            end += 0.6
    if peak_pos > 0.6:
        end += 0.4
    nc = sum(1 for c in ctx["cuts"] if s < c < e)
    max_speech_hook = max((f["hook"] for f in feats), default=0.0)
    parts = {
        "energy": _mean(zl, a, b),
        "peak": peak,
        "peak_pos": round(float(peak_pos), 2),
        "motion": _mean(zm, a, b),
        "pacing": min(3.0, nc / dur * 10.0),
        "speech": min(3.0, nw / dur),
        "emotion": min(3.0, emo / dur * 10.0),
        "hook": hook,
        "speech_hook": round(float(max_speech_hook), 1),
        "payoff": end,
        "dead_air": dead,
        "length": -abs(dur - ctx["target"]) / ctx["target"],
    }
    total = sum(ctx["weights"].get(k, 0.0) * v for k, v in parts.items())
    return total, parts


WEIGHTS = {"energy": 1.0, "peak": 0.6, "motion": 0.5, "pacing": 0.3, "speech": 0.4, "emotion": 0.6,
           "hook": 1.0, "payoff": 0.5, "dead_air": -2.5, "length": 1.0}


def rank_moments(segments, sig, duration, n="auto", min_len=20.0, max_len=45.0, target=32.0, weights=None,
                 min_gap=2.0):
    """Return non-overlapping clips: [{start, end, score, virality, parts, sentences}] best first.
    
    If n is 'auto', None, or 0, automatically selects all non-overlapping moments meeting
    virality score >= 72 and engagement signals (peak >= 1.0 or motion >= 0.4 or speech hook >= 25)
    without an arbitrary cap.
    """
    if duration < min_len:
        min_len = max(1.0, duration)
        max_len = duration
    res = sig["res"]
    loud = np.asarray(sig["loud"], dtype=np.float64)
    motion = np.asarray(sig["motion"], dtype=np.float64)
    zl, zm = robust_z(loud), robust_z(motion)
    k = max(1, int(2 / res))
    zl_smooth = np.convolve(zl, np.ones(k) / k, mode="same") if zl.size else zl
    sents = sentences_from_segments(segments)
    speech_mask = np.zeros(len(zl), dtype=bool)
    for s in sents:
        speech_mask[int(s["start"] / res): int(s["end"] / res) + 1] = True
    ctx = {"res": res, "zl": zl, "zm": zm, "zl_smooth": zl_smooth, "sents": sents,
           "s_starts": [s["start"] for s in sents], "s_ends": [s["end"] for s in sents],
           "feats": [sentence_features(s) for s in sents], "speech_mask": speech_mask,
           "cuts": sig.get("cuts", []), "target": target, "weights": weights or WEIGHTS}
    # Detect natural audio pauses/valleys to snap clean boundaries during non-dialogue combat
    transients = []
    if len(zl) > 10:
        valleys = np.where((zl[1:-1] < zl[:-2]) & (zl[1:-1] < zl[2:]) & (zl[1:-1] < 0.0))[0] + 1
        transients = [round(float(v * res), 2) for v in valleys]
    pts = clean_points(sents, duration, cuts=list(sig.get("cuts", [])) + transients)
    cands = []
    for i, s in enumerate(pts):
        j = bisect.bisect_left(pts, s + min_len - 1e-6)
        while j < len(pts) and pts[j] - s <= max_len + 1e-6:
            total, parts = score_window(s, pts[j], ctx)
            cands.append((total, s, pts[j], parts))
            j += 1
    if not cands:
        return []
    cands.sort(key=lambda c: -c[0])
    all_scores = np.array([c[0] for c in cands])
    
    is_auto = (n is None) or (isinstance(n, (int, float)) and n <= 0) or (isinstance(n, str) and n.strip().lower() in ("auto", "0", "none", "all"))
    picked = []
    
    if is_auto:
        for total, s, e, parts in cands:
            pct = float((all_scores < total).mean()) if len(all_scores) > 1 else 1.0
            virality = int(round(40 + 59 * pct))
            speech_hook = parts.get("speech_hook", 0.0)
            has_engagement = (parts.get("peak", 0.0) >= 1.0 or parts.get("motion", 0.0) >= 0.4 or speech_hook >= 25.0)
            if virality < 72 or not has_engagement:
                continue
            if any(s < pe + min_gap and e > ps - min_gap for ps, pe, *_ in picked):
                continue
            picked.append((s, e, total, parts, virality))
            
        # Fallback to at least the top candidate if no clip strictly crossed 72
        if not picked and cands:
            total, s, e, parts = cands[0]
            pct = float((all_scores < total).mean()) if len(all_scores) > 1 else 1.0
            virality = int(round(40 + 59 * pct))
            picked.append((s, e, total, parts, virality))
    else:
        target_count = int(n)
        for total, s, e, parts in cands:
            if any(s < pe + min_gap and e > ps - min_gap for ps, pe, *_ in picked):
                continue
            pct = float((all_scores < total).mean()) if len(all_scores) > 1 else 1.0
            virality = int(round(40 + 59 * pct))
            picked.append((s, e, total, parts, virality))
            if len(picked) >= target_count:
                break

    out = []
    for s, e, total, parts, virality in picked:
        out.append({
            "start": round(s, 2),
            "end": round(e, 2),
            "score": round(total, 3),
            "virality": virality,
            "parts": {k: round(v, 3) for k, v in parts.items()},
            "sentences": [x for x in sents if x["start"] >= s - 1e-6 and x["end"] <= e + 1e-6]
        })
    return out


def reasons(parts):
    """Human-readable why-this-clip list for the report."""
    r = []
    if parts["energy"] > 0.5: r.append("louder than most of the video")
    if parts["peak"] > 1.5: r.append("big audio spike (action/reaction)")
    if parts["motion"] > 0.5: r.append("lots of on-screen motion")
    if parts["pacing"] > 1.0: r.append("fast scene cuts")
    if parts["emotion"] > 1.0: r.append("emotional / intense lines")
    if parts["hook"] > 1.0: r.append("strong first 3 seconds")
    if parts["payoff"] >= 0.6: r.append("ends on a complete line or climax")
    if parts["dead_air"] > 0.15: r.append(f"WARNING: {int(parts['dead_air'] * 100)}% quiet/static")
    return r or ["best available window"]

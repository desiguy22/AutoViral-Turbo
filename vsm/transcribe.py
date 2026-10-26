"""Local Whisper transcription with word-level timestamps (needed for word-by-word captions)."""
import json
import os


def transcribe(wav_path, cache_json, model_name="base", language=None, device=None):
    """Return list of segments: {start, end, text, words:[{w, start, end}]}. Cached in cache_json."""
    if os.path.isfile(cache_json):
        with open(cache_json, encoding="utf-8") as f:
            return json.load(f)
    import torch
    import whisper

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = whisper.load_model(model_name, device=device)
    result = model.transcribe(wav_path, language=language, word_timestamps=True, verbose=None,
                              fp16=(device == "cuda"), condition_on_previous_text=False)
    segments = []
    for s in result["segments"]:
        words = [{"w": w["word"].strip(), "start": round(float(w["start"]), 3), "end": round(float(w["end"]), 3)}
                 for w in s.get("words", []) if w["word"].strip()]
        text = s["text"].strip()
        if not text:
            continue
        segments.append({"start": round(float(s["start"]), 3), "end": round(float(s["end"]), 3),
                         "text": text, "no_speech_prob": round(float(s.get("no_speech_prob", 0)), 3),
                         "words": words})
    segments = drop_hallucinations(segments)
    tmp = cache_json + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=1, ensure_ascii=False)
    os.replace(tmp, cache_json)
    return segments


def drop_hallucinations(segments):
    """Whisper invents text over music/gunfire. Drop likely-silent segments and exact runaway repeats."""
    out = []
    for s in segments:
        if s.get("no_speech_prob", 0) > 0.8:
            continue
        if len(out) >= 2 and s["text"] == out[-1]["text"] == out[-2]["text"]:
            continue
        out.append(s)
    return out

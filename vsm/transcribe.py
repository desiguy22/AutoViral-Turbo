"""Local Whisper transcription with word-level timestamps (needed for word-by-word captions).

Language Policy:
- English only by default.
- Hindi ('hi') and Marathi ('mr') are preserved if spoken by the creator.
- All other foreign scripts (Korean Hangul, Cyrillic, Chinese, etc.) and background noise
  are strictly translated or constrained to English.
"""
import json
import os
import re


def detect_allowed_language(wav_path, model):
    """Detect language, strictly enforcing English unless Hindi ('hi') or Marathi ('mr') is spoken."""
    import whisper
    try:
        audio = whisper.load_audio(wav_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        _, probs = model.detect_language(mel)
        top_lang = max(probs, key=probs.get)
        if top_lang in ("hi", "mr") and probs[top_lang] > 0.35:
            return top_lang
    except Exception:
        pass
    return "en"


def transcribe(wav_path, cache_json, model_name="base", language=None, device=None):
    """Return list of segments: {start, end, text, words:[{w, start, end}]}. Cached in cache_json."""
    if os.path.isfile(cache_json):
        with open(cache_json, encoding="utf-8") as f:
            return json.load(f)
    import torch
    import whisper

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = whisper.load_model(model_name, device=device)

    if language is None:
        language = detect_allowed_language(wav_path, model)
    print(f"-> Speech language locked to: {language.upper()} (Allowed: EN, HI, MR)")

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
    """Whisper invents text over music/gunfire. Drop likely-silent segments, single-char artifacts, and exact runaway repeats."""
    out = []
    for s in segments:
        txt = s["text"].strip()
        if s.get("no_speech_prob", 0) > 0.75:
            continue
        # Drop single-char noise or common Whisper silence hallucinations
        if len(txt) <= 2 or txt.lower() in ("0h00", "oh", "um", "uh", "...", "--"):
            continue
        if len(out) >= 1 and txt == out[-1]["text"]:
            continue
        if len(out) >= 2 and txt == out[-2]["text"]:
            continue
        out.append(s)
    return out

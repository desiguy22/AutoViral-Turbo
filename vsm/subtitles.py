"""Word-by-word animated subtitle generator (ASS format) for Instagram Reels and YouTube Shorts.

Creates modern high-retention karaoke/highlight subtitles from Whisper word timestamps:
- Bold uppercase Impact font with heavy dark outline (8px) and drop shadow for 100% legibility
- Highlights active spoken word in Neon Yellow while surrounding words are Crisp White
- Groups words in short 3-4 word phrases to keep viewers reading effortlessly
- Safe-zone positioned in the lower-third (avoiding Instagram/TikTok UI buttons)
"""
import math
import os


def format_ass_time(seconds):
    """Format seconds into ASS timestamp format: H:MM:SS.cs"""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - math.floor(seconds)) * 100))
    if centis >= 100:
        centis = 99
    return f"{hrs}:{mins:02d}:{secs:02d}.{centis:02d}"


# High-dopamine contextual keyword triggers
DANGER_WORDS = {
    "FIRE", "KILL", "KILLED", "DEAD", "DIE", "DYING", "RPG", "BOMB", "BOOM", "ATTACK",
    "GUN", "ENEMY", "HOSTILE", "TARGET", "EXPLOSION", "GRENADE", "DANGER", "DOWN", "AMBUSH",
    "MISSILE", "ROCKET", "BLADE", "KNIFE", "SHOOT", "SHOT", "SHOTS", "SNIPER"
}
VICTORY_WORDS = {
    "NICE", "WIN", "WON", "SAFE", "HIT", "HEADSHOT", "GOOD", "SECURE", "CLUTCH", "PERFECT",
    "YES", "EXTRACT", "CLEAR", "COVERED", "REVIVE", "REVIVED"
}
SHOCK_WORDS = {
    "WHAT", "WHY", "HOW", "WAIT", "OMG", "GOD", "HOLY", "CRAZY", "INSANE", "IMPOSSIBLE",
    "NEVER", "NO", "RUN", "HURRY", "STOP"
}


def get_active_word_markup(word_str):
    """Return ASS markup with dynamic pop micro-animation and contextual keyword coloring."""
    clean = word_str.replace("{", "").replace("}", "")
    raw = clean.strip("!?,.:;\"'").upper()
    
    if raw in DANGER_WORDS:
        color = r"\c&H000055FF&"  # Electric Orange-Red
    elif raw in VICTORY_WORDS:
        color = r"\c&H0033FF33&"  # Vibrant Neon Green
    elif raw in SHOCK_WORDS:
        color = r"\c&H00FFFF00&"  # Electric Cyan
    else:
        color = r"\c&H0000FFFF&"  # Neon Yellow
        
    # Scale bounce: scale to 120% in 70ms, settling to 105%
    anim = r"\t(0,70,\fscx120\fscy120)\t(70,160,\fscx105\fscy105)"
    return f"{{{color}{anim}}}{clean}{{\\rViralMain}}"


def create_ass_subtitles(clip_sentences, clip_start, clip_end, out_ass_path):
    """Generate an .ass subtitle file with word-level karaoke highlighting for the clip."""
    all_words = []
    for s in clip_sentences:
        for w in s.get("words", []):
            w_st = max(0.0, w["start"] - clip_start)
            w_et = max(0.0, w["end"] - clip_start)
            word_str = w["w"].strip().upper()
            if word_str and w_et > w_st:
                all_words.append({
                    "word": word_str,
                    "start": w_st,
                    "end": w_et
                })
                
    # ASS Header with Impact / Arial Black bold styling
    header = """[Script Info]
Title: AutoViral Turbo Animated Subtitles
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: None
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ViralMain,Impact,74,&H00FFFFFF,&H0000FFFF,&H00000000,&H90000000,-1,0,0,0,100,100,2,0,1,8,4,2,60,60,380,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    if not all_words:
        with open(out_ass_path, "w", encoding="utf-8") as f:
            f.write(header)
        return out_ass_path

    # Chunk words into small visual phrases (2 to 4 words each for fast reading)
    chunks = []
    current_chunk = []
    for w in all_words:
        current_chunk.append(w)
        if len(current_chunk) >= 3 or w["word"].endswith((".", "!", "?", "...", ",")):
            chunks.append(current_chunk)
            current_chunk = []
    if current_chunk:
        chunks.append(current_chunk)

    events = []
    for chunk in chunks:
        chunk_start = chunk[0]["start"]
        chunk_end = chunk[-1]["end"]
        
        for active_idx, active_word in enumerate(chunk):
            w_start = active_word["start"]
            if active_idx + 1 < len(chunk):
                w_end = chunk[active_idx + 1]["start"]
            else:
                w_end = active_word["end"] + 0.15
            w_end = min(chunk_end + 0.25, max(w_start + 0.08, w_end))
            
            line_parts = []
            for idx, w in enumerate(chunk):
                if idx == active_idx:
                    line_parts.append(get_active_word_markup(w["word"]))
                else:
                    clean_w = w["word"].replace("{", "").replace("}", "")
                    line_parts.append(clean_w)
                    
            full_line_text = " ".join(line_parts)
            st_str = format_ass_time(w_start)
            et_str = format_ass_time(w_end)
            events.append(f"Dialogue: 0,{st_str},{et_str},ViralMain,,0,0,0,,{full_line_text}")

    with open(out_ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events) + "\n")
        
    return out_ass_path

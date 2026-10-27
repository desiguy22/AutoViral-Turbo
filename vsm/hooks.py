"""Viral Hook, Title, Description, and Hashtag generator for Shorts and Reels.

Generates:
- 2.0s On-screen Hook card (headline + urgency badge)
- YouTube/Reels Title (used as the file name, sanitized for Windows)
- Upload-ready Description & high-engagement Hashtags
- Virality score evaluation via hookscore.py

Operates 100% locally:
- Primary: Local neural LLM (Qwen2.5-0.5B-Instruct running on RTX 4090 GPU)
- Fallback: Calibrated heuristic formulas calibrated with hookscore.py
- Zero external cloud APIs, 100% offline.
"""
import os
import re
import json
import numpy as np

from . import hookscore

# Windows-safe filename sanitizer
ILLEGAL_CHARS = re.compile(r'[\\/:*?"<>|]')

_GLOBAL_LLM = None
_GLOBAL_TOKENIZER = None


def sanitize_title_for_filename(title):
    """Sanitize title into a clean, safe Windows filename."""
    clean = ILLEGAL_CHARS.sub("", title)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean[:80].strip() or "viral_short"


def get_local_llm():
    """Lazy load local LLM on GPU."""
    global _GLOBAL_LLM, _GLOBAL_TOKENIZER
    if _GLOBAL_LLM is None:
        try:
            import torch
            os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
            tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float16,
                device_map='cuda',
                local_files_only=True
            )
            _GLOBAL_TOKENIZER = tok
            _GLOBAL_LLM = model
        except Exception:
            _GLOBAL_LLM = False
            _GLOBAL_TOKENIZER = False
    if _GLOBAL_LLM is False:
        return None, None
    return _GLOBAL_LLM, _GLOBAL_TOKENIZER


def extract_visual_combat_signals(src_video, start_t, end_t, peak_pos=0.5):
    """Extract 3 lightweight keyframes (start, combat peak, end) and analyze visual combat signals.
    
    Detects:
    - Explosions / flash / fireball bursts
    - Thermal optics / Night-vision tactical gear
    - Weapon ADS (Aim Down Sights) reticle / scoped crosshairs
    
    Returns list of descriptive visual signal tokens.
    """
    if not src_video or not os.path.isfile(src_video):
        return []
        
    try:
        import cv2
    except ImportError:
        return []

    signals = []
    dur = max(1.0, end_t - start_t)
    t_start = start_t + min(1.0, dur * 0.1)
    t_peak = start_t + dur * peak_pos
    t_end = max(start_t, end_t - min(1.0, dur * 0.1))
    
    timestamps = [("start", t_start), ("peak", t_peak), ("end", t_end)]
    frames = []
    
    try:
        cap = cv2.VideoCapture(src_video)
        if not cap.isOpened():
            return []
            
        for label, t in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ret, frame = cap.read()
            if ret and frame is not None:
                small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
                frames.append((label, small))
        cap.release()
    except Exception:
        return []
        
    if not frames:
        return []
        
    has_explosion = False
    has_night_vision = False
    has_thermal = False
    has_ads_reticle = False
    
    for label, frame in frames:
        # 1. Explosion / Flash check
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        mean_v = float(np.mean(v))
        if mean_v > 195:
            has_explosion = True
        fire_mask = (h >= 5) & (h <= 25) & (s >= 130) & (v >= 160)
        if float(np.mean(fire_mask)) > 0.04:
            has_explosion = True
            
        # 2. Night-vision / Thermal Optics
        b, g, r = cv2.split(frame)
        mean_b, mean_g, mean_r = float(np.mean(b)), float(np.mean(g)), float(np.mean(r))
        if mean_g > 40 and mean_g > 1.35 * mean_r and mean_g > 1.35 * mean_b:
            has_night_vision = True
            
        ch_diff = np.mean(np.abs(r.astype(float) - g.astype(float))) + np.mean(np.abs(g.astype(float) - b.astype(float)))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if ch_diff < 10.0 and float(np.std(gray)) > 45.0 and mean_v > 30:
            has_thermal = True
            
        # 3. Weapon Reticle / ADS (Aim Down Sights)
        ih, iw = frame.shape[:2]
        center_crop = gray[int(ih*0.35):int(ih*0.65), int(iw*0.35):int(iw*0.65)]
        outer_mask = np.ones_like(gray, dtype=bool)
        outer_mask[int(ih*0.25):int(ih*0.75), int(iw*0.25):int(iw*0.75)] = False
        outer_pixels = gray[outer_mask]
        
        center_edges = cv2.Canny(center_crop, 60, 150)
        center_edge_density = float(np.mean(center_edges > 0))
        outer_mean = float(np.mean(outer_pixels))
        
        if center_edge_density > 0.05 and (outer_mean < 80 or outer_mean < float(np.mean(center_crop)) * 0.7):
            has_ads_reticle = True
            
    if has_explosion:
        signals.append("EXPLOSION_FLASH")
    if has_night_vision:
        signals.append("TACTICAL_NIGHT_VISION")
    if has_thermal:
        signals.append("THERMAL_OPTICS")
    if has_ads_reticle:
        signals.append("WEAPON_RETICLE_ADS")
        
    return signals


def generate_metadata_local_llm(moment, video_title="Gameplay", existing_titles=None, src_video=None):
    """Generate viral metadata using local neural LLM on GPU with visual tokens."""
    model, tok = get_local_llm()
    if model is None or tok is None:
        return None

    sents = moment.get("sentences", [])
    full_text = " ".join(s["text"] for s in sents).strip()
    parts = moment.get("parts", {})

    # Extract lightweight multimodal vision combat signals
    visual_signals = []
    if src_video:
        visual_signals = extract_visual_combat_signals(
            src_video,
            moment.get("start", 0.0),
            moment.get("end", 30.0),
            peak_pos=parts.get("peak_pos", 0.5)
        )

    # Action context when voice lines are sparse or silent
    if not full_text:
        full_text = (
            f"High-intensity gameplay combat sequence (Audio Peak: {parts.get('peak', 1.5):.1f}, "
            f"Motion Energy: {parts.get('motion', 1.0):.1f}, Pacing: {parts.get('pacing', 1.0):.1f}). "
            f"Pure tactical gunplay and explosive action with zero dialogue."
        )

    visual_clause = ""
    if visual_signals:
        v_tokens = " ".join(f"[{s}]" for s in visual_signals)
        visual_clause = f" Visual combat tokens: {v_tokens}."

    avoid_clause = ""
    if existing_titles:
        sample_avoid = [t for t in list(existing_titles)[-8:] if t]
        if sample_avoid:
            avoid_clause = "\nDo NOT use or repeat any of these titles:\n" + "\n".join(f"- {t}" for t in sample_avoid) + "\n"

    prompt = (
        "<|im_start|>system\n"
        "You are an elite viral shorts creator. Given gameplay dialogue or action signals, return JSON only with:\n"
        "- title: High-CTR YouTube Shorts / Reels title (4-8 words, engaging, capitalized)\n"
        "- hook_top: 2-5 words ALL CAPS punchy headline\n"
        "- hook_sub: 2-4 words urgent prompt (e.g. WATCH CLOSELY!! or WAIT FOR IT...)\n"
        "- theme: category (action, betrayal, standoff, clutch, chaos, lore)\n"
        f"{avoid_clause}\n"
        "Example 1:\n"
        "Input: Price: RPG on the roof! Take cover right now!\n"
        "Output: {\"title\": \"Ambushed By RPGs On The Roof\", \"hook_top\": \"ENEMY RPG AMBUSH INCOMING\", \"hook_sub\": \"TAKE COVER RIGHT NOW!!\", \"theme\": \"action\"}\n\n"
        "Example 2:\n"
        "Input: High-intensity gameplay combat sequence (Audio Peak: 2.2, Motion Energy: 1.8). Pure tactical gunplay. Visual combat tokens: [EXPLOSION_FLASH] [WEAPON_RETICLE_ADS].\n"
        "Output: {\"title\": \"Point Blank Crossfire Survival\", \"hook_top\": \"ENEMY FORCES OVERRUN POSITION\", \"hook_sub\": \"WATCH THIS PLAY!!\", \"theme\": \"action\"}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Input: {full_text[:300]}{visual_clause}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    try:
        inputs = tok(prompt, return_tensors='pt').to('cuda')
        outputs = model.generate(**inputs, max_new_tokens=90, do_sample=True, temperature=0.7)
        raw = tok.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Extract JSON block
        json_match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            title = data.get("title", "").strip().strip('"\'')
            hook_top = data.get("hook_top", "").strip().strip('"\'').upper()
            hook_sub = data.get("hook_sub", "").strip().strip('"\'').upper()
            theme = data.get("theme", "action").strip()

            if title and hook_top and hook_sub:
                # Score hook
                _, h_score, h_verdict, h_formula = hookscore.score(f"{hook_top} - {hook_sub}")
                
                desc = (
                    f"{title}\n\n"
                    f"Clip from {video_title} (Timestamp: {moment['start']}s - {moment['end']}s).\n"
                    f"Full high-intensity action captured in 4K 60FPS.\n\n"
                    f"Subscribe for daily gaming shorts & tactical moments!\n\n"
                    f"#Shorts #CallOfDuty #Gaming #ModernWarfare #Reels #TikTok #FPS #Clutch #Gamer"
                )
                
                return {
                    "title": title,
                    "filename": sanitize_title_for_filename(title),
                    "hook_top": hook_top,
                    "hook_sub": hook_sub,
                    "hook_score": h_score,
                    "hook_formula": h_formula,
                    "description": desc,
                    "hashtags": ["#Shorts", "#CallOfDuty", "#Gaming", "#ModernWarfare", "#Reels", "#ViralGaming"],
                    "theme": theme,
                    "engine": "local_llm"
                }
    except Exception:
        pass

    return None


TACTICAL_FALLBACKS = [
    ("WHEN THE SQUAD GETS PINNED DOWN", "DON'T LOOK AWAY!!", "Pinned Down Under Heavy Suppression", "action"),
    ("POINT BLANK GUNFIGHT CLUTCH", "WATCH THIS REACTION!!", "Point Blank Gunfight Reaction Time", "clutch"),
    ("THEY THOUGHT THEY HAD US TRAPPED", "BIGGEST MISTAKE EVER!", "They Thought They Had Us Trapped", "action"),
    ("ZERO MARGIN FOR ERROR", "HOLD YOUR BREATH...", "Zero Margin For Error in Gunfight", "suspense"),
    ("WHEN REINFORCEMENTS DON'T ARRIVE", "FIGHT FOR YOUR LIFE!", "When Reinforcements Never Arrive", "action"),
    ("BREACH AND CLEAR IN SILENCE", "PERFECT EXECUTION!!", "Breach And Clear Tactical Infiltration", "stealth"),
    ("LAST MAG AND ENEMY CLOSING IN", "OUT OF AMMO!!", "Last Magazine Against Entire Squad", "clutch"),
    ("PERFECT FLANK ATTACK", "NEVER SAW IT COMING!", "Sneaking Up Behind Enemy Lines", "tactical"),
    ("SNIPER STANDOFF AT 300 METERS", "ONE SHOT ONE KILL!", "High Stakes Sniper Duel", "sniper"),
    ("WHEN EVERYTHING GOES COMPLETELY WRONG", "CAN WE SURVIVE THIS?", "When The Mission Goes Off The Rails", "chaos"),
    ("THE MOST CHAOTIC FIREFIGHT EVER", "BULLETS FLYING EVERYWHERE!", "Chaotic Firefight Under Heavy Fire", "chaos"),
    ("HEAVY CALIBER CROSSFIRE", "DUCK AND COVER!!", "Caught in Heavy Caliber Crossfire", "action"),
    ("SOLO CLUTCH IN HOSTILE TERRITORY", "NEVER GIVE UP!", "Solo Clutch in Hostile Territory", "clutch"),
    ("SURPRISE AMBUSH OUT OF NOWHERE", "CHECK YOUR CORNERS!!", "Ambushed While Pushing Forward", "action"),
    ("HIGH INTENSITY COMBAT RUN", "SPRINT TO COVER NOW!!", "High Velocity Tactical Push", "action")
]


VISUAL_CANDIDATES = {
    "TACTICAL_NIGHT_VISION": [
        ("NIGHT VISION TACTICAL INFILTRATION", "PERFECT SILENT EXECUTION!!", "Night Vision Tactical Infiltration", "stealth"),
        ("TACTICAL NIGHT VISION AMBUSH", "HUNTING IN THE DARK...", "Stealth Night Vision Ambush", "stealth"),
    ],
    "THERMAL_OPTICS": [
        ("THERMAL SCOPE ENEMY SPOTTED", "TARGET LOCKED IN SMOKE!!", "Thermal Optics Target Acquisition", "stealth"),
        ("SEEING THROUGH SMOKE WITH THERMAL", "CANNOT HIDE FROM THIS!", "Thermal Sight Wallbang Elimination", "clutch"),
    ],
    "EXPLOSION_FLASH": [
        ("EXPLOSIVE ARTILLERY DETONATION", "SURVIVING THE DIRECT BLAST!!", "Surviving Heavy Explosive Artillery Blast", "action"),
        ("CLOSE CALL AIR STRIKE DETONATION", "RUN FOR COVER RIGHT NOW!!", "Surviving Point Blank Rocket Explosion", "action"),
    ],
    "WEAPON_RETICLE_ADS": [
        ("POINT BLANK CROSSHAIR LOCK", "ONE SHOT ONE KILL!!", "Precision Weapon Reticle Elimination", "clutch"),
        ("SNIPER CROSSHAIR HEADSHOT", "FLICK OF THE CENTURY!!", "Instant ADS Crosshair Elimination", "clutch"),
    ],
}


def generate_metadata_heuristic(moment, video_title="Gameplay", existing_titles=None, src_video=None):
    """Heuristic rule-based viral hook and title generation with deduplication and visual awareness."""
    sents = moment.get("sentences", [])
    full_text = " ".join(s["text"] for s in sents).strip()
    parts = moment.get("parts", {})
    txt_lower = full_text.lower()
    used = set(existing_titles or [])
    
    # Check visual combat signals if video provided
    visual_signals = []
    if src_video:
        visual_signals = extract_visual_combat_signals(
            src_video,
            moment.get("start", 0.0),
            moment.get("end", 30.0),
            peak_pos=parts.get("peak_pos", 0.5)
        )
    
    cand = None
    # 1. Match unused visual combat candidate if signals present
    for sig in visual_signals:
        for v_cand in VISUAL_CANDIDATES.get(sig, []):
            if v_cand[2] not in used:
                cand = v_cand
                break
        if cand is not None:
            break

    # 2. Match speech keyword context if no visual candidate selected
    if cand is None:
        if any(k in txt_lower for k in ["price", "ghost", "shepherd", "makarov", "bravo"]):
            cand = ("CAPTAIN PRICE'S COLD-BLOODED RULE", "TASK FORCE 141 DOCTRINE", "Captain Price Reveals His Coldest Rule", "lore")
        elif any(k in txt_lower for k in ["rpg", "missile", "rocket", "artillery", "grenade"]):
            cand = ("WHEN AN ENTIRE ARMY LAUNCHES AN AMBUSH", "GRAB THE RPG RIGHT NOW!!", "The Most Chaotic RPG Ambush Ever", "action")
        elif any(k in txt_lower for k in ["civilian", "hostage", "save", "evacuate", "bridge"]):
            cand = ("CIVILIANS TRAPPED UNDER DIRECT FIRE", "CHECK YOUR FIRE!!", "Protecting Civilians Under Direct Fire", "rescue")
        elif any(k in txt_lower for k in ["knife", "blade", "cut", "stab"]):
            cand = ("HE PULLED OUT A CONCEALED BLADE", "WATCH HIS HANDS!!", "Close Quarters Concealed Knife Struggle", "suspense")
        elif any(k in txt_lower for k in ["lost", "down", "dead", "killed", "die"]):
            cand = ("POV: YOUR SQUAD GETS AMBUSHED", "NOBODY SURVIVES THIS...", "POV When Your Mission Goes Completely Wrong", "disaster")
        elif any(k in txt_lower for k in ["armor", "tank", "convoy", "vehicle"]):
            cand = ("WHEN ENEMY ARMOR ROLLS UP ON YOU", "WHERE IS OUR AIR SUPPORT?!", "Heavy Armored Convoy Rolls In Out Of Nowhere", "action")
        elif any(k in txt_lower for k in ["stand down", "face me", "wrong side", "kill you"]):
            cand = ("FINAL 1V1 STANDOFF", "STAND DOWN BEFORE I PUT YOU DOWN!", "Intense 1v1 Standoff Before You Pull The Trigger", "showdown")

    # If no keyword matched or candidate title was already used, select a fresh unique action title
    if cand is None or cand[2] in used:
        for fallback in TACTICAL_FALLBACKS:
            if fallback[2] not in used:
                cand = fallback
                break
        if cand is None:
            cand = TACTICAL_FALLBACKS[0]

    hook_top, hook_sub, title, theme = cand
    _, h_score, h_verdict, h_formula = hookscore.score(f"{hook_top} - {hook_sub}")
    
    desc = (
        f"{title}\n\n"
        f"Clip from {video_title} (Timestamp: {moment['start']}s - {moment['end']}s).\n"
        f"Full high-intensity action captured in 4K 60FPS.\n\n"
        f"Subscribe for daily gaming shorts & tactical moments!\n\n"
        f"#Shorts #CallOfDuty #Gaming #ModernWarfare #Reels #TikTok #FPS #Clutch #Gamer"
    )
    
    hashtags = ["#Shorts", "#CallOfDuty", "#Gaming", "#ModernWarfare", "#Reels", "#ViralGaming"]
    
    return {
        "title": title,
        "filename": sanitize_title_for_filename(title),
        "hook_top": hook_top,
        "hook_sub": hook_sub,
        "hook_score": h_score,
        "hook_formula": h_formula,
        "description": desc,
        "hashtags": hashtags,
        "theme": theme,
        "engine": "heuristic"
    }


def generate_metadata(moment, video_title="Gameplay", use_local_llm=True, existing_titles=None, src_video=None):
    """Generate viral metadata, trying local LLM first, falling back to heuristics with deduplication and vision signals."""
    if use_local_llm:
        res = generate_metadata_local_llm(moment, video_title, existing_titles=existing_titles, src_video=src_video)
        if res is not None:
            # If the LLM returned a title already seen, don't blindly duplicate
            if existing_titles and res["title"] in existing_titles:
                pass
            else:
                return res

    return generate_metadata_heuristic(moment, video_title, existing_titles=existing_titles, src_video=src_video)

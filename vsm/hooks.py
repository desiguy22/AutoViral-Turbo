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


def generate_metadata_local_llm(moment, video_title="Gameplay", existing_titles=None):
    """Generate viral metadata using local neural LLM on GPU."""
    model, tok = get_local_llm()
    if model is None or tok is None:
        return None

    sents = moment.get("sentences", [])
    full_text = " ".join(s["text"] for s in sents).strip()
    parts = moment.get("parts", {})

    # Action context when voice lines are sparse or silent
    if not full_text:
        full_text = (
            f"High-intensity gameplay combat sequence (Audio Peak: {parts.get('peak', 1.5):.1f}, "
            f"Motion Energy: {parts.get('motion', 1.0):.1f}, Pacing: {parts.get('pacing', 1.0):.1f}). "
            f"Pure tactical gunplay and explosive action with zero dialogue."
        )

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
        "Input: High-intensity gameplay combat sequence (Audio Peak: 2.2, Motion Energy: 1.8). Pure tactical gunplay.\n"
        "Output: {\"title\": \"Point Blank Crossfire Survival\", \"hook_top\": \"ENEMY FORCES OVERRUN POSITION\", \"hook_sub\": \"WATCH THIS PLAY!!\", \"theme\": \"action\"}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Input: {full_text[:300]}\n"
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


def generate_metadata_heuristic(moment, video_title="Gameplay", existing_titles=None):
    """Heuristic rule-based viral hook and title generation with deduplication."""
    sents = moment.get("sentences", [])
    full_text = " ".join(s["text"] for s in sents).strip()
    parts = moment.get("parts", {})
    txt_lower = full_text.lower()
    used = set(existing_titles or [])
    
    cand = None
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


def generate_metadata(moment, video_title="Gameplay", use_local_llm=True, existing_titles=None):
    """Generate viral metadata, trying local LLM first, falling back to heuristics with deduplication."""
    if use_local_llm:
        res = generate_metadata_local_llm(moment, video_title, existing_titles=existing_titles)
        if res is not None:
            # If the LLM returned a title already seen, don't blindly duplicate
            if existing_titles and res["title"] in existing_titles:
                pass
            else:
                return res

    return generate_metadata_heuristic(moment, video_title, existing_titles=existing_titles)

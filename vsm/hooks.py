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


def generate_metadata_local_llm(moment, video_title="Gameplay"):
    """Generate viral metadata using local neural LLM on GPU."""
    model, tok = get_local_llm()
    if model is None or tok is None:
        return None

    sents = moment.get("sentences", [])
    full_text = " ".join(s["text"] for s in sents).strip()
    if not full_text:
        return None

    prompt = (
        "<|im_start|>system\n"
        "You are an elite viral shorts creator. Given gameplay dialogue, return JSON only with:\n"
        "- title: High-CTR YouTube Shorts / Reels title (4-8 words, engaging, capitalized)\n"
        "- hook_top: 2-5 words ALL CAPS punchy headline\n"
        "- hook_sub: 2-4 words urgent prompt (e.g. WATCH CLOSELY!! or WAIT FOR IT...)\n"
        "- theme: category (action, betrayal, standoff, clutch, chaos, lore)\n\n"
        "Example 1:\n"
        "Input: Price: RPG on the roof! Take cover right now!\n"
        "Output: {\"title\": \"Ambushed By RPGs On The Roof\", \"hook_top\": \"ENEMY RPG AMBUSH INCOMING\", \"hook_sub\": \"TAKE COVER RIGHT NOW!!\", \"theme\": \"action\"}\n\n"
        "Example 2:\n"
        "Input: Ghost: He pulled out a concealed blade! Check your six!\n"
        "Output: {\"title\": \"Close Quarters Concealed Knife Struggle\", \"hook_top\": \"HE PULLED A CONCEALED BLADE\", \"hook_sub\": \"WATCH HIS HANDS!!\", \"theme\": \"suspense\"}\n"
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


def generate_metadata_heuristic(moment, video_title="Gameplay"):
    """Heuristic rule-based viral hook and title generation from dialogue and signals."""
    sents = moment.get("sentences", [])
    full_text = " ".join(s["text"] for s in sents).strip()
    parts = moment.get("parts", {})
    txt_lower = full_text.lower()
    
    if any(k in txt_lower for k in ["price", "ghost", "shepherd", "makarov", "bravo"]):
        hook_top = "CAPTAIN PRICE'S COLD-BLOODED RULE"
        hook_sub = "TASK FORCE 141 DOCTRINE"
        title = "Captain Price Reveals His Coldest Rule"
        theme = "lore"
    elif any(k in txt_lower for k in ["rpg", "missile", "rocket", "artillery", "grenade"]):
        hook_top = "WHEN AN ENTIRE ARMY LAUNCHES AN AMBUSH"
        hook_sub = "GRAB THE RPG RIGHT NOW!!"
        title = "The Most Chaotic RPG Ambush Ever"
        theme = "action"
    elif any(k in txt_lower for k in ["civilian", "hostage", "save", "evacuate", "bridge"]):
        hook_top = "CIVILIANS TRAPPED UNDER DIRECT FIRE"
        hook_sub = "CHECK YOUR FIRE!!"
        title = "Protecting Civilians Under Direct Fire"
        theme = "rescue"
    elif any(k in txt_lower for k in ["knife", "blade", "cut", "stab"]):
        hook_top = "HE PULLED OUT A CONCEALED BLADE"
        hook_sub = "WATCH HIS HANDS!!"
        title = "Close Quarters Concealed Knife Struggle"
        theme = "suspense"
    elif any(k in txt_lower for k in ["lost", "down", "dead", "killed", "die"]):
        hook_top = "POV: YOUR SQUAD GETS AMBUSHED"
        hook_sub = "NOBODY SURVIVES THIS..."
        title = "POV When Your Mission Goes Completely Wrong"
        theme = "disaster"
    elif any(k in txt_lower for k in ["armor", "tank", "convoy", "vehicle"]):
        hook_top = "WHEN ENEMY ARMOR ROLLS UP ON YOU"
        hook_sub = "WHERE IS OUR AIR SUPPORT?!"
        title = "Heavy Armored Convoy Rolls In Out Of Nowhere"
        theme = "action"
    elif any(k in txt_lower for k in ["stand down", "face me", "wrong side", "kill you"]):
        hook_top = "FINAL 1V1 STANDOFF"
        hook_sub = "STAND DOWN BEFORE I PUT YOU DOWN!"
        title = "Intense 1v1 Standoff Before You Pull The Trigger"
        theme = "showdown"
    elif parts.get("energy", 0) > 0.8 or parts.get("peak", 0) > 2.0:
        hook_top = "THE MOST INTENSE MOMENT IN CALL OF DUTY"
        hook_sub = "MAXIMUM ADRENALINE!!"
        title = "Absolute Chaos in Call of Duty"
        theme = "action"
    else:
        hook_top = "YOU WON'T BELIEVE WHAT HAPPENED NEXT"
        hook_sub = "WAIT TILL THE END..."
        title = "Unbelievable Tactical Moment"
        theme = "curiosity"

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


def generate_metadata(moment, video_title="Gameplay", use_local_llm=True):
    """Generate viral metadata, trying local LLM first, falling back to heuristics."""
    if use_local_llm:
        res = generate_metadata_local_llm(moment, video_title)
        if res is not None:
            return res

    return generate_metadata_heuristic(moment, video_title)

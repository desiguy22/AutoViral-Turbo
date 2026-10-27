"""Autonomous Publishing & Scheduling Engine for Viral Shorts Maker.

Provides:
- Automated scheduling queue assigning 2-3 shorts/day at optimal viral hours:
  * 12:00 PM (Lunch peak)
  * 5:00 PM (Evening commute surge)
  * 8:00 PM (Prime-time chill/viral window)
- Sorts by highest virality score first to maximize initial algorithmic momentum.
- Multi-platform dispatch packets for YouTube Shorts & TikTok.
- High-engagement pinned discussion comments to boost watch time and comment velocity.
- Export to JSON dispatch packets and CSV scheduling sheets.
"""
import os
import re
import json
import csv
from datetime import datetime, timedelta

DEFAULT_SLOTS = ["12:00", "17:00", "20:00"]
DEFAULT_SLOT_LABELS = {
    "12:00": "12:00 PM (Lunch Peak)",
    "17:00": "5:00 PM (Evening Rush)",
    "20:00": "8:00 PM (Prime-Time Peak)"
}

CADENCE_PRESETS = {
    1: ["17:00"],
    2: ["12:00", "20:00"],
    3: ["12:00", "17:00", "20:00"]
}

PINNED_COMMENT_TEMPLATES = {
    "action": [
        "Did that RPG shot actually connect or was it pure splash damage? What would YOU have done here? Drop your take below! 👇",
        "Absolute chaos in this push! Would you have pushed forward or fallen back? Let's debate below! 👇"
    ],
    "clutch": [
        "Would you have held your ground or retreated in this 1v1? Rate this clutch play 1-10 below! 👇",
        "Out of ammo and surrounded—could you have survived this? Drop your reaction below! 👇"
    ],
    "stealth": [
        "Cleanest tactical flank or completely oblivious enemies? Tell me honestly in the comments! 👇",
        "Silent execution vs loud breach: which playstyle do you run? Drop your loadout below! 👇"
    ],
    "sniper": [
        "Was that hit registration clean or pure luck? Comment your sniper sensitivity below! 👇",
        "One shot, one kill under heavy pressure. Could you make this shot? Rate it 1-10! 👇"
    ],
    "chaos": [
        "On a scale of 1-10, how chaotic was this firefight? Who would've panicked first? 😂👇",
        "When the entire plan goes completely off the rails! What's your worst mission fail? Tell us below! 👇"
    ],
    "lore": [
        "Is Captain Price's doctrine harsh or completely necessary for survival? Let's discuss in the comments! 👇",
        "Task Force 141 coldest moments—where does this rank? Drop your thoughts below! 👇"
    ],
    "default": [
        "What's the best tactical play in this situation? Drop your reaction and rate this clip 1-10! 👇",
        "Watch closely at the final seconds—did you catch what happened? Tell us below! 👇"
    ]
}


def create_pinned_comment(short):
    """Generate high-engagement debate-starting pinned discussion comment."""
    theme = str(short.get("theme", "default") or "default").strip().lower()
    hook_top = str(short.get("hook_top", "") or "")
    title = str(short.get("title", "") or "")
    
    templates = PINNED_COMMENT_TEMPLATES.get(theme, PINNED_COMMENT_TEMPLATES["default"])
    idx = abs(hash(title + hook_top)) % len(templates)
    return templates[idx]


def build_dispatch_packet(short, scheduled_dt, slot_label=""):
    """Construct upload-ready platform dispatch packets for YouTube and TikTok."""
    title = short.get("title", "Viral Short")
    desc = short.get("description", "")
    raw_hashtags = short.get("hashtags", ["#Shorts", "#Gaming", "#Viral"])
    if isinstance(raw_hashtags, str):
        tag_list = [t.strip() for t in re.split(r"[\s,]+", raw_hashtags) if t.strip()]
    elif isinstance(raw_hashtags, (list, tuple)):
        tag_list = [str(t).strip() for t in raw_hashtags if str(t).strip()]
    else:
        tag_list = ["#Shorts", "#Gaming", "#Viral"]
    hash_str = " ".join(t if t.startswith("#") else f"#{t}" for t in tag_list)
    clean_tags = [t.replace("#", "") for t in tag_list]
        
    pinned = create_pinned_comment(short)
    iso_time = scheduled_dt.strftime("%Y-%m-%dT%H:%M:%S")
    
    # YouTube Shorts payload
    yt_title = f"{title} #Shorts"
    if len(yt_title) > 100:
        yt_title = yt_title[:97] + "..."
        
    yt_packet = {
        "title": yt_title,
        "description": f"{desc}\n\n📌 PINNED DISCUSSION:\n{pinned}",
        "tags": clean_tags,
        "category_id": "20",  # Gaming
        "privacy_status": "private",
        "scheduled_publish_time": iso_time,
        "made_for_kids": False,
        "pinned_comment": pinned
    }
    
    # TikTok payload
    h_top = short.get('hook_top') or title
    h_sub = short.get('hook_sub') or ''
    tt_caption = f"{h_top} - {h_sub} {hash_str}".strip()
    if len(tt_caption) > 150:
        tt_caption = tt_caption[:147] + "..."
        
    tt_packet = {
        "caption": tt_caption,
        "scheduled_publish_time": iso_time,
        "allow_comments": True,
        "allow_duet": True,
        "allow_stitch": True,
        "pinned_comment": pinned
    }
    
    return {
        "title": title,
        "filename": short.get("filename", ""),
        "thumbnail": short.get("thumbnail", ""),
        "duration": short.get("duration", 30.0),
        "virality_score": short.get("virality_score", 75),
        "hook_top": short.get("hook_top", ""),
        "hook_sub": short.get("hook_sub", ""),
        "scheduled_date": scheduled_dt.strftime("%Y-%m-%d"),
        "scheduled_time": scheduled_dt.strftime("%H:%M"),
        "slot_label": slot_label,
        "iso_timestamp": iso_time,
        "pinned_comment": pinned,
        "platforms": {
            "youtube": yt_packet,
            "tiktok": tt_packet
        }
    }


def generate_schedule(shorts, start_date=None, slots_per_day=3, times=None):
    """Generate prioritized publishing schedule for a collection of shorts.
    
    Sorts by virality_score descending (best first).
    Assigns slots evenly (e.g. 12:00 PM, 5:00 PM, 8:00 PM).
    """
    if not shorts:
        return {"total_scheduled": 0, "queue": []}
        
    if times is not None:
        times = list(times)
    else:
        times = list(CADENCE_PRESETS.get(slots_per_day, DEFAULT_SLOTS[:max(1, min(len(DEFAULT_SLOTS), slots_per_day))]))
    if not times:
        times = ["17:00"]
        
    # Sort candidates by virality_score descending
    sorted_shorts = sorted(shorts, key=lambda s: s.get("virality_score", 0), reverse=True)
    
    if start_date is None:
        # Default start tomorrow
        cur_date = datetime.now() + timedelta(days=1)
    elif isinstance(start_date, str):
        cur_date = datetime.strptime(start_date, "%Y-%m-%d")
    else:
        cur_date = start_date
        
    cur_date = cur_date.replace(hour=0, minute=0, second=0, microsecond=0)
    
    queue = []
    slot_idx = 0
    day_offset = 0
    
    for rank, s in enumerate(sorted_shorts, 1):
        slot_time_str = times[slot_idx]
        hr, mn = map(int, slot_time_str.split(":"))
        slot_dt = cur_date + timedelta(days=day_offset, hours=hr, minutes=mn)
        slot_label = DEFAULT_SLOT_LABELS.get(slot_time_str, f"{slot_time_str} Peak")
        
        packet = build_dispatch_packet(s, slot_dt, slot_label=slot_label)
        packet["rank"] = rank
        queue.append(packet)
        
        slot_idx += 1
        if slot_idx >= len(times):
            slot_idx = 0
            day_offset += 1
            
    return {
        "generated_at": datetime.now().isoformat(),
        "total_scheduled": len(queue),
        "slots_per_day": len(times),
        "time_slots": times,
        "queue": queue
    }


def save_schedule_files(schedule_data, folder_path):
    """Save schedule_queue.json and schedule_manifest.csv in the specified output folder."""
    os.makedirs(folder_path, exist_ok=True)
    json_path = os.path.join(folder_path, "schedule_queue.json")
    csv_path = os.path.join(folder_path, "schedule_manifest.csv")
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(schedule_data, f, indent=2, ensure_ascii=False)
        
    fieldnames = [
        "rank", "scheduled_date", "scheduled_time", "slot_label",
        "virality_score", "title", "filename", "thumbnail",
        "pinned_comment", "youtube_title", "tiktok_caption"
    ]
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in schedule_data.get("queue", []):
            writer.writerow({
                "rank": item["rank"],
                "scheduled_date": item["scheduled_date"],
                "scheduled_time": item["scheduled_time"],
                "slot_label": item["slot_label"],
                "virality_score": item["virality_score"],
                "title": item["title"],
                "filename": item["filename"],
                "thumbnail": item.get("thumbnail", ""),
                "pinned_comment": item["pinned_comment"],
                "youtube_title": item["platforms"]["youtube"]["title"],
                "tiktok_caption": item["platforms"]["tiktok"]["caption"]
            })
            
    return json_path, csv_path


def load_shorts_from_folder(folder_path):
    """Load metadata of all rendered shorts in a folder."""
    shorts = []
    if not os.path.isdir(folder_path):
        return shorts
        
    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith(".json") and fname not in (
            "signals.json", "transcript.json", "tracker_comparison.json", "schedule_queue.json"
        ):
            meta_path = os.path.join(folder_path, fname)
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "title" in data and "filename" in data:
                        shorts.append(data)
            except Exception:
                pass
    return shorts

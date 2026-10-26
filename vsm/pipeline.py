"""Main Viral Shorts Maker Pipeline.

End-to-end execution:
Input Video -> Signals & Transcript -> Ranked Moments -> Subject Tracking & Framing (OpenCV / YOLOv8 / Benchmark)
-> Word-level Animated Subtitles -> 2s Hook Banner -> 1080x1920 60fps NVENC Render
-> <output_dir>/<video_name>/<title>.mp4 + Metadata JSON + Markdown Summary.
"""
import os
import sys
import json
import time

from . import media
from . import signals
from . import transcribe
from . import moments
from . import tracking
from . import subtitles
from . import hooks
from . import render


def process_video(src_video, output_dir=None, num_shorts=5, min_len=20.0, max_len=45.0,
                  target_len=32.0, whisper_model="base", forced_layout=None,
                  tracker="auto", use_local_llm=True, progress_cb=None):
    """Process a single video into high-engagement viral shorts.
    
    Args:
        src_video: path to input video file (e.g. S:/edited vid/codmw4_3.mp4)
        output_dir: base directory to store results (defaults to parent_dir/viral_shorts)
        num_shorts: number of viral shorts to produce
        min_len: minimum clip duration in seconds
        max_len: maximum clip duration in seconds
        target_len: ideal target clip duration
        whisper_model: Whisper model size ('base', 'small', 'medium')
        forced_layout: 'blur_padding', 'smart_crop', or None (auto)
        tracker: 'auto', 'yolo', 'opencv', or 'compare'
        use_local_llm: bool, whether to use local neural LLM for hooks & titles
        progress_cb: optional callback fn(step_str, pct_float)
    """
    t_start_total = time.time()
    src_video = os.path.abspath(src_video)
    if not os.path.isfile(src_video):
        raise FileNotFoundError(f"Video file not found: {src_video}")

    video_name = os.path.splitext(os.path.basename(src_video))[0]
    
    # 1. Setup output folder structure:
    # "create a seperate folder with video name and dump all shorts in that folder,
    # also the shorts name should be the title of the shorts"
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(src_video), "viral shorts")
        
    video_out_dir = os.path.join(output_dir, video_name)
    cache_dir = os.path.join(video_out_dir, ".cache")
    os.makedirs(video_out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    print(f"\n=================================================================")
    print(f"🎬 VIRAL SHORTS MAKER: {video_name}")
    print(f"   Source:     {src_video}")
    print(f"   Output:     {video_out_dir}")
    print(f"   Target:     {num_shorts} Shorts ({min_len}s - {max_len}s)")
    print(f"   Tracker:    {tracker.upper()} (Benchmark & Framing)")
    print(f"   Local LLM:  {'Enabled (Qwen2.5 on GPU)' if use_local_llm else 'Disabled (Heuristic)'}")
    print(f"=================================================================")

    # 2. Probe source video
    if progress_cb: progress_cb("Probing media stream...", 0.05)
    probe_info = media.probe(src_video)
    duration = probe_info["duration"]
    print(f"-> Probed: {probe_info['width']}x{probe_info['height']} @ {probe_info['fps']:.1f}fps | Duration: {duration/60:.1f} mins ({duration:.1f}s)")

    # 3. Audio extraction
    if progress_cb: progress_cb("Extracting 16kHz audio...", 0.10)
    wav_path = os.path.join(cache_dir, "audio_16k.wav")
    media.extract_audio(src_video, wav_path)

    # 4. Transcription with word-level timestamps
    if progress_cb: progress_cb("Transcribing audio with Whisper...", 0.20)
    transcript_cache = os.path.join(cache_dir, "transcript.json")
    print("-> Transcribing speech on GPU...")
    segments = transcribe.transcribe(wav_path, transcript_cache, model_name=whisper_model)
    print(f"-> Speech transcribed: {len(segments)} segments")

    # 5. Audio & Visual Signals (Loudness, Motion, Scene Cuts)
    if progress_cb: progress_cb("Computing loudness & visual signals...", 0.35)
    signals_cache = os.path.join(cache_dir, "signals.json")
    print("-> Analyzing audio dynamics & motion energy...")
    sig = signals.compute(src_video, wav_path, duration, signals_cache)

    # 6. Rank top viral moments
    if progress_cb: progress_cb("Ranking top viral moments...", 0.45)
    print("-> Scoring moments across 10 viral retention metrics...")
    ranked = moments.rank_moments(segments, sig, duration, n=num_shorts, min_len=min_len, max_len=max_len, target=target_len)
    print(f"-> Selected {len(ranked)} high-potential moments")

    # If tracker is 'compare', run comparative benchmark on first moment and save visual frame
    tracker_reports = {}
    if tracker == "compare":
        print("\n-> Running OpenCV vs YOLOv8 Tracker Benchmark...")
        comp_img_path = os.path.join(video_out_dir, "tracker_comparison.jpg")
        first_m = ranked[0]
        first_report = tracking.compare_trackers(src_video, first_m["start"], first_m["end"], num_samples=16, visual_dump_path=comp_img_path)
        tracker_reports["overall"] = first_report
        with open(os.path.join(video_out_dir, "tracker_comparison.json"), "w", encoding="utf-8") as f:
            json.dump(first_report, f, indent=2)
        print(f"   Benchmark Winner: {first_report['winner'].upper()}")
        print(f"   OpenCV: {first_report['opencv']['avg_latency_ms']:.1f}ms/frame (face: {first_report['opencv']['face_rate']*100:.0f}%, std: {first_report['opencv']['jitter_std']:.3f})")
        print(f"   YOLOv8: {first_report['yolo']['avg_latency_ms']:.1f}ms/frame (person: {first_report['yolo']['detection_rate']*100:.0f}%, std: {first_report['yolo']['jitter_std']:.3f})")
        print(f"   Visual overlay saved: {comp_img_path}")

    # 7. Render each moment into a complete Short
    rendered_shorts = []
    seen_filenames = set()

    for idx, moment in enumerate(ranked, 1):
        pct = 0.50 + 0.45 * (idx / max(1, len(ranked)))
        if progress_cb: progress_cb(f"Processing Short {idx}/{len(ranked)}...", pct)

        m_start = moment["start"]
        m_end = moment["end"]
        m_dur = m_end - m_start

        # Generate metadata (Title, Hook, Description, Hashtags) via Local LLM or Heuristics
        meta = hooks.generate_metadata(moment, video_title=video_name, use_local_llm=use_local_llm)
        
        # Ensure unique Windows-safe filename
        base_filename = meta["filename"]
        final_filename = base_filename
        suffix = 1
        while final_filename in seen_filenames:
            final_filename = f"{base_filename}_{suffix}"
            suffix += 1
        seen_filenames.add(final_filename)

        out_mp4 = os.path.join(video_out_dir, f"{final_filename}.mp4")
        out_meta = os.path.join(video_out_dir, f"{final_filename}.json")
        ass_path = os.path.join(cache_dir, f"short_{idx}.ass")

        if os.path.isfile(out_mp4) and os.path.getsize(out_mp4) > 1024 * 1024 and os.path.isfile(out_meta):
            print(f"\n[{idx}/{len(ranked)}] '{meta['title']}' already rendered ({round(os.path.getsize(out_mp4)/(1024*1024), 1)} MB), skipping.")
            try:
                with open(out_meta, "r", encoding="utf-8") as f:
                    rendered_shorts.append(json.load(f))
                continue
            except Exception:
                pass

        # Determine framing layout
        if forced_layout:
            layout = forced_layout
            crop_x = 0.5
            report = {"forced": True}
        else:
            layout, crop_x, report = tracking.analyze_framing(
                src_video, m_start, m_end,
                tracker=tracker,
                visual_dump_path=os.path.join(cache_dir, f"tracker_{idx}.jpg") if tracker == "compare" else None
            )

        print(f"\n[{idx}/{len(ranked)}] Rendering: '{meta['title']}'")
        print(f"      File:      {final_filename}.mp4")
        print(f"      Time:      {m_start:.1f}s - {m_end:.1f}s ({m_dur:.1f}s)")
        print(f"      Layout:    {layout} (Crop X: {crop_x:.2f}) [Tracker: {tracker}]")
        print(f"      Hook (2s): {meta['hook_top']} | {meta['hook_sub']} (Score: {meta['hook_score']}/100)")
        print(f"      Engine:    {meta.get('engine', 'heuristic')}")

        # Generate word-level animated subtitles
        subtitles.create_ass_subtitles(moment["sentences"], m_start, m_end, ass_path)

        # Render video
        res = render.render_short(
            src_video=src_video,
            start_t=m_start,
            end_t=m_end,
            layout=layout,
            crop_x_pct=crop_x,
            hook_top=meta["hook_top"],
            hook_sub=meta["hook_sub"],
            ass_path=ass_path,
            out_mp4=out_mp4
        )

        # Save metadata sidecar
        full_meta = {
            "title": meta["title"],
            "filename": f"{final_filename}.mp4",
            "start": m_start,
            "end": m_end,
            "duration": m_dur,
            "virality_score": moment["virality"],
            "hook_top": meta["hook_top"],
            "hook_sub": meta["hook_sub"],
            "hook_score": meta["hook_score"],
            "hook_formula": meta["hook_formula"],
            "layout": layout,
            "tracker": tracker,
            "engine": meta.get("engine", "heuristic"),
            "description": meta["description"],
            "hashtags": meta["hashtags"],
            "why_this_clip": moments.reasons(moment["parts"]),
            "size_mb": res["size_mb"]
        }
        with open(out_meta, "w", encoding="utf-8") as f:
            json.dump(full_meta, f, indent=2, ensure_ascii=False)

        rendered_shorts.append(full_meta)
        print(f"      Done in {res['elapsed']:.1f}s ({res['size_mb']:.1f} MB) -> {out_mp4}")

    # 8. Generate Markdown Summary Report
    summary_md = os.path.join(video_out_dir, "SUMMARY.md")
    write_summary(summary_md, video_name, src_video, rendered_shorts, time.time() - t_start_total, tracker_reports.get("overall"))

    if progress_cb: progress_cb("Complete!", 1.0)
    print(f"\n🎉 ALL {len(rendered_shorts)} VIRAL SHORTS GENERATED SUCCESSFULLY IN {time.time()-t_start_total:.1f}s!")
    print(f"📂 Output Folder: {video_out_dir}")
    return rendered_shorts


def write_summary(md_path, video_name, src_video, shorts, total_sec, comp_report=None):
    lines = [
        f"# Viral Shorts Summary: {video_name}",
        "",
        f"- **Source Video**: `{src_video}`",
        f"- **Total Shorts Generated**: {len(shorts)}",
        f"- **Pipeline Time**: {total_sec:.1f} seconds",
        f"- **Resolution**: 1080x1920 (9:16 vertical @ 60fps)",
        ""
    ]

    if comp_report:
        lines.extend([
            "## Subject Tracker Benchmark Analysis (OpenCV vs YOLOv8)",
            "",
            f"- **Winner Tracker**: **{comp_report['winner'].upper()}**",
            f"- **Recommended Framing**: `{comp_report['layout']}`",
            "",
            "| Metric | OpenCV (Haar + Saliency) | YOLOv8 (Neural Person) |",
            "|---|---|---|",
            f"| **Detection Rate** | {comp_report['opencv']['face_rate']*100:.0f}% faces ({comp_report['opencv']['detection_rate']*100:.0f}% total) | {comp_report['yolo']['detection_rate']*100:.0f}% persons |",
            f"| **Spatial Stability (Jitter Std)** | {comp_report['opencv']['jitter_std']:.3f} | {comp_report['yolo']['jitter_std']:.3f} |",
            f"| **Latency** | {comp_report['opencv']['avg_latency_ms']:.1f} ms/frame | {comp_report['yolo']['avg_latency_ms']:.1f} ms/frame |",
            "",
            "**Benchmark Insights**:",
        ])
        for r in comp_report.get("reasons", []):
            lines.append(f"- {r}")
        lines.append("")

    lines.extend([
        "## Generated Shorts",
        "",
        "| # | File / Title | Dur | Virality | Hook Card (0 - 2.5s) | Framing | Engine |",
        "|---|---|---|---|---|---|---|"
    ])
    for idx, s in enumerate(shorts, 1):
        lines.append(
            f"| **{idx}** | **{s['title']}**<br>`{s['filename']}` | {s['duration']:.1f}s | {s['virality_score']}/100 | **{s['hook_top']}**<br>*{s['hook_sub']}* | {s['layout']} | {s.get('engine', 'heuristic')} |"
        )
    lines.extend([
        "",
        "## Upload-Ready Metadata",
        ""
    ])
    for idx, s in enumerate(shorts, 1):
        lines.extend([
            f"### Short {idx}: {s['title']}",
            f"- **File**: `{s['filename']}` ({s['size_mb']:.1f} MB)",
            f"- **Timestamp**: {s['start']:.1f}s - {s['end']:.1f}s",
            f"- **Virality Score**: {s['virality_score']}/100",
            f"- **Hook Headline**: `{s['hook_top']}`",
            f"- **Hook Subtitle**: `{s['hook_sub']}`",
            f"- **Why this clip**: {', '.join(s['why_this_clip'])}",
            "",
            "**Description**:",
            "```text",
            s["description"],
            "```",
            ""
        ])
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

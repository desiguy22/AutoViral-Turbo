import unittest
import os
import sys
import subprocess
from datetime import datetime

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np
from vsm import hookscore
from vsm import moments
from vsm import tracking
from vsm import subtitles
from vsm import hooks
from vsm import render
from vsm import publisher


class TestViralShortsMaker(unittest.TestCase):

    def test_hookscore(self):
        txt = "97% of creators quit before day 30. Here is why."
        parts, verdict, name, hits = hookscore.score(txt)
        self.assertGreater(verdict, 40)
        self.assertEqual(name, "The Statistic")

    def test_sentence_chunking(self):
        mock_segments = [
            {
                "start": 0.0, "end": 4.0, "text": "Hello world! We are moving now.",
                "words": [
                    {"w": "Hello", "start": 0.1, "end": 0.5},
                    {"w": "world!", "start": 0.6, "end": 1.0},
                    {"w": "We", "start": 1.5, "end": 1.8},
                    {"w": "are", "start": 1.9, "end": 2.1},
                    {"w": "moving", "start": 2.2, "end": 2.7},
                    {"w": "now.", "start": 2.8, "end": 3.2}
                ]
            }
        ]
        sents = moments.sentences_from_segments(mock_segments)
        self.assertEqual(len(sents), 2)
        self.assertEqual(sents[0]["text"], "Hello world!")
        self.assertEqual(sents[1]["text"], "We are moving now.")

    def test_title_sanitizer(self):
        title = "POV: Can You Survive 100% of This?! / Wild <Run>"
        clean = hooks.sanitize_title_for_filename(title)
        self.assertNotIn(":", clean)
        self.assertNotIn("/", clean)
        self.assertNotIn("<", clean)
        self.assertNotIn(">", clean)
        self.assertNotIn("?", clean)
        self.assertTrue(len(clean) > 0)

    def test_opencv_tracker(self):
        tracker = tracking.OpenCVTracker()
        img = np.zeros((180, 320, 3), dtype=np.uint8)
        img[60:120, 240:280] = 255  # bright block on right
        res = tracker.detect(img)
        self.assertGreater(res["x_center"], 0.6)
        self.assertEqual(res["type"], "saliency")

    def test_yolo_tracker(self):
        tracker = tracking.YOLOTracker()
        self.assertTrue(tracker.is_available())
        dummy = np.zeros((360, 640, 3), dtype=np.uint8)
        res = tracker.detect(dummy)
        self.assertIsNotNone(res)
        self.assertIn("x_center", res)

    def test_local_llm_hooks(self):
        moment = {
            "start": 10.0,
            "end": 30.0,
            "sentences": [{"text": "Price: RPG on the roof! Take cover immediately!"}],
            "parts": {"energy": 0.9, "peak": 2.5}
        }
        meta = hooks.generate_metadata(moment, video_title="CODMW4", use_local_llm=True)
        self.assertTrue(len(meta["title"]) > 0)
        self.assertTrue(len(meta["hook_top"]) > 0)
        self.assertTrue(len(meta["hook_sub"]) > 0)
        self.assertGreaterEqual(meta["hook_score"], 0)

    def test_subtitles_ass(self):
        mock_sents = [
            {
                "words": [
                    {"w": "FIRE", "start": 10.0, "end": 10.4},
                    {"w": "THE", "start": 10.5, "end": 10.7},
                    {"w": "RPG", "start": 10.8, "end": 11.2}
                ]
            }
        ]
        test_ass = os.path.join(HERE, "test_out.ass")
        try:
            subtitles.create_ass_subtitles(mock_sents, clip_start=10.0, clip_end=15.0, out_ass_path=test_ass)
            self.assertTrue(os.path.isfile(test_ass))
            with open(test_ass, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("Dialogue:", content)
            self.assertIn(r"\c&H", content)  # Karaoke color highlight
            self.assertIn(r"\t(", content)   # Scale pop animation
        finally:
            if os.path.isfile(test_ass):
                os.remove(test_ass)

    def test_upload_manifest(self):
        sample_shorts = [
            {
                "title": "Epic RPG Ambush",
                "filename": "Epic RPG Ambush.mp4",
                "thumbnail": "Epic RPG Ambush_thumb.jpg",
                "duration": 28.5,
                "virality_score": 92,
                "hook_top": "ENEMY RPG INCOMING",
                "hook_sub": "TAKE COVER RIGHT NOW!!",
                "description": "Epic moment in Modern Warfare",
                "hashtags": ["#Shorts", "#Gaming"]
            }
        ]
        test_csv = os.path.join(HERE, "test_manifest.csv")
        try:
            from vsm import pipeline
            pipeline.write_manifest(test_csv, sample_shorts)
            self.assertTrue(os.path.isfile(test_csv))
            with open(test_csv, "r", encoding="utf-8") as f:
                csv_data = f.read()
            self.assertIn("Epic RPG Ambush", csv_data)
            self.assertIn("virality_score", csv_data)
        finally:
            if os.path.isfile(test_csv):
                os.remove(test_csv)

    def test_filter_graph_builder(self):
        probe = {"width": 3840, "height": 2160}
        fg_blur = render.build_filter_graph(probe, "blur_padding", 0.5, "TOP HOOK", "SUB HOOK", "sub.ass")
        self.assertIn("boxblur", fg_blur)
        self.assertIn("TOP HOOK", fg_blur)

        fg_crop = render.build_filter_graph(probe, "smart_crop", 0.6, "TOP HOOK", "SUB HOOK", "sub.ass")
        self.assertIn("crop=", fg_crop)
        self.assertIn("TOP HOOK", fg_crop)

    def test_title_deduplication(self):
        moment = {
            "start": 0.0,
            "end": 30.0,
            "sentences": [],
            "parts": {"energy": 0.9, "peak": 2.5}
        }
        seen = set()
        for _ in range(5):
            meta = hooks.generate_metadata(moment, video_title="CODMW4", use_local_llm=False, existing_titles=seen)
            self.assertNotIn(meta["title"], seen)
            seen.add(meta["title"])
        self.assertEqual(len(seen), 5)

    def test_uncapped_auto_detection(self):
        # Create signals spanning 150s with multiple intense peaks
        res = 0.5
        n_samples = int(150 / res)
        loud = np.sin(np.linspace(0, 10, n_samples)) * 10 - 25
        # Inject prominent action spikes
        loud[40:50] += 15.0
        loud[120:130] += 18.0
        loud[200:210] += 20.0
        motion = np.random.uniform(0.1, 0.3, n_samples)
        motion[40:50] = 0.8
        motion[120:130] = 0.9
        motion[200:210] = 0.95
        sig = {"res": res, "loud": loud.tolist(), "motion": motion.tolist(), "cuts": [25.0, 65.0, 105.0]}
        mock_segments = [
            {"start": 18.0, "end": 26.0, "text": "RPG incoming on roof!", "words": []},
            {"start": 58.0, "end": 66.0, "text": "Hostile gunship overhead!", "words": []},
            {"start": 98.0, "end": 106.0, "text": "Target down, secure the flank!", "words": []}
        ]
        
        # Test uncapped auto detection (n='auto', n=0, n='0', n=None, n='all')
        res_auto = moments.rank_moments(mock_segments, sig, duration=150.0, n="auto", min_len=20.0, max_len=40.0)
        self.assertGreater(len(res_auto), 0)
        for m in res_auto:
            self.assertGreaterEqual(m["virality"], 40)
            self.assertIn("speech_hook", m["parts"])
            self.assertIn("peak_pos", m["parts"])
            
        res_zero = moments.rank_moments(mock_segments, sig, duration=150.0, n=0, min_len=20.0, max_len=40.0)
        self.assertEqual(len(res_auto), len(res_zero))

        res_str_zero = moments.rank_moments(mock_segments, sig, duration=150.0, n="0", min_len=20.0, max_len=40.0)
        self.assertEqual(len(res_auto), len(res_str_zero))

        res_none = moments.rank_moments(mock_segments, sig, duration=150.0, n=None, min_len=20.0, max_len=40.0)
        self.assertEqual(len(res_auto), len(res_none))

    def test_punch_zoom_filter_graph(self):
        probe = {"width": 3840, "height": 2160}
        fg_punched = render.build_filter_graph(probe, "smart_crop", 0.5, "ENEMY RPG", "TAKE COVER", "sub.ass", peak_t=12.0)
        self.assertIn("crop=939:1669:70:125", fg_punched)
        self.assertIn("enable='between(t,11.75,12.40)'", fg_punched)
        self.assertIn("ENEMY RPG", fg_punched)

    def test_audio_enhancement_assets(self):
        bass_path, tension_path = render.ensure_audio_assets()
        self.assertTrue(os.path.isfile(bass_path))
        self.assertTrue(os.path.isfile(tension_path))
        self.assertGreater(os.path.getsize(bass_path), 5000)
        self.assertGreater(os.path.getsize(tension_path), 50000)

    def test_publisher_schedule_and_packets(self):
        mock_shorts = [
            {"title": "Clutch RPG Duel", "filename": "Clutch RPG Duel.mp4", "virality_score": 94, "theme": "clutch", "hook_top": "ENEMY RPG"},
            {"title": "Night Vision Breach", "filename": "Night Vision Breach.mp4", "virality_score": 88, "theme": "stealth", "hook_top": "SILENT INFILTRATION"},
            {"title": "Chaotic Crossfire", "filename": "Chaotic Crossfire.mp4", "virality_score": 79, "theme": "action", "hook_top": "SURROUNDED"},
            {"title": "Sniper Elimination", "filename": "Sniper Elimination.mp4", "virality_score": 75, "theme": "sniper", "hook_top": "ONE SHOT"}
        ]
        
        sched = publisher.generate_schedule(mock_shorts, start_date="2026-11-01", slots_per_day=3)
        self.assertEqual(sched["total_scheduled"], 4)
        queue = sched["queue"]
        
        # Verify sorted by virality score descending
        virality_order = [q["virality_score"] for q in queue]
        self.assertEqual(virality_order, [94, 88, 79, 75])
        
        # Verify slot assignment: first 3 on Day 1, 4th on Day 2
        self.assertEqual(queue[0]["scheduled_date"], "2026-11-01")
        self.assertEqual(queue[0]["scheduled_time"], "12:00")
        self.assertEqual(queue[1]["scheduled_time"], "17:00")
        self.assertEqual(queue[2]["scheduled_time"], "20:00")
        self.assertEqual(queue[3]["scheduled_date"], "2026-11-02")
        self.assertEqual(queue[3]["scheduled_time"], "12:00")
        
        # Verify pinned comments and platform dispatch packets
        for item in queue:
            self.assertTrue(len(item["pinned_comment"]) > 10)
            self.assertIn("👇", item["pinned_comment"])
            self.assertIn("youtube", item["platforms"])
            self.assertIn("tiktok", item["platforms"])
            self.assertTrue(len(item["platforms"]["youtube"]["title"]) > 0)
            self.assertTrue(len(item["platforms"]["tiktok"]["caption"]) > 0)

    def test_publisher_string_hashtags_and_cadence(self):
        # Test string hashtags normalization
        item = {
            "title": "Tactical Flank",
            "filename": "Tactical Flank.mp4",
            "virality_score": 90,
            "theme": "stealth",
            "hook_top": "SILENT FLANK",
            "hashtags": "#Shorts #Gaming #CallOfDuty"
        }
        packet = publisher.build_dispatch_packet(item, datetime(2026, 11, 1, 17, 0))
        yt_tags = packet["platforms"]["youtube"]["tags"]
        self.assertEqual(yt_tags, ["Shorts", "Gaming", "CallOfDuty"])
        self.assertIn("#Shorts", packet["platforms"]["tiktok"]["caption"])
        self.assertIn("#CallOfDuty", packet["platforms"]["tiktok"]["caption"])

        # Test cadence presets
        sched_1 = publisher.generate_schedule([item], slots_per_day=1)
        self.assertEqual(sched_1["queue"][0]["scheduled_time"], "17:00")
        sched_2 = publisher.generate_schedule([item, item], slots_per_day=2)
        self.assertEqual(sched_2["queue"][0]["scheduled_time"], "12:00")
        self.assertEqual(sched_2["queue"][1]["scheduled_time"], "20:00")

    def test_visual_combat_signals_synthetic_and_fallback(self):
        import cv2
        # Non-existent video path must return empty list without crashing
        res = hooks.extract_visual_combat_signals("non_existent_video.mp4", 0.0, 10.0)
        self.assertEqual(res, [])
        res_none = hooks.extract_visual_combat_signals(None, 0.0, 10.0)
        self.assertEqual(res_none, [])

        # Synthetic video with explosion flash
        synth_mp4 = os.path.join(HERE, "test_synth_flash.mp4")
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(synth_mp4, fourcc, 10, (320, 180))
            flash_frame = np.full((180, 320, 3), 245, dtype=np.uint8)
            for _ in range(30):
                out.write(flash_frame)
            out.release()
            
            sig_res = hooks.extract_visual_combat_signals(synth_mp4, 0.0, 3.0)
            self.assertIn("EXPLOSION_FLASH", sig_res)
            
            # Check metadata generation with visual signal awareness
            moment = {
                "start": 0.0,
                "end": 3.0,
                "sentences": [],
                "parts": {"energy": 0.8, "peak": 2.0, "peak_pos": 0.5}
            }
            meta = hooks.generate_metadata(moment, video_title="COD", use_local_llm=False, src_video=synth_mp4)
            self.assertIn("Artillery", meta["title"])
        finally:
            if os.path.isfile(synth_mp4):
                os.remove(synth_mp4)

    def test_render_short_no_audio_video(self):
        # Verify render_short does not crash on videos without audio stream
        no_audio_mp4 = os.path.join(HERE, "test_no_audio_src.mp4")
        test_out = os.path.join(HERE, "test_no_audio_out.mp4")
        test_ass = os.path.join(HERE, "test_no_audio.ass")
        cmd_make = [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "testsrc=duration=2:size=640x360:rate=24",
            "-c:v", "libx264", "-an", no_audio_mp4
        ]
        subprocess.run(cmd_make, check=True, capture_output=True)
        with open(test_ass, "w") as f:
            f.write("[Script Info]\nTitle: Test\nScriptType: v4.00+\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        try:
            res = render.render_short(no_audio_mp4, 0.0, 2.0, "smart_crop", 0.5, "HEADLINE", "URGENT", test_ass, test_out, peak_t=1.0)
            self.assertTrue(res["success"])
            self.assertTrue(os.path.isfile(test_out))
            self.assertGreater(os.path.getsize(test_out), 1000)
        finally:
            for p in (no_audio_mp4, test_out, test_ass):
                if os.path.isfile(p):
                    os.remove(p)

    def test_edge_cases_empty_and_short_video(self):
        # Empty signals and short duration should handle gracefully
        sig = {"res": 1.0, "loud": [0.0] * 5, "motion": [0.0] * 5, "cuts": []}
        res = moments.rank_moments([], sig, duration=5.0, n="auto", min_len=20.0)
        self.assertIsInstance(res, list)

        # Empty schedule
        empty_sched = publisher.generate_schedule([])
        self.assertEqual(empty_sched["total_scheduled"], 0)
        self.assertEqual(empty_sched["queue"], [])

        # Negative or None peak_t in build_filter_graph
        probe = {"width": 1920, "height": 1080}
        fg_none = render.build_filter_graph(probe, "smart_crop", 0.5, "T", "S", "sub.ass", peak_t=None)
        self.assertNotIn("crop=939:1669", fg_none)
        fg_zero = render.build_filter_graph(probe, "smart_crop", 0.5, "T", "S", "sub.ass", peak_t=0.0)
        self.assertNotIn("crop=939:1669", fg_zero)


if __name__ == "__main__":
    unittest.main()

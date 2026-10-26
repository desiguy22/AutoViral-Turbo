import unittest
import os
import sys

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


if __name__ == "__main__":
    unittest.main()

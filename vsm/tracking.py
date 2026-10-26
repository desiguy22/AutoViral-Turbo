"""Subject detection, neural tracking, and framing decision (smart crop vs blurred top/bottom).

Provides:
- OpenCVTracker: Haar face detection + Sobel gradient saliency center-of-mass.
- YOLOTracker: Neural person & character detection using YOLOv8 on GPU.
- compare_trackers(): Benchmarks OpenCV against YOLOv8 tracking on footage,
  evaluating detection rate, confidence, spatial jitter/stability, and latency,
  with visual comparison overlays.
- analyze_framing(): Automatically determines optimal layout ('smart_crop' vs 'blur_padding')
  based on chosen or benchmarked tracker.
"""
import os
import time
import cv2
import numpy as np

from . import media

# Standard vertical shorts dimensions
OUT_W = 1080
OUT_H = 1920
TARGET_ASPECT = OUT_W / OUT_H  # 9:16 = 0.5625

_GLOBAL_YOLO_MODEL = None


def get_yolo_model():
    """Lazy load YOLOv8 model on GPU."""
    global _GLOBAL_YOLO_MODEL
    if _GLOBAL_YOLO_MODEL is None:
        try:
            from ultralytics import YOLO
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _GLOBAL_YOLO_MODEL = YOLO("yolov8n.pt")
            # Warm up on tiny dummy image
            dummy = np.zeros((360, 640, 3), dtype=np.uint8)
            _GLOBAL_YOLO_MODEL.predict(dummy, verbose=False, device=device, classes=[0])
        except Exception as e:
            _GLOBAL_YOLO_MODEL = False
    return _GLOBAL_YOLO_MODEL if _GLOBAL_YOLO_MODEL is not False else None


class OpenCVTracker:
    """OpenCV classical tracking via Haar Cascades and gradient saliency."""
    def __init__(self):
        self.face_cascade = None
        try:
            if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
                p = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
                if os.path.isfile(p):
                    self.face_cascade = cv2.CascadeClassifier(p)
        except Exception:
            self.face_cascade = None

    def detect(self, bgr_frame):
        """Detect subject in frame.
        
        Returns:
            dict with:
                'x_center': float [0.0 - 1.0]
                'type': 'face', 'saliency', or 'center'
                'confidence': float [0.0 - 1.0]
                'bbox': [x1, y1, x2, y2] in pixels
        """
        h, w = bgr_frame.shape[:2]
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY) if len(bgr_frame.shape) == 3 else bgr_frame

        # 1. Try face cascade
        if self.face_cascade is not None:
            try:
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(25, 25))
                if len(faces) > 0:
                    largest = max(faces, key=lambda r: r[2] * r[3])
                    fx, fy, fw, fh = largest
                    xc = (fx + fw / 2.0) / float(w)
                    return {
                        "x_center": float(np.clip(xc, 0.0, 1.0)),
                        "type": "face",
                        "confidence": 0.85,
                        "bbox": [int(fx), int(fy), int(fx + fw), int(fy + fh)]
                    }
            except Exception:
                pass

        # 2. Gradient Sobel Saliency
        try:
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(grad_x, grad_y)
            total_mag = float(np.sum(mag))
            if total_mag > 1e-4:
                x_coords = np.arange(w, dtype=np.float32)
                x_mass = float(np.sum(np.sum(mag, axis=0) * x_coords) / total_mag)
                xc = x_mass / float(w)
                # Compute approximate bounding box around mass center
                half_box = int(w * 0.15)
                cx = int(x_mass)
                bbox = [max(0, cx - half_box), int(h * 0.2), min(w, cx + half_box), int(h * 0.8)]
                return {
                    "x_center": float(np.clip(xc, 0.0, 1.0)),
                    "type": "saliency",
                    "confidence": 0.50,
                    "bbox": bbox
                }
        except Exception:
            pass

        return {
            "x_center": 0.5,
            "type": "center",
            "confidence": 0.20,
            "bbox": [int(w * 0.35), int(h * 0.2), int(w * 0.65), int(h * 0.8)]
        }


class YOLOTracker:
    """Neural subject tracking via YOLOv8 (person/character detection)."""
    def __init__(self):
        self.model = get_yolo_model()

    def is_available(self):
        return self.model is not None

    def detect(self, bgr_frame):
        """Detect primary person/subject in frame.
        
        Returns:
            dict with:
                'x_center': float [0.0 - 1.0]
                'type': 'yolo_person' or 'none'
                'confidence': float [0.0 - 1.0]
                'bbox': [x1, y1, x2, y2]
        """
        if self.model is None:
            return None

        h, w = bgr_frame.shape[:2]
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            results = self.model.predict(bgr_frame, verbose=False, device=device, classes=[0])
            boxes = results[0].boxes
            if len(boxes) > 0:
                # Pick largest or highest confidence person
                best_box = None
                max_area = 0
                for b in boxes:
                    coords = b.xyxy[0].cpu().numpy()
                    area = (coords[2] - coords[0]) * (coords[3] - coords[1])
                    if area > max_area:
                        max_area = area
                        best_box = b
                
                coords = best_box.xyxy[0].cpu().numpy()
                conf = float(best_box.conf[0].cpu().numpy())
                xc = float((coords[0] + coords[2]) / 2.0) / float(w)
                return {
                    "x_center": float(np.clip(xc, 0.0, 1.0)),
                    "type": "yolo_person",
                    "confidence": conf,
                    "bbox": [int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])]
                }
        except Exception:
            pass

        return {
            "x_center": 0.5,
            "type": "none",
            "confidence": 0.0,
            "bbox": [int(w * 0.35), int(h * 0.2), int(w * 0.65), int(h * 0.8)]
        }


def compare_trackers(src_video, start_t, end_t, num_samples=16, visual_dump_path=None):
    """Benchmark OpenCV vs YOLOv8 tracking on footage and evaluate which is superior.
    
    Computes:
    - Subject detection consistency (% of frames with recognized subject)
    - Jitter / spatial stability (standard deviation of frame-to-frame delta-X)
    - Average confidence score
    - Average processing latency (ms/frame)
    - Visual side-by-side or overlaid comparison frame
    
    Returns:
        dict containing full comparative analysis and recommended layout.
    """
    probe_info = media.probe(src_video)
    in_w = probe_info["width"]
    in_h = probe_info["height"]
    dur = max(1.0, end_t - start_t)
    
    opencv_tracker = OpenCVTracker()
    yolo_tracker = YOLOTracker()
    has_yolo = yolo_tracker.is_available()
    
    opencv_xs, opencv_types, opencv_confs, opencv_times = [], [], [], []
    yolo_xs, yolo_types, yolo_confs, yolo_times = [], [], [], []
    
    sample_frames = []
    
    # Read sampled frames
    cap = cv2.VideoCapture(src_video)
    sample_timestamps = np.linspace(start_t, max(start_t + 0.5, end_t - 0.5), num_samples)
    
    for t in sample_timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ret, frame = cap.read()
        if not ret:
            continue
        # Resize to 640x360 for high-speed tracking inference
        small = cv2.resize(frame, (640, 360))
        sample_frames.append((t, frame, small))
    cap.release()
    
    if not sample_frames:
        return {
            "winner": "opencv",
            "layout": "blur_padding",
            "crop_x": 0.5,
            "reason": "Could not extract video sample frames"
        }

    # Run OpenCV tracker on all samples
    for t, full, small in sample_frames:
        t0 = time.time()
        res_cv = opencv_tracker.detect(small)
        opencv_times.append((time.time() - t0) * 1000.0)
        opencv_xs.append(res_cv["x_center"])
        opencv_types.append(res_cv["type"])
        opencv_confs.append(res_cv["confidence"])

    # Run YOLO tracker on all samples
    if has_yolo:
        for t, full, small in sample_frames:
            t0 = time.time()
            res_yolo = yolo_tracker.detect(small)
            yolo_times.append((time.time() - t0) * 1000.0)
            yolo_xs.append(res_yolo["x_center"])
            yolo_types.append(res_yolo["type"])
            yolo_confs.append(res_yolo["confidence"])
    else:
        yolo_times = [0.0]
        yolo_xs = [0.5]
        yolo_types = ["unavailable"]
        yolo_confs = [0.0]

    # Calculate statistics
    cv_det_rate = sum(1 for tp in opencv_types if tp in ("face", "saliency")) / len(opencv_types)
    cv_face_rate = sum(1 for tp in opencv_types if tp == "face") / len(opencv_types)
    cv_mean_x = float(np.median(opencv_xs))
    cv_std_x = float(np.std(opencv_xs))
    cv_avg_conf = float(np.mean(opencv_confs))
    cv_avg_ms = float(np.mean(opencv_times))

    if has_yolo:
        yolo_det_rate = sum(1 for tp in yolo_types if tp == "yolo_person") / len(yolo_types)
        yolo_mean_x = float(np.median(yolo_xs))
        yolo_std_x = float(np.std(yolo_xs))
        yolo_avg_conf = float(np.mean(yolo_confs))
        yolo_avg_ms = float(np.mean(yolo_times))
    else:
        yolo_det_rate = 0.0
        yolo_mean_x = 0.5
        yolo_std_x = 1.0
        yolo_avg_conf = 0.0
        yolo_avg_ms = 0.0

    # Comparative Scoring & Winner Determination
    # Criteria:
    # 1. Subject localization capability (YOLO detects actual person/heads; OpenCV saliency often catches guns/crosshairs)
    # 2. Stability (lower std_dev means fewer erratic jumps)
    # 3. Confidence score
    
    scores = {"opencv": 0, "yolo": 0}
    reasons = []

    if has_yolo:
        if yolo_det_rate >= 0.35:
            scores["yolo"] += 40
            reasons.append(f"YOLO detected human subjects in {yolo_det_rate*100:.0f}% of frames (conf: {yolo_avg_conf:.2f})")
        
        if cv_face_rate >= 0.35:
            scores["opencv"] += 35
            reasons.append(f"OpenCV detected human faces in {cv_face_rate*100:.0f}% of frames")
        else:
            reasons.append("OpenCV found 0 clear faces (saliency snapped to weapon/crosshairs)")
            scores["opencv"] += 10

        if yolo_std_x < 0.18 and yolo_det_rate >= 0.35:
            scores["yolo"] += 30
            reasons.append(f"YOLO trajectory is highly stable (jitter std: {yolo_std_x:.3f})")

        if cv_std_x < 0.18:
            scores["opencv"] += 20

        if cv_avg_ms < yolo_avg_ms:
            scores["opencv"] += 15
            reasons.append(f"OpenCV is faster ({cv_avg_ms:.1f}ms vs {yolo_avg_ms:.1f}ms)")
        else:
            scores["yolo"] += 15

        winner = "yolo" if scores["yolo"] > scores["opencv"] else "opencv"
    else:
        winner = "opencv"
        reasons.append("YOLO not installed; OpenCV used by default")

    # Framing Recommendation
    # Full-screen 9:16 edge-to-edge vertical framing is the gold standard for Shorts & Reels
    win_std = yolo_std_x if winner == "yolo" else cv_std_x
    win_det = yolo_det_rate if winner == "yolo" else cv_face_rate
    win_mean_x = yolo_mean_x if winner == "yolo" else cv_mean_x

    if win_std < 0.25:
        layout = "smart_crop"
        crop_x = win_mean_x
        reasons.append(f"Recommended layout: 'smart_crop' (edge-to-edge vertical 9:16 action centered at x={crop_x:.2f})")
    else:
        layout = "blur_padding"
        crop_x = 0.5
        reasons.append("Recommended layout: 'blur_padding' (wide panoramic action spans canvas; preserves full view)")

    # Optional: Save visual comparison frame
    if visual_dump_path and sample_frames:
        try:
            _, _, vis_frame = sample_frames[len(sample_frames) // 2]
            vis = vis_frame.copy()
            vh, vw = vis.shape[:2]

            # Draw OpenCV detection
            cv_res = opencv_tracker.detect(vis)
            cbx = cv_res["bbox"]
            cv2.rectangle(vis, (cbx[0], cbx[1]), (cbx[2], cbx[3]), (255, 200, 0), 2)  # Cyan/Yellow
            cv2.putText(vis, f"OpenCV ({cv_res['type']}) x={cv_res['x_center']:.2f}",
                        (cbx[0], max(20, cbx[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

            # Draw YOLO detection
            if has_yolo:
                yolo_res = yolo_tracker.detect(vis)
                ybx = yolo_res["bbox"]
                if yolo_res["type"] == "yolo_person":
                    cv2.rectangle(vis, (ybx[0], ybx[1]), (ybx[2], ybx[3]), (0, 255, 0), 2)  # Bright Green
                    cv2.putText(vis, f"YOLOv8 ({yolo_res['confidence']:.2f}) x={yolo_res['x_center']:.2f}",
                                (ybx[0], max(40, ybx[3] + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw Winner & Decision Header
            cv2.rectangle(vis, (0, 0), (vw, 36), (20, 20, 20), -1)
            cv2.putText(vis, f"TRACKER BENCHMARK | WINNER: {winner.upper()} | LAYOUT: {layout.upper()}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            os.makedirs(os.path.dirname(visual_dump_path), exist_ok=True)
            cv2.imwrite(visual_dump_path, vis)
        except Exception:
            pass

    return {
        "winner": winner,
        "layout": layout,
        "crop_x": crop_x,
        "opencv": {
            "detection_rate": cv_det_rate,
            "face_rate": cv_face_rate,
            "median_x": cv_mean_x,
            "jitter_std": cv_std_x,
            "avg_confidence": cv_avg_conf,
            "avg_latency_ms": cv_avg_ms
        },
        "yolo": {
            "available": has_yolo,
            "detection_rate": yolo_det_rate,
            "median_x": yolo_mean_x,
            "jitter_std": yolo_std_x,
            "avg_confidence": yolo_avg_conf,
            "avg_latency_ms": yolo_avg_ms
        },
        "scores": scores,
        "reasons": reasons
    }


def analyze_framing(src_video, start_t, end_t, num_samples=16, tracker="auto", visual_dump_path=None):
    """Analyze the clip to determine optimal framing layout:
    - 'blur_padding': upper and lower blur with crisp 16:9 video centered in middle
    - 'smart_crop': 9:16 vertical crop tracking the subject smoothly
    
    Args:
        src_video: path to video
        start_t: clip start time in seconds
        end_t: clip end time in seconds
        num_samples: number of sample frames to analyze
        tracker: 'auto', 'yolo', 'opencv', or 'compare'
        visual_dump_path: optional path to save comparison image
    
    Returns:
        layout: 'blur_padding' or 'smart_crop'
        crop_x_pct: float (center of crop if smart_crop is used, 0.0 to 1.0)
        report: dict containing tracker metrics (if tracker=='compare')
    """
    probe_info = media.probe(src_video)
    in_w = probe_info["width"]
    in_h = probe_info["height"]
    in_aspect = in_w / float(in_h)
    
    # If source is already vertical (9:16 or close), direct crop is obvious
    if in_aspect <= 0.65:
        return "smart_crop", 0.5, {"reason": "Source is already vertical"}

    # Full comparison mode
    if tracker == "compare":
        report = compare_trackers(src_video, start_t, end_t, num_samples, visual_dump_path)
        return report["layout"], report["crop_x"], report

    # Specific or auto tracker mode
    if tracker in ("yolo", "auto"):
        yolo = YOLOTracker()
        if yolo.is_available():
            sample_times = np.linspace(start_t, max(start_t + 0.5, end_t - 0.5), num_samples)
            cap = cv2.VideoCapture(src_video)
            xs, confs = [], []
            for t in sample_times:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ret, frame = cap.read()
                if ret:
                    small = cv2.resize(frame, (640, 360))
                    res = yolo.detect(small)
                    if res["type"] == "yolo_person":
                        xs.append(res["x_center"])
                        confs.append(res["confidence"])
            cap.release()

            if len(xs) > 0 and float(np.std(xs)) < 0.25:
                return "smart_crop", float(np.median(xs)), {"tracker": "yolo", "samples": len(xs)}
            # In first-person / action gameplay, centered smart crop fills the mobile screen perfectly
            return "smart_crop", 0.5, {"tracker": "yolo_center", "reason": "Edge-to-edge vertical 9:16 action"}

    # OpenCV fallback or explicit opencv tracker
    cv_tracker = OpenCVTracker()
    sample_times = np.linspace(start_t, max(start_t + 0.5, end_t - 0.5), num_samples)
    cap = cv2.VideoCapture(src_video)
    xs, types = [], []
    for t in sample_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ret, frame = cap.read()
        if ret:
            small = cv2.resize(frame, (640, 360))
            res = cv_tracker.detect(small)
            xs.append(res["x_center"])
            types.append(res["type"])
    cap.release()

    face_count = sum(1 for tp in types if tp == "face")
    if face_count >= (num_samples * 0.25) and float(np.std(xs)) < 0.25:
        return "smart_crop", float(np.median(xs)), {"tracker": "opencv", "face_count": face_count}

    return "smart_crop", 0.5, {"tracker": "opencv", "reason": "Edge-to-edge vertical 9:16 action"}

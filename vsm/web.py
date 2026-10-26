"""Web Interface for Viral Shorts Maker.

Simple, zero-dependency browser dashboard:
- Input video path & settings (number of shorts, duration, layout)
- Live processing progress bar & log stream
- Preview gallery for rendered 9:16 Shorts with playable video and copyable titles/hashtags
- Runs locally at http://localhost:8080
"""
import http.server
import socketserver
import json
import os
import sys
import threading
import urllib.parse

from . import pipeline

PORT = 8080
CURRENT_JOB = {
    "status": "idle",
    "step": "",
    "progress": 0.0,
    "results": [],
    "error": None
}

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Viral Shorts / Reel Maker</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0f1117;
    --card: #1a1d27;
    --border: #2a2e3d;
    --accent: #ff0055;
    --accent-glow: rgba(255, 0, 85, 0.4);
    --yellow: #ffd600;
    --text: #ffffff;
    --subtext: #8e94a8;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 30px;
    max-width: 1400px;
    margin: 0 auto;
  }
  header {
    margin-bottom: 30px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 20px;
  }
  h1 { font-size: 2.2rem; font-weight: 800; }
  h1 span { color: var(--accent); }
  p.subtitle { color: var(--subtext); margin-top: 6px; font-size: 0.95rem; }
  .grid { display: grid; grid-template-columns: 380px 1fr; gap: 30px; }
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 24px;
  }
  label {
    display: block;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 8px;
    color: var(--subtext);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  input, select {
    width: 100%;
    padding: 12px 14px;
    background: #0d0f14;
    border: 1px solid var(--border);
    border-radius: 8px;
    color: #fff;
    font-size: 0.95rem;
    margin-bottom: 18px;
  }
  input:focus, select:focus {
    border-color: var(--accent);
    outline: none;
    box-shadow: 0 0 10px var(--accent-glow);
  }
  button.btn {
    width: 100%;
    padding: 14px;
    background: linear-gradient(135deg, #ff0055, #ff4400);
    border: none;
    border-radius: 8px;
    color: white;
    font-weight: 800;
    font-size: 1rem;
    cursor: pointer;
    box-shadow: 0 4px 15px var(--accent-glow);
    transition: transform 0.15s ease, opacity 0.15s ease;
  }
  button.btn:hover { transform: translateY(-1px); opacity: 0.95; }
  button.btn:disabled { background: #333; cursor: not-allowed; box-shadow: none; transform: none; }
  .progress-wrap { margin-top: 20px; display: none; }
  .progress-bar-bg {
    width: 100%;
    height: 10px;
    background: #0d0f14;
    border-radius: 5px;
    overflow: hidden;
    margin-top: 8px;
  }
  .progress-bar-fill {
    height: 100%;
    width: 0%;
    background: linear-gradient(90deg, #ffd600, #ff0055);
    transition: width 0.3s ease;
  }
  .status-text { font-size: 0.88rem; color: var(--yellow); margin-top: 6px; }
  .gallery {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 24px;
  }
  .short-card {
    background: #141721;
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }
  .video-container {
    width: 100%;
    aspect-ratio: 9/16;
    background: #000;
    max-height: 480px;
  }
  video { width: 100%; height: 100%; object-fit: contain; }
  .short-info { padding: 16px; flex-grow: 1; display: flex; flex-direction: column; justify-content: space-between; }
  .short-title { font-size: 1rem; font-weight: 700; margin-bottom: 6px; }
  .hook-badge {
    display: inline-block;
    background: rgba(255, 214, 0, 0.15);
    color: var(--yellow);
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 8px;
  }
  .virality-pill {
    float: right;
    background: rgba(255, 0, 85, 0.2);
    color: #ff3377;
    font-size: 0.75rem;
    font-weight: 800;
    padding: 3px 8px;
    border-radius: 20px;
  }
  .desc-box {
    background: #090b0e;
    border: 1px solid #1e2230;
    border-radius: 6px;
    padding: 8px;
    font-size: 0.78rem;
    color: var(--subtext);
    margin-top: 10px;
    max-height: 80px;
    overflow-y: auto;
    white-space: pre-wrap;
  }
</style>
</head>
<body>
  <header>
    <h1>⚡ Viral Shorts <span>& Reel Maker</span></h1>
    <p class="subtitle">Turn any raw or long gameplay/video into viral, attention-grabbing 9:16 Shorts with smart tracking, hooks & animated captions.</p>
  </header>

  <div class="grid">
    <div class="card">
      <form id="renderForm">
        <label>Input Video File Path</label>
        <input type="text" id="videoPath" placeholder="S:\\edited vid\\codmw4_3.mp4" required value="S:\\edited vid\\codmw4_3.mp4">

        <label>Number of Viral Shorts</label>
        <input type="number" id="numShorts" value="5" min="1" max="25">

        <label>Tracking & Framing Engine</label>
        <select id="tracker">
          <option value="compare">Compare Benchmark (OpenCV vs YOLOv8 side-by-side)</option>
          <option value="auto">Auto (YOLOv8 GPU if available, OpenCV fallback)</option>
          <option value="yolo">YOLOv8 Neural Detection (GPU)</option>
          <option value="opencv">OpenCV Classical (Haar + Saliency)</option>
        </select>

        <label>Framing Layout</label>
        <select id="layout">
          <option value="auto">Auto (Subject Tracked or Dual Blur Fallback)</option>
          <option value="blur_padding">Blurred Upper/Lower (Preserves 16:9 HUD)</option>
          <option value="smart_crop">Smart 9:16 Crop (Subject Centered)</option>
        </select>

        <label style="display:flex; align-items:center; gap:8px; cursor:pointer;">
          <input type="checkbox" id="localLlm" checked style="width:auto; margin:0;">
          Use Local Neural LLM (Qwen2.5 on RTX 4090 GPU)
        </label>

        <label>Target Duration (Seconds)</label>
        <input type="number" id="targetLen" value="32" min="15" max="60">

        <button type="submit" class="btn" id="submitBtn">Generate Viral Shorts</button>
      </form>

      <div class="progress-wrap" id="progressWrap">
        <label>Processing Progress</label>
        <div class="status-text" id="statusText">Starting...</div>
        <div class="progress-bar-bg">
          <div class="progress-bar-fill" id="progressFill"></div>
        </div>
      </div>
    </div>

    <div>
      <div id="galleryPlaceholder" style="text-align: center; color: var(--subtext); padding: 80px 20px; border: 2px dashed var(--border); border-radius: 14px;">
        <h3>No Shorts rendered yet</h3>
        <p style="margin-top: 8px;">Select a video on the left and click "Generate Viral Shorts".</p>
      </div>
      <div class="gallery" id="gallery"></div>
    </div>
  </div>

<script>
  const form = document.getElementById('renderForm');
  const btn = document.getElementById('submitBtn');
  const pWrap = document.getElementById('progressWrap');
  const pFill = document.getElementById('progressFill');
  const sText = document.getElementById('statusText');
  const gallery = document.getElementById('gallery');
  const placeholder = document.getElementById('galleryPlaceholder');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    btn.disabled = true;
    pWrap.style.display = 'block';
    
    const payload = {
      video: document.getElementById('videoPath').value,
      num: parseInt(document.getElementById('numShorts').value),
      tracker: document.getElementById('tracker').value,
      layout: document.getElementById('layout').value,
      local_llm: document.getElementById('localLlm').checked,
      target_len: parseFloat(document.getElementById('targetLen').value)
    };

    try {
      await fetch('/api/start', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
      pollStatus();
    } catch (err) {
      alert('Error starting job: ' + err);
      btn.disabled = false;
    }
  });

  async function pollStatus() {
    const res = await fetch('/api/status');
    const data = await res.json();
    
    sText.innerText = data.step || 'Processing...';
    pFill.style.width = Math.round(data.progress * 100) + '%';

    if (data.status === 'done') {
      btn.disabled = false;
      sText.innerText = 'Completed!';
      renderGallery(data.results);
    } else if (data.status === 'error') {
      btn.disabled = false;
      sText.innerText = 'Error: ' + data.error;
    } else {
      setTimeout(pollStatus, 1500);
    }
  }

  function renderGallery(shorts) {
    if (!shorts || shorts.length === 0) return;
    placeholder.style.display = 'none';
    gallery.innerHTML = '';
    
    shorts.forEach(s => {
      const card = document.createElement('div');
      card.className = 'short-card';
      card.innerHTML = `
        <div class="video-container">
          <video controls preload="metadata">
            <source src="/api/video?path=${encodeURIComponent(s.filename_abs)}" type="video/mp4">
          </video>
        </div>
        <div class="short-info">
          <div>
            <span class="virality-pill">🔥 ${s.virality_score}/100</span>
            <span class="hook-badge">${s.hook_formula || 'VIRAL HOOK'}</span>
            <div class="short-title">${s.title}</div>
            <div style="font-size:0.8rem; color:#aaa;">${s.duration.toFixed(1)}s • ${s.layout}</div>
          </div>
          <div class="desc-box">${s.description}</div>
        </div>
      `;
      gallery.appendChild(card);
    });
  }
</script>
</body>
</html>
"""


class WebHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        if url.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            return

        if url.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(CURRENT_JOB).encode("utf-8"))
            return

        if url.path == "/api/video":
            qs = urllib.parse.parse_qs(url.query)
            vid_path = qs.get("path", [None])[0]
            if vid_path and os.path.isfile(vid_path):
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(os.path.getsize(vid_path)))
                self.end_headers()
                with open(vid_path, "rb") as f:
                    while chunk := f.read(64 * 1024):
                        self.wfile.write(chunk)
                return
            self.send_error(404, "File Not Found")
            return

        self.send_error(404)

    def do_POST(self):
        if self.path == "/api/start":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = json.loads(body.decode("utf-8"))
            
            # Start background pipeline
            threading.Thread(target=run_pipeline_job, args=(data,), daemon=True).start()
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"started"}')
            return

        self.send_error(404)


def run_pipeline_job(data):
    global CURRENT_JOB
    CURRENT_JOB["status"] = "running"
    CURRENT_JOB["error"] = None
    CURRENT_JOB["results"] = []
    
    def cb(step, pct):
        CURRENT_JOB["step"] = step
        CURRENT_JOB["progress"] = pct

    try:
        results = pipeline.process_video(
            src_video=data["video"],
            num_shorts=data.get("num", 5),
            target_len=data.get("target_len", 32.0),
            forced_layout=None if data.get("layout") == "auto" else data.get("layout"),
            tracker=data.get("tracker", "auto"),
            use_local_llm=data.get("local_llm", True),
            progress_cb=cb
        )
        # Attach absolute paths for video streaming
        for r in results:
            video_name = os.path.splitext(os.path.basename(data["video"]))[0]
            base_dir = os.path.join(os.path.dirname(data["video"]), "viral_shorts", video_name)
            r["filename_abs"] = os.path.join(base_dir, r["filename"])

        CURRENT_JOB["results"] = results
        CURRENT_JOB["status"] = "done"
        CURRENT_JOB["progress"] = 1.0
    except Exception as e:
        CURRENT_JOB["status"] = "error"
        CURRENT_JOB["error"] = str(e)


def start_server(port=PORT):
    server = socketserver.TCPServer(("0.0.0.0", port), WebHandler)
    print(f"\n=======================================================")
    print(f"🚀 VIRAL SHORTS MAKER WEB UI RUNNING AT:")
    print(f"   http://localhost:{port}")
    print(f"=======================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    start_server()

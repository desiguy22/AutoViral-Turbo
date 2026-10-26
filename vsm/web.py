"""Web Studio & Review Dashboard for AutoViral Turbo.

Zero-dependency local browser studio:
- Full 9:16 vertical video player with mobile framing mockup
- Browse all rendered shorts across all video collections
- One-click copy metadata (title, description, tags) for YouTube Shorts & TikTok
- Real-time search and folder filtering
- Live GPU pipeline launcher with progress streaming
- Runs locally at http://localhost:8080
"""
import http.server
import socketserver
import json
import os
import sys
import glob
import threading
import urllib.parse

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from . import pipeline

PORT = 8501
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
<title>AutoViral Turbo Studio</title>
<link href="https://fonts.googleapis.com/css2?family=Impact&family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #090a0f;
    --card: #13151f;
    --card-hover: #191c2b;
    --border: #232738;
    --accent: #ff0055;
    --accent-glow: rgba(255, 0, 85, 0.35);
    --neon-yellow: #ffff00;
    --green: #00e676;
    --text: #ffffff;
    --subtext: #8e95ad;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Inter', -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 24px 32px;
    max-width: 1600px;
    margin: 0 auto;
  }
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
    flex-wrap: wrap;
    gap: 16px;
  }
  .brand { display: flex; align-items: center; gap: 12px; }
  .logo-pill {
    background: linear-gradient(135deg, #ff0055, #ff4500);
    color: #fff;
    font-weight: 800;
    padding: 6px 14px;
    border-radius: 8px;
    font-size: 0.85rem;
    letter-spacing: 1px;
    text-transform: uppercase;
  }
  h1 { font-size: 1.8rem; font-weight: 800; letter-spacing: -0.5px; }
  .stats-pill {
    display: flex;
    gap: 16px;
    background: var(--card);
    padding: 8px 16px;
    border-radius: 10px;
    border: 1px solid var(--border);
    font-size: 0.85rem;
  }
  .stats-pill strong { color: var(--green); }
  
  .nav-tabs {
    display: flex;
    gap: 10px;
    margin-bottom: 24px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 12px;
  }
  .tab-btn {
    padding: 10px 20px;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 8px;
    color: var(--subtext);
    font-weight: 600;
    cursor: pointer;
    font-size: 0.95rem;
    transition: all 0.2s;
  }
  .tab-btn.active {
    background: var(--card);
    border-color: var(--border);
    color: #fff;
  }
  .tab-btn:hover:not(.active) { color: #fff; }

  /* Library View */
  .controls-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    margin-bottom: 24px;
    flex-wrap: wrap;
  }
  .folder-pills { display: flex; gap: 8px; flex-wrap: wrap; }
  .folder-pill {
    padding: 6px 14px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    cursor: pointer;
    color: var(--subtext);
    transition: all 0.15s;
  }
  .folder-pill.active {
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
  }
  .search-box {
    background: var(--card);
    border: 1px solid var(--border);
    padding: 8px 16px;
    border-radius: 8px;
    color: #fff;
    width: 260px;
    font-size: 0.9rem;
    outline: none;
  }
  .search-box:focus { border-color: var(--accent); }
  .btn-manifest {
    background: #1c2130;
    border: 1px solid var(--border);
    color: var(--green);
    padding: 8px 16px;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.2s;
  }
  .btn-manifest:hover {
    background: rgba(0, 230, 118, 0.15);
    border-color: var(--green);
  }

  .shorts-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 24px;
  }
  .short-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    transition: transform 0.2s, border-color 0.2s;
  }
  .short-card:hover {
    transform: translateY(-4px);
    border-color: #3b4259;
    background: var(--card-hover);
  }
  .player-wrapper {
    position: relative;
    width: 100%;
    aspect-ratio: 9 / 16;
    background: #000;
  }
  .player-wrapper video {
    width: 100%;
    height: 100%;
    object-fit: cover;
    background: #000;
  }
  .card-body {
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    flex: 1;
  }
  .card-title {
    font-size: 1rem;
    font-weight: 700;
    line-height: 1.35;
    color: #fff;
  }
  .card-meta-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.8rem;
    color: var(--subtext);
  }
  .virality-score {
    background: rgba(0, 230, 118, 0.15);
    color: var(--green);
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 6px;
    border: 1px solid rgba(0, 230, 118, 0.3);
  }
  .btn-copy {
    padding: 8px 14px;
    background: #1c2130;
    border: 1px solid var(--border);
    border-radius: 8px;
    color: #fff;
    font-size: 0.8rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    width: 100%;
    margin-top: 6px;
  }
  .btn-copy:hover {
    background: var(--accent);
    border-color: var(--accent);
  }

  /* Generator View */
  .gen-container {
    max-width: 700px;
    margin: 0 auto;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 28px;
  }
  label {
    display: block;
    font-size: 0.8rem;
    font-weight: 700;
    margin-bottom: 8px;
    color: var(--subtext);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  input, select {
    width: 100%;
    padding: 12px 14px;
    background: #090a0f;
    border: 1px solid var(--border);
    border-radius: 8px;
    color: #fff;
    font-size: 0.95rem;
    margin-bottom: 20px;
    outline: none;
  }
  input:focus, select:focus { border-color: var(--accent); }
  .btn-primary {
    width: 100%;
    padding: 14px;
    background: linear-gradient(135deg, #ff0055, #ff4400);
    border: none;
    border-radius: 8px;
    color: white;
    font-weight: 800;
    font-size: 1rem;
    cursor: pointer;
    transition: opacity 0.2s;
  }
  .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }
  .progress-wrap {
    margin-top: 24px;
    display: none;
  }
  .progress-bar-bg {
    background: #090a0f;
    height: 12px;
    border-radius: 6px;
    overflow: hidden;
    margin-top: 8px;
    border: 1px solid var(--border);
  }
  .progress-bar-fill {
    width: 0%;
    height: 100%;
    background: linear-gradient(90deg, #ff0055, var(--green));
    transition: width 0.3s;
  }

  /* Toast */
  #toast {
    position: fixed;
    bottom: 24px;
    right: 24px;
    background: var(--green);
    color: #000;
    font-weight: 700;
    padding: 12px 20px;
    border-radius: 8px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    opacity: 0;
    transform: translateY(10px);
    transition: all 0.25s;
    pointer-events: none;
    z-index: 100;
  }
  #toast.show { opacity: 1; transform: translateY(0); }
</style>
</head>
<body>

<header>
  <div class="brand">
    <div class="logo-pill">AutoViral Turbo</div>
    <h1>Studio Dashboard</h1>
  </div>
  <div class="stats-pill" id="statsPill">
    <div>Total Shorts: <strong id="statCount">--</strong></div>
    <div>Total Size: <strong id="statSize">--</strong></div>
    <div>Framing: <strong>9:16 Full-Bleed</strong></div>
  </div>
</header>

<div class="nav-tabs">
  <button class="tab-btn active" onclick="switchTab('library')">📁 Shorts Library (<span id="tabCount">0</span>)</button>
  <button class="tab-btn" onclick="switchTab('generate')">⚡ New Generator</button>
</div>

<!-- LIBRARY TAB -->
<div id="libraryTab">
  <div class="controls-bar">
    <div class="folder-pills" id="folderPills">
      <div class="folder-pill active" onclick="filterFolder('ALL')">All Collections</div>
    </div>
    <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
      <input type="text" class="search-box" id="searchBox" placeholder="Search by title or hook..." oninput="filterSearch()">
      <button class="btn-manifest" onclick="downloadManifest()">📥 Export Manifest CSV</button>
    </div>
  </div>

  <div class="shorts-grid" id="shortsGrid">
    <div style="grid-column: 1/-1; text-align: center; color: var(--subtext); padding: 80px 0;">Loading rendered shorts...</div>
  </div>
</div>

<!-- GENERATOR TAB -->
<div id="generateTab" style="display: none;">
  <div class="gen-container">
    <h2 style="margin-bottom: 20px;">Generate Viral Shorts</h2>
    <form id="genForm">
      <label>Input Video Path</label>
      <input type="text" id="videoPath" value="S:/edited vid/codmw4_3.mp4" placeholder="S:/edited vid/video.mp4" required>

      <label>Number of Shorts</label>
      <input type="number" id="numShorts" value="10" min="1" max="25">

      <label>Framing Layout</label>
      <select id="layout">
        <option value="auto">Auto (Smart Crop with Crosshair Bias)</option>
        <option value="smart_crop">Smart 9:16 Crop (Edge-to-Edge Action)</option>
        <option value="blur_padding">Blurred Upper/Lower (Dual Letterbox)</option>
      </select>

      <label>Subject Tracking Engine</label>
      <select id="tracker">
        <option value="auto">Auto (YOLOv8 GPU with Reticle Anchor)</option>
        <option value="yolo">YOLOv8 Neural Detection</option>
        <option value="opencv">OpenCV Classical Saliency</option>
      </select>

      <button type="submit" class="btn-primary" id="genBtn">Launch AutoViral Pipeline</button>
    </form>

    <div class="progress-wrap" id="progressWrap">
      <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: var(--subtext);">
        <span id="progressStep">Processing...</span>
        <span id="progressPct">0%</span>
      </div>
      <div class="progress-bar-bg">
        <div class="progress-bar-fill" id="progressFill"></div>
      </div>
    </div>
  </div>
</div>

<div id="toast">Copied to Clipboard!</div>

<script>
let ALL_SHORTS = [];
let ACTIVE_FOLDER = 'ALL';

async function loadLibrary() {
  try {
    const res = await fetch('/api/library');
    const data = await res.json();
    document.getElementById('statCount').innerText = data.total_shorts;
    document.getElementById('statSize').innerText = data.total_gb + ' GB';
    document.getElementById('tabCount').innerText = data.total_shorts;

    ALL_SHORTS = [];
    const pillsContainer = document.getElementById('folderPills');
    pillsContainer.innerHTML = '<div class="folder-pill active" onclick="filterFolder(\\'ALL\\')">All Collections (' + data.total_shorts + ')</div>';

    for (const [folder, list] of Object.entries(data.folders)) {
      ALL_SHORTS.push(...list);
      const p = document.createElement('div');
      p.className = 'folder-pill';
      p.innerText = folder + ' (' + list.length + ')';
      p.onclick = () => filterFolder(folder);
      pillsContainer.appendChild(p);
    }
    renderCards(ALL_SHORTS);
  } catch (err) {
    document.getElementById('shortsGrid').innerHTML = '<div style="color:red; text-align:center;">Failed to load library: ' + err + '</div>';
  }
}

function filterFolder(folder) {
  ACTIVE_FOLDER = folder;
  document.querySelectorAll('.folder-pill').forEach(el => {
    el.classList.toggle('active', el.innerText.startsWith(folder) || (folder === 'ALL' && el.innerText.startsWith('All')));
  });
  filterSearch();
}

function filterSearch() {
  const query = document.getElementById('searchBox').value.toLowerCase().trim();
  const filtered = ALL_SHORTS.filter(s => {
    const matchFolder = (ACTIVE_FOLDER === 'ALL' || s.folder === ACTIVE_FOLDER);
    const matchQuery = !query || s.title.toLowerCase().includes(query) || (s.hook_top && s.hook_top.toLowerCase().includes(query));
    return matchFolder && matchQuery;
  });
  renderCards(filtered);
}

function renderCards(list) {
  const grid = document.getElementById('shortsGrid');
  if (list.length === 0) {
    grid.innerHTML = '<div style="grid-column: 1/-1; text-align: center; color: var(--subtext); padding: 80px 0;">No matching shorts found.</div>';
    return;
  }
  grid.innerHTML = '';
  list.forEach(s => {
    const card = document.createElement('div');
    card.className = 'short-card';
    const posterAttr = s.thumb_path ? `poster="/api/thumb?path=${encodeURIComponent(s.thumb_path)}"` : '';
    card.innerHTML = `
      <div class="player-wrapper">
        <video controls preload="metadata" playsinline ${posterAttr} src="/api/video?path=${encodeURIComponent(s.path)}"></video>
      </div>
      <div class="card-body">
        <div class="card-title">${escapeHtml(s.title)}</div>
        <div class="card-meta-row">
          <span>📁 ${escapeHtml(s.folder)}</span>
          <span class="virality-score">⚡ ${s.virality_score}/100</span>
        </div>
        <div class="card-meta-row">
          <span>⏱ ${s.duration ? s.duration.toFixed(1) + 's' : ''}</span>
          <span>💾 ${s.size_mb} MB</span>
        </div>
        <button class="btn-copy">📋 Copy Title & Tags</button>
      </div>
    `;
    const copyBtn = card.querySelector('.btn-copy');
    copyBtn.addEventListener('click', () => copyMeta(s));
    grid.appendChild(card);
  });
}

function copyTextToClipboard(text) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    return navigator.clipboard.writeText(text).catch(() => fallbackCopy(text));
  } else {
    return fallbackCopy(text);
  }
}

function fallbackCopy(text) {
  const textArea = document.createElement("textarea");
  textArea.value = text;
  textArea.style.position = "fixed";
  textArea.style.left = "-999999px";
  textArea.style.top = "-999999px";
  document.body.appendChild(textArea);
  textArea.focus();
  textArea.select();
  try {
    document.execCommand('copy');
  } catch (err) {
    console.error('Fallback copy failed', err);
  }
  document.body.removeChild(textArea);
  return Promise.resolve();
}

function copyMeta(s) {
  const tags = (s.hashtags && s.hashtags.length) ? (Array.isArray(s.hashtags) ? s.hashtags.join(' ') : s.hashtags) : '#Shorts #Gaming #Viral';
  const text = `${s.title}\n\n${s.description || ''}\n\n${tags}`;
  copyTextToClipboard(text).then(() => {
    showToast('Copied Title & Tags for "' + s.title + '"!');
  });
}

function downloadManifest() {
  window.open('/api/manifest?folder=' + encodeURIComponent(ACTIVE_FOLDER), '_blank');
}

function showToast(msg) {
  const t = document.getElementById('toast');
  t.innerText = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2500);
}

function switchTab(tab) {
  document.querySelectorAll('.tab-btn').forEach((btn, idx) => {
    btn.classList.toggle('active', (tab === 'library' && idx === 0) || (tab === 'generate' && idx === 1));
  });
  document.getElementById('libraryTab').style.display = (tab === 'library') ? 'block' : 'none';
  document.getElementById('generateTab').style.display = (tab === 'generate') ? 'block' : 'none';
}

function escapeHtml(s) {
  return (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// Generator
document.getElementById('genForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = document.getElementById('genBtn');
  const pWrap = document.getElementById('progressWrap');
  btn.disabled = true;
  pWrap.style.display = 'block';

  const payload = {
    video: document.getElementById('videoPath').value,
    num: parseInt(document.getElementById('numShorts').value),
    tracker: document.getElementById('tracker').value,
    layout: document.getElementById('layout').value
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
  document.getElementById('progressStep').innerText = data.step || 'Processing...';
  const pct = Math.round(data.progress * 100);
  document.getElementById('progressPct').innerText = pct + '%';
  document.getElementById('progressFill').style.width = pct + '%';

  if (data.status === 'done') {
    document.getElementById('genBtn').disabled = false;
    showToast('Shorts successfully generated!');
    loadLibrary();
    switchTab('library');
  } else if (data.status === 'error') {
    document.getElementById('genBtn').disabled = false;
    alert('Pipeline error: ' + data.error);
  } else {
    setTimeout(pollStatus, 1500);
  }
}

loadLibrary();
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
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(CURRENT_JOB).encode("utf-8"))
            return

        if url.path == "/api/library":
            base_dir = "S:/edited vid/viral shorts"
            folders = {}
            total_count = 0
            total_size = 0
            if os.path.isdir(base_dir):
                for d in sorted(os.listdir(base_dir)):
                    dp = os.path.join(base_dir, d)
                    if os.path.isdir(dp) and not d.startswith((".", "__", "frames", "visual", "vsm")):
                        mp4s = sorted(glob.glob(os.path.join(dp, "*.mp4")))
                        shorts_list = []
                        for m in mp4s:
                            base_name = os.path.splitext(os.path.basename(m))[0]
                            meta_file = os.path.join(dp, f"{base_name}.json")
                            meta = {}
                            if os.path.isfile(meta_file):
                                try:
                                    with open(meta_file, "r", encoding="utf-8") as mf:
                                        meta = json.load(mf)
                                except Exception:
                                    pass
                            size_mb = round(os.path.getsize(m) / (1024 * 1024), 1)
                            total_size += os.path.getsize(m)
                            shorts_list.append({
                                "filename": os.path.basename(m),
                                "folder": d,
                                "title": meta.get("title", base_name),
                                "duration": meta.get("duration", 30.0),
                                "virality_score": meta.get("virality_score", 85),
                                "hook_top": meta.get("hook_top", "TACTICAL ACTION"),
                                "hook_sub": meta.get("hook_sub", "WATCH CLOSELY!!"),
                                "hook_score": meta.get("hook_score", 45),
                                "description": meta.get("description", ""),
                                "hashtags": meta.get("hashtags", []),
                                "size_mb": size_mb,
                                "thumb_path": os.path.join(dp, f"{base_name}_thumb.jpg").replace("\\", "/") if os.path.isfile(os.path.join(dp, f"{base_name}_thumb.jpg")) else "",
                                "path": m.replace("\\", "/")
                            })
                        total_count += len(shorts_list)
                        folders[d] = shorts_list

            data = {
                "folders": folders,
                "total_shorts": total_count,
                "total_gb": round(total_size / (1024 * 1024 * 1024), 2)
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))
            return

        if url.path == "/api/video":
            qs = urllib.parse.parse_qs(url.query)
            vid_path = qs.get("path", [None])[0]
            if vid_path and os.path.isfile(vid_path):
                file_size = os.path.getsize(vid_path)
                range_header = self.headers.get("Range")
                if range_header and range_header.startswith("bytes="):
                    try:
                        parts = range_header[6:].split("-")
                        start = int(parts[0]) if parts[0] else 0
                        end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
                        end = min(end, file_size - 1)
                        content_len = end - start + 1
                        
                        self.send_response(206)
                        self.send_header("Content-Type", "video/mp4")
                        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                        self.send_header("Content-Length", str(content_len))
                        self.send_header("Accept-Ranges", "bytes")
                        self.end_headers()
                        
                        with open(vid_path, "rb") as f:
                            f.seek(start)
                            remaining = content_len
                            while remaining > 0:
                                chunk = f.read(min(64 * 1024, remaining))
                                if not chunk:
                                    break
                                self.wfile.write(chunk)
                                remaining -= len(chunk)
                        return
                    except Exception:
                        return
                
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(file_size))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                with open(vid_path, "rb") as f:
                    while chunk := f.read(64 * 1024):
                        self.wfile.write(chunk)
                return
            self.send_error(404, "File Not Found")
            return

        if url.path == "/api/thumb":
            qs = urllib.parse.parse_qs(url.query)
            thumb_path = qs.get("path", [None])[0]
            if thumb_path and os.path.isfile(thumb_path):
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(os.path.getsize(thumb_path)))
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
                with open(thumb_path, "rb") as f:
                    while chunk := f.read(64 * 1024):
                        self.wfile.write(chunk)
                return
            self.send_error(404, "Thumbnail Not Found")
            return

        if url.path == "/api/manifest":
            qs = urllib.parse.parse_qs(url.query)
            folder = qs.get("folder", ["ALL"])[0]
            base_dir = "S:/edited vid/viral shorts"
            manifest_file = os.path.join(base_dir, folder, "upload_manifest.csv")
            if folder != "ALL" and os.path.isfile(manifest_file):
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="{folder}_manifest.csv"')
                self.end_headers()
                with open(manifest_file, "rb") as f:
                    self.wfile.write(f.read())
                return
            # If ALL, combine manifests
            combined_rows = []
            headers = None
            if os.path.isdir(base_dir):
                for d in sorted(os.listdir(base_dir)):
                    mf = os.path.join(base_dir, d, "upload_manifest.csv")
                    if os.path.isfile(mf):
                        with open(mf, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                            if lines:
                                if headers is None:
                                    headers = lines[0]
                                combined_rows.extend(lines[1:])
            if headers:
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="all_shorts_manifest.csv"')
                self.end_headers()
                self.wfile.write(headers.encode("utf-8") + "".join(combined_rows).encode("utf-8"))
                return
            self.send_error(404, "No Manifest Found")
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
            num_shorts=data.get("num", 10),
            target_len=data.get("target_len", 32.0),
            forced_layout=None if data.get("layout") == "auto" else data.get("layout"),
            tracker=data.get("tracker", "auto"),
            use_local_llm=data.get("local_llm", True),
            progress_cb=cb
        )
        CURRENT_JOB["results"] = results
        CURRENT_JOB["status"] = "done"
        CURRENT_JOB["progress"] = 1.0
    except Exception as e:
        CURRENT_JOB["status"] = "error"
        CURRENT_JOB["error"] = str(e)


def start_server(port=PORT):
    server = socketserver.TCPServer(("0.0.0.0", port), WebHandler)
    print(f"\n=======================================================")
    print(f">> AUTOVIRAL TURBO STUDIO RUNNING AT:")
    print(f"   http://localhost:{port}")
    print(f"=======================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    start_server()

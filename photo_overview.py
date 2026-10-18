#!/usr/bin/env python3
"""
photo_overview.py - a single-file overview of AI-generated images.

Run it inside a folder of images and it starts a local webserver that
lists every image in a table: a preview in the first column, followed by
the generation info ComfyUI embedded in the file - model, LoRAs and the
positive / negative prompts (plus sampler settings).

Usage:
    python3 photo_overview.py [--port 8000] [--host 127.0.0.1] [--folder PATH] [--recursive] [--open]

No third-party dependencies - standard library only.
"""

import argparse
import json
import mimetypes
import re
import socket
import struct
import sys
import threading
import webbrowser
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, unquote

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# Input names that hold a model file on ComfyUI loader nodes.
MODEL_INPUTS = ("ckpt_name", "unet_name", "model_name", "base_ckpt_name")

# Input names that may hold prompt text on text-encode / string nodes.
TEXT_INPUTS = ("text", "text_g", "text_l", "t5xxl", "clip_l", "prompt", "string", "value", "positive", "negative")

SAMPLER_SETTINGS = ("seed", "noise_seed", "steps", "cfg", "sampler_name", "scheduler", "denoise")


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def read_png_text(path: Path):
    """Return the tEXt / zTXt / iTXt chunks of a PNG as a {key: text} dict."""
    chunks = {}
    with path.open("rb") as f:
        if f.read(8) != PNG_SIGNATURE:
            return chunks
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            length, ctype = struct.unpack(">I4s", header)
            if ctype == b"IEND":
                break
            if ctype == b"IDAT":
                f.seek(length + 4, 1)  # skip image data + CRC
                continue
            data = f.read(length)
            f.read(4)  # CRC
            try:
                if ctype == b"tEXt":
                    key, _, value = data.partition(b"\0")
                    chunks[key.decode("latin-1")] = value.decode("latin-1")
                elif ctype == b"zTXt":
                    key, _, rest = data.partition(b"\0")
                    chunks[key.decode("latin-1")] = zlib.decompress(rest[1:]).decode("latin-1")
                elif ctype == b"iTXt":
                    key, _, rest = data.partition(b"\0")
                    compressed, rest = rest[0], rest[2:]
                    _lang, _, rest = rest.partition(b"\0")
                    _tkey, _, text = rest.partition(b"\0")
                    if compressed:
                        text = zlib.decompress(text)
                    chunks[key.decode("latin-1")] = text.decode("utf-8")
            except (zlib.error, UnicodeDecodeError, IndexError):
                continue
    return chunks


def read_embedded_json(path: Path):
    """Find ComfyUI's "prompt" / "workflow" JSON in non-PNG files (WebP/JPEG
    savers store them in EXIF as e.g. 'prompt:{...}'). Best effort."""
    try:
        raw = path.read_bytes()
    except OSError:
        return {}
    found = {}
    for key in ("prompt", "workflow"):
        marker = (key + ":{").encode()
        start = raw.find(marker)
        if start < 0:
            continue
        start += len(key) + 1
        end = raw.find(b"\0", start)
        blob = raw[start:end if end >= 0 else len(raw)]
        try:
            found[key] = blob.decode("utf-8", "replace")
        except UnicodeDecodeError:
            continue
    return found


def is_link(value):
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], (str, int)) and isinstance(value[1], int)


def collect_texts(graph, node_id, seen, depth=0):
    """Follow a conditioning (or string) input back through the API graph
    and collect all prompt texts that feed into it."""
    node_id = str(node_id)
    if depth > 30 or node_id in seen or node_id not in graph:
        return []
    seen.add(node_id)
    inputs = graph[node_id].get("inputs", {})
    texts = []
    for key in TEXT_INPUTS:
        value = inputs.get(key)
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())
        elif is_link(value):
            texts.extend(collect_texts(graph, value[0], seen, depth + 1))
    if texts:
        return texts
    # Pass-through nodes (ConditioningCombine, ControlNetApply, FluxGuidance, ...)
    for key, value in inputs.items():
        if is_link(value) and ("conditioning" in key.lower() or key.lower() in ("positive", "negative", "cond")):
            texts.extend(collect_texts(graph, value[0], seen, depth + 1))
    return texts


def parse_api_prompt(graph):
    """Extract info from ComfyUI's API-format graph (the PNG "prompt" chunk)."""
    info = {"models": [], "loras": [], "positive": [], "negative": [], "settings": {}}
    if not isinstance(graph, dict):
        return info

    for node in graph.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {}) or {}
        ctype = node.get("class_type", "")

        for key in MODEL_INPUTS:
            if isinstance(inputs.get(key), str):
                info["models"].append(inputs[key])

        if isinstance(inputs.get("lora_name"), str):
            strength = inputs.get("strength_model", inputs.get("strength"))
            info["loras"].append({"name": inputs["lora_name"], "strength": strength if not is_link(strength) else None})
        # rgthree "Power Lora Loader" and similar: lora_1: {on, lora, strength}
        for value in inputs.values():
            if isinstance(value, dict) and isinstance(value.get("lora"), str) and value.get("lora") != "None":
                if value.get("on", True):
                    info["loras"].append({"name": value["lora"], "strength": value.get("strength")})

        if "positive" in inputs or "negative" in inputs:
            for side in ("positive", "negative"):
                if is_link(inputs.get(side)):
                    for text in collect_texts(graph, inputs[side][0], set()):
                        if text not in info[side]:
                            info[side].append(text)
        elif ctype.endswith("Guider") and is_link(inputs.get("conditioning")):
            # Flux-style BasicGuider: a single, positive conditioning.
            for text in collect_texts(graph, inputs["conditioning"][0], set()):
                if text not in info["positive"]:
                    info["positive"].append(text)

        if any(word in ctype for word in ("Sampler", "Scheduler", "Noise")):
            for key in SAMPLER_SETTINGS:
                value = inputs.get(key)
                if value is not None and not is_link(value) and key not in info["settings"]:
                    info["settings"][key] = value

    # No sampler wired up (or an unknown one): show every encoder's text.
    if not info["positive"] and not info["negative"]:
        for node_id, node in graph.items():
            if isinstance(node, dict) and "TextEncode" in node.get("class_type", ""):
                for text in collect_texts(graph, node_id, set()):
                    if text not in info["positive"]:
                        info["positive"].append(text)
    return info


def parse_ui_workflow(workflow):
    """Fallback for files with only the UI "workflow" chunk."""
    info = {"models": [], "loras": [], "positive": [], "negative": [], "settings": {}}
    for node in (workflow or {}).get("nodes", []):
        ntype = node.get("type", "")
        widgets = node.get("widgets_values") or []
        if not isinstance(widgets, list) or not widgets:
            continue
        if ntype in ("CheckpointLoaderSimple", "CheckpointLoader", "UNETLoader", "ImageOnlyCheckpointLoader"):
            if isinstance(widgets[-1 if ntype == "CheckpointLoader" else 0], str):
                info["models"].append(widgets[-1 if ntype == "CheckpointLoader" else 0])
        elif ntype.startswith("LoraLoader"):
            info["loras"].append({"name": widgets[0], "strength": widgets[1] if len(widgets) > 1 else None})
        elif "TextEncode" in ntype and isinstance(widgets[0], str) and widgets[0].strip():
            info["positive"].append(widgets[0].strip())
    return info


def parse_a1111(params: str):
    """Fallback for Automatic1111 / Forge style "parameters" text."""
    info = {"models": [], "loras": [], "positive": [], "negative": [], "settings": {}}
    prompt, _, rest = params.partition("\nNegative prompt:")
    negative, _, settings = rest.partition("\nSteps:")
    if not rest:
        prompt, _, settings = params.partition("\nSteps:")
    settings = "Steps:" + settings if settings else ""
    info["positive"] = [prompt.strip()] if prompt.strip() else []
    info["negative"] = [negative.strip()] if negative.strip() else []
    for name, weight in re.findall(r"<lora:([^:>]+):?([^>]*)>", prompt):
        info["loras"].append({"name": name, "strength": weight or None})
    for key, value in re.findall(r"([A-Za-z][\w ]*?): ([^,]+)", settings):
        if key == "Model":
            info["models"].append(value.strip())
        elif key in ("Steps", "Sampler", "CFG scale", "Seed", "Schedule type"):
            info["settings"][key.lower()] = value.strip()
    return info


def extract_info(path: Path):
    if path.suffix.lower() == ".png":
        chunks = read_png_text(path)
    else:
        chunks = read_embedded_json(path)

    info = None
    if "prompt" in chunks:
        try:
            info = parse_api_prompt(json.loads(chunks["prompt"]))
        except json.JSONDecodeError:
            info = None
    if (info is None or not any(info[k] for k in ("models", "loras", "positive"))) and "workflow" in chunks:
        try:
            info = parse_ui_workflow(json.loads(chunks["workflow"]))
        except (json.JSONDecodeError, AttributeError):
            pass
    if info is None and "parameters" in chunks:
        info = parse_a1111(chunks["parameters"])

    if info is None:
        return {"has_meta": False, "models": [], "loras": [], "positive": [], "negative": [], "settings": {}}
    # De-duplicate while keeping order.
    info["models"] = list(dict.fromkeys(info["models"]))
    seen, loras = set(), []
    for lora in info["loras"]:
        if lora["name"] not in seen:
            seen.add(lora["name"])
            loras.append(lora)
    info["loras"] = loras
    info["has_meta"] = True
    return info


# ---------------------------------------------------------------------------
# Folder scanning (cached by mtime so reloading the page is cheap)
# ---------------------------------------------------------------------------

_cache = {}
_cache_lock = threading.Lock()


def find_images(folder: Path, recursive: bool):
    pattern = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(
        (p for p in pattern if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: p.relative_to(folder).as_posix().lower(),
    )


def scan_folder(folder: Path, recursive: bool):
    rows = []
    for path in find_images(folder, recursive):
        stat = path.stat()
        key = (str(path), stat.st_mtime, stat.st_size)
        with _cache_lock:
            info = _cache.get(key)
        if info is None:
            try:
                info = extract_info(path)
            except OSError:
                info = {"has_meta": False, "models": [], "loras": [], "positive": [], "negative": [], "settings": {}}
            with _cache_lock:
                _cache[key] = info
        rows.append({
            "src": path.relative_to(folder).as_posix(),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            **info,
        })
    return {"title": folder.name or str(folder), "images": rows}


# ---------------------------------------------------------------------------
# Web page
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Photo overview</title>
<style>
  :root {
    --bg: #f6f6f4; --panel: #ffffff; --text: #1d1d1f; --muted: #6b6b70;
    --line: #e2e2de; --accent: #3a5bd9; --chip: #eef1fb; --chip-text: #2a3f99;
    --neg: #fbeeee; --neg-text: #8a2b2b;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #151517; --panel: #1e1e21; --text: #ececee; --muted: #9a9aa2;
      --line: #2e2e33; --accent: #8aa2ff; --chip: #26304f; --chip-text: #c3cfff;
      --neg: #3a2224; --neg-text: #f0b6b6;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
  header { position: sticky; top: 0; z-index: 5; background: var(--bg);
           border-bottom: 1px solid var(--line); padding: 12px 16px;
           display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: center; }
  h1 { font-size: 18px; margin: 0; font-weight: 600; }
  .count { color: var(--muted); }
  .controls { margin-left: auto; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  input[type=search], select, button {
    font: inherit; color: var(--text); background: var(--panel);
    border: 1px solid var(--line); border-radius: 6px; padding: 5px 9px; }
  input[type=search] { width: 260px; max-width: 100%; }
  label.toggle { color: var(--muted); display: flex; gap: 5px; align-items: center; }
  main { padding: 12px 16px 40px; }
  table { width: 100%; border-collapse: collapse; background: var(--panel);
          border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
  th, td { text-align: left; vertical-align: top; padding: 10px; border-bottom: 1px solid var(--line); }
  th { font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: .04em;
       color: var(--muted); background: var(--panel); }
  th.sortable { cursor: pointer; user-select: none; }
  th.sortable:hover { color: var(--text); }
  tr:last-child td { border-bottom: none; }
  td.preview { width: 1%; }
  td.preview img { display: block; width: var(--thumb, 180px); height: var(--thumb, 180px);
                   object-fit: contain; background: var(--bg); border-radius: 4px; cursor: zoom-in; }
  .fname { font-weight: 600; word-break: break-all; margin-top: 6px; font-size: 12px; max-width: var(--thumb, 180px); }
  .sub { color: var(--muted); font-size: 12px; }
  td.model { width: 16%; word-break: break-word; }
  td.loras { width: 16%; }
  .chip { display: inline-block; background: var(--chip); color: var(--chip-text);
          border-radius: 4px; padding: 1px 6px; margin: 0 4px 4px 0; font-size: 12px; word-break: break-all; }
  .chip b { font-weight: 600; margin-left: 4px; }
  .prompt { white-space: pre-wrap; word-break: break-word; position: relative; }
  .prompt + .prompt { margin-top: 8px; padding-top: 8px; border-top: 1px dashed var(--line); }
  .neg { background: var(--neg); color: var(--neg-text); border-radius: 4px; padding: 6px 8px; margin-top: 8px; }
  .neg::before { content: "Negative: "; font-weight: 600; }
  .settings { margin-top: 8px; color: var(--muted); font-size: 12px; }
  .none { color: var(--muted); font-style: italic; }
  .copy { float: right; margin: 0 0 4px 8px; font-size: 11px; padding: 1px 6px; cursor: pointer; opacity: .6; }
  .copy:hover { opacity: 1; }
  .hide-neg .neg { display: none; }
  #lightbox { position: fixed; inset: 0; background: rgba(0,0,0,.9); display: none;
              align-items: center; justify-content: center; z-index: 10; cursor: zoom-out; }
  #lightbox.open { display: flex; }
  #lightbox img { max-width: 96vw; max-height: 96vh; object-fit: contain; }
  @media (max-width: 760px) {
    table, tbody, tr, td { display: block; width: 100% !important; }
    thead { display: none; }
    tr { border-bottom: 1px solid var(--line); }
    td { border: none; padding: 6px 10px; }
    td.preview img { width: 100%; height: auto; max-height: 70vh; }
    .fname { max-width: none; }
  }
</style>
</head>
<body>
<header>
  <h1 id="title">Photo overview</h1>
  <span class="count" id="count"></span>
  <div class="controls">
    <input type="search" id="filter" placeholder="Filter by name, model, LoRA, prompt…">
    <select id="sort">
      <option value="name">Sort: name</option>
      <option value="new">Sort: newest</option>
      <option value="old">Sort: oldest</option>
      <option value="model">Sort: model</option>
    </select>
    <select id="thumb">
      <option value="120">Small previews</option>
      <option value="180" selected>Medium previews</option>
      <option value="300">Large previews</option>
    </select>
    <label class="toggle"><input type="checkbox" id="showNeg" checked> negative</label>
    <button id="reload" title="Rescan the folder">Reload</button>
  </div>
</header>
<main>
  <table>
    <thead><tr>
      <th>Preview</th>
      <th class="sortable" data-sort="model">Model</th>
      <th>LoRAs</th>
      <th>Prompt</th>
    </tr></thead>
    <tbody id="rows"></tbody>
  </table>
</main>
<div id="lightbox"><img alt=""></div>
<script>
const API_URL = "__API_URL__";
const PHOTO_PREFIX = "__PHOTO_PREFIX__";
let data = { images: [] };

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const base = (p) => p.split("/").pop().replace(/\.[^.]+$/, "");
const photoUrl = (src) => PHOTO_PREFIX + src.split("/").map(encodeURIComponent).join("/");

function fmtDate(t) { return new Date(t * 1000).toLocaleString(); }
function fmtSize(b) { return b > 1e6 ? (b / 1e6).toFixed(1) + " MB" : Math.round(b / 1e3) + " KB"; }

function haystack(img) {
  return [img.src, ...img.models, ...img.loras.map((l) => l.name), ...img.positive, ...img.negative]
    .join("\n").toLowerCase();
}

function render() {
  const terms = $("filter").value.toLowerCase().split(/\s+/).filter(Boolean);
  let list = data.images.filter((img) => { const h = haystack(img); return terms.every((t) => h.includes(t)); });
  const sort = $("sort").value;
  const cmp = {
    name: (a, b) => a.src.localeCompare(b.src, undefined, { numeric: true }),
    new: (a, b) => b.mtime - a.mtime,
    old: (a, b) => a.mtime - b.mtime,
    model: (a, b) => (a.models[0] || "￿").localeCompare(b.models[0] || "￿") || a.src.localeCompare(b.src),
  }[sort];
  list = list.slice().sort(cmp);

  $("count").textContent = list.length === data.images.length
    ? `${data.images.length} images` : `${list.length} of ${data.images.length} images`;

  const none = '<span class="none">—</span>';
  $("rows").innerHTML = list.map((img) => {
    const models = img.models.length ? img.models.map((m) => `<div>${esc(base(m))}</div>`).join("") : none;
    const loras = img.loras.length
      ? img.loras.map((l) => `<span class="chip" title="${esc(l.name)}">${esc(base(l.name))}${
          l.strength != null && l.strength !== "" ? `<b>${esc(l.strength)}</b>` : ""}</span>`).join("")
      : none;
    let prompt = img.positive.map((p) =>
      `<div class="prompt"><button class="copy" data-text="${esc(p)}">copy</button>${esc(p)}</div>`).join("");
    prompt += img.negative.map((p) => `<div class="prompt neg">${esc(p)}</div>`).join("");
    const s = img.settings || {};
    const settings = Object.keys(s).map((k) => `${esc(k)}: ${esc(s[k])}`).join(" · ");
    if (settings) prompt += `<div class="settings">${settings}</div>`;
    if (!img.has_meta) prompt = '<span class="none">No generation info found</span>';
    return `<tr>
      <td class="preview"><img loading="lazy" src="${photoUrl(img.src)}" alt="" data-src="${esc(img.src)}">
        <div class="fname">${esc(img.src)}</div>
        <div class="sub">${fmtDate(img.mtime)} · ${fmtSize(img.size)}</div></td>
      <td class="model">${models}</td>
      <td class="loras">${loras}</td>
      <td>${prompt}</td>
    </tr>`;
  }).join("");
}

async function load() {
  const res = await fetch(API_URL, { cache: "no-store" });
  data = await res.json();
  $("title").textContent = data.title;
  document.title = data.title + " · Photo overview";
  render();
}

function applyThumb() { document.documentElement.style.setProperty("--thumb", $("thumb").value + "px"); }

$("filter").addEventListener("input", render);
$("sort").addEventListener("change", render);
$("thumb").addEventListener("change", applyThumb);
$("showNeg").addEventListener("change", (e) => document.body.classList.toggle("hide-neg", !e.target.checked));
$("reload").addEventListener("click", load);
document.querySelector('th[data-sort="model"]').addEventListener("click", () => { $("sort").value = "model"; render(); });

$("rows").addEventListener("click", (e) => {
  const btn = e.target.closest(".copy");
  if (btn) {
    navigator.clipboard.writeText(btn.dataset.text).then(() => {
      btn.textContent = "copied"; setTimeout(() => (btn.textContent = "copy"), 1200);
    });
    return;
  }
  const img = e.target.closest("td.preview img");
  if (img) {
    $("lightbox").querySelector("img").src = img.src;
    $("lightbox").classList.add("open");
  }
});
$("lightbox").addEventListener("click", () => $("lightbox").classList.remove("open"));
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("lightbox").classList.remove("open"); });

applyThumb();
load();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class OverviewHandler(BaseHTTPRequestHandler):
    folder = None
    recursive = False

    def log_message(self, fmt, *args):
        pass  # keep the console quiet

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/":
            html = HTML_PAGE.replace("__API_URL__", "/api/data").replace("__PHOTO_PREFIX__", "/photo/")
            self._send_bytes(html.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/data":
            self._send_json(scan_folder(self.folder, self.recursive))
        elif path.startswith("/photo/"):
            self._serve_photo(path[len("/photo/"):])
        else:
            self.send_error(404, "Not found")

    def _serve_photo(self, name):
        full_path = (self.folder / name).resolve()
        # Only serve image files inside the served folder.
        if (self.folder not in full_path.parents
                or full_path.suffix.lower() not in IMAGE_EXTENSIONS
                or not full_path.is_file()):
            self.send_error(404, "Photo not found")
            return
        content_type = mimetypes.guess_type(full_path.name)[0] or "application/octet-stream"
        self._send_bytes(full_path.read_bytes(), content_type, cache=True)

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, data, content_type, cache=False):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if cache:
            self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(data)


def pick_port(host, start_port):
    port = start_port
    for _ in range(20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return port
            except OSError:
                port += 1
    return start_port


def main():
    parser = argparse.ArgumentParser(description="Serve an overview table of a folder's AI-generated images.")
    parser.add_argument("--port", type=int, default=8000, help="port to listen on (default 8000)")
    parser.add_argument("--host", default="0.0.0.0", help="host to bind (default 0.0.0.0)")
    parser.add_argument("--open", action="store_true", help="open the overview in your default browser")
    parser.add_argument("--folder", default=".", help="folder of images to serve (default: current folder)")
    parser.add_argument("--recursive", "-r", action="store_true", help="include images in subfolders")
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        print(f"Not a folder: {folder}", file=sys.stderr)
        sys.exit(1)

    OverviewHandler.folder = folder
    OverviewHandler.recursive = args.recursive

    overview = scan_folder(folder, args.recursive)
    with_meta = sum(1 for img in overview["images"] if img["has_meta"])

    port = pick_port(args.host, args.port)
    server = ThreadingHTTPServer((args.host, port), OverviewHandler)
    display_host = "localhost" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{display_host}:{port}/"

    print(f"Photo overview for '{folder.name}' ({len(overview['images'])} images, {with_meta} with generation info)")
    print(f"Serving at {url}  (Ctrl+C to stop)")

    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
        server.shutdown()


if __name__ == "__main__":
    main()

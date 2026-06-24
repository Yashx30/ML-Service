import argparse
import csv
import glob
import json
import os
import sys
import time
from urllib.parse import quote

import requests
from tqdm import tqdm

#lightweight local file server for testing local images(optional)
try:
    from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
    import threading
except Exception:
    ThreadingHTTPServer = None
    SimpleHTTPRequestHandler = None
    threading = None


def start_local_server(root_dir: str, port: int = 8001):
    """Start a background HTTP server to serve files from root_dir."""
    if ThreadingHTTPServer is None:
        raise RuntimeError("http.server not available in this environment.")
    os.chdir(root_dir)

    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, fmt, *args):
            # Keeps the console clean
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", port), QuietHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread, port


def collect_inputs(input_spec: str):
    """
    input_spec can be:
      - a directory (all jpg/png/jpeg under it recursively)
      - a glob pattern (e.g. C:\\images\\*.jpg)
      - a text file with one URL or path per line
    Returns list[str] of paths or URLs.
    """
    if os.path.isdir(input_spec):
        patterns = [os.path.join(input_spec, "**", ext) for ext in ("*.jpg", "*.jpeg", "*.png")]
        files = []
        for p in patterns:
            files.extend(glob.glob(p, recursive=True))
        return sorted(files)

    if os.path.isfile(input_spec) and input_spec.lower().endswith(".txt"):
        with open(input_spec, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    hits = glob.glob(input_spec, recursive=True)
    if hits:
        return sorted(hits)

    # if nothing matched, maybe it's a single URL
    if input_spec.startswith("http://") or input_spec.startswith("https://"):
        return [input_spec]

    raise ValueError(f"No inputs found for: {input_spec}")


def to_media_url(item: str, serve_local: bool, root: str, port: int):
    """
    If item is already an http(s) URL, return as-is.
    If it's a local path and serve_local=True, map it to http://127.0.0.1:<port>/<relpath>
    Else, return None (unsupported).
    """
    if item.startswith("http://") or item.startswith("https://"):
        return item

    if serve_local:
        # Compute relative path from root and URL-encode it
        rel = os.path.relpath(os.path.abspath(item), os.path.abspath(root))
        rel_url = "/".join(quote(part) for part in rel.split(os.sep))
        return f"http://127.0.0.1:{port}/{rel_url}"

    return None


def main():
    ap = argparse.ArgumentParser(description="Batch demo: call FastAPI /v1/analyze over many images and save CSV.")
    ap.add_argument("--input", required=True,
                    help="Directory, glob pattern, or a .txt file with one URL/path per line.")
    ap.add_argument("--api", default="http://127.0.0.1:8000", help="Base URL of ML API (default: %(default)s)")
    ap.add_argument("--out", default="results.csv", help="Output CSV path (default: %(default)s)")
    ap.add_argument("--patient-id", default="00000000-0000-0000-0000-000000000000", help="Optional patient UUID")
    ap.add_argument("--views", default="front", help="Comma-separated views, e.g., front,left,right")
    ap.add_argument("--media-type", default="image", choices=["image", "video"], help="Media type")
    ap.add_argument("--serve-local", action="store_true",
                    help="If set, will serve local files via a tiny HTTP server for testing.")
    ap.add_argument("--port", type=int, default=8001, help="Port for local file server (default: %(default)s)")
    args = ap.parse_args()

    items = collect_inputs(args.input)
    if not items:
        print("No inputs collected.", file=sys.stderr)
        sys.exit(2)

    httpd = None
    root_dir = None
    if args.serve_local:
        # choose root as common parent of all local files (ignore URLs)
        local_files = [p for p in items if not (p.startswith("http://") or p.startswith("https://"))]
        if not local_files:
            print("No local files found; --serve-local ignored.", file=sys.stderr)
        else:
            root_dir = os.path.commonpath([os.path.abspath(p) for p in local_files])
            # If commonpath resolves to the file itself, use its directory
            if os.path.isfile(root_dir):
                root_dir = os.path.dirname(root_dir)
            httpd, thread, port = start_local_server(root_dir, args.port)
            print(f"[info] Serving local files from: {root_dir} at http://127.0.0.1:{port}")

    # CSV columns - 6-class format only
    out_cols = [
        "image",
        "overall_status",
        "quality_score",
        "calculus_confidence",
        "caries_confidence",
        "gingivitis_confidence",
        "mouth_ulcer_confidence",
        "tooth_discoloration_confidence",
        "hypodontia_confidence",
        "error"
    ]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols)
        writer.writeheader()
        
        ok = 0
        failed = 0

        for item in tqdm(items, desc="Analyzing"):
            # Resolve to media_url
            media_url = to_media_url(item, args.serve_local, root_dir or ".", args.port)
            if not media_url:
                writer.writerow({
                    "image": item,
                    "overall_status": "",
                    "quality_score": "",
                    "calculus_confidence": "",
                    "caries_confidence": "",
                    "gingivitis_confidence": "",
                    "mouth_ulcer_confidence": "",
                    "tooth_discoloration_confidence": "",
                    "hypodontia_confidence": "",
                    "error": "unsupported_path_without_serve_local"
                })
                failed += 1
                continue

            body = {
                "media_url": media_url,
                "media_type": args.media_type,
                "patient_id": args.patient_id,
                "views": [v.strip() for v in args.views.split(",") if v.strip()]
            }

            # call API with simple retry
            url = args.api.rstrip("/") + "/v1/analyze"
            error_msg = ""
            resp_json = None
            for attempt in range(3):
                try:
                    r = requests.post(url, json=body, timeout=30)
                    if r.status_code == 200:
                        resp_json = r.json()
                        break
                    else:
                        error_msg = f"http_{r.status_code}"
                except Exception as e:
                    error_msg = f"exc_{type(e).__name__}"
                time.sleep(0.5)

            if resp_json is None:
                row = {
                    "image": item,
                    "overall_status": "",
                    "quality_score": "",
                    "calculus_confidence": "",
                    "caries_confidence": "",
                    "gingivitis_confidence": "",
                    "mouth_ulcer_confidence": "",
                    "tooth_discoloration_confidence": "",
                    "hypodontia_confidence": "",
                    "error": error_msg or "unknown_error"
                }
                writer.writerow(row)
                failed += 1
                continue

            try:
                # Build row for 6-class format
                row = {
                    "image": item,
                    "overall_status": resp_json.get("overall_status", ""),
                    "quality_score": resp_json.get("quality", {}).get("score", ""),
                    "calculus_confidence": "",
                    "caries_confidence": "",
                    "gingivitis_confidence": "",
                    "mouth_ulcer_confidence": "",
                    "tooth_discoloration_confidence": "",
                    "hypodontia_confidence": "",
                    "error": ""
                }

                # Map findings to confidences
                for det in resp_json.get("findings", []):
                    label = det.get("type", "")
                    conf = det.get("confidence", "")
                    if label == "Calculus":
                        row["calculus_confidence"] = conf
                    elif label == "Caries":
                        row["caries_confidence"] = conf
                    elif label == "Gingivitis":
                        row["gingivitis_confidence"] = conf
                    elif label == "Mouth Ulcer":
                        row["mouth_ulcer_confidence"] = conf
                    elif label == "Tooth Discoloration":
                        row["tooth_discoloration_confidence"] = conf
                    elif label == "Hypodontia":
                        row["hypodontia_confidence"] = conf

                writer.writerow(row)
                ok += 1
            except Exception as e:
                row = {
                    "image": item,
                    "overall_status": "",
                    "quality_score": "",
                    "calculus_confidence": "",
                    "caries_confidence": "",
                    "gingivitis_confidence": "",
                    "mouth_ulcer_confidence": "",
                    "tooth_discoloration_confidence": "",
                    "hypodontia_confidence": "",
                    "error": f"parse_{type(e).__name__}"
                }
                writer.writerow(row)
                failed += 1

    if httpd is not None:
        httpd.shutdown()

    print(f"\nDone. Wrote: {os.path.abspath(args.out)}")
    print(f"Success: {ok}  |  Failed: {failed}")


if __name__ == "__main__":
    main()
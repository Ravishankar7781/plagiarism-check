"""PlagCheck: plagiarism + AI-writing checker with rewrite suggestions.

    python app.py serve                 open the web app (http://127.0.0.1:8765)
    python app.py index <folder>        add your own files to the comparison index
    python app.py check <file> [...]    check files from the command line, write HTML reports
"""
import argparse
import base64
import json
import os
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import aiscore  # noqa: E402
import detector  # noqa: E402
import rewriter  # noqa: E402
import sources  # noqa: E402
import textutil as tu  # noqa: E402

DB_PATH = os.environ.get("PLAGCHECK_DB", os.path.join(HERE, "plagcheck_index.db"))
UI_PATH = os.path.join(HERE, "static", "index.html")
MAX_CHARS = 300_000

_index = None
_index_lock = threading.Lock()


def get_index():
    global _index
    with _index_lock:
        if _index is None:
            _index = sources.LocalIndex(DB_PATH)
        return _index


# ====================================================================== pipeline
def run_check(text, opts, log=print):
    t0 = time.time()
    text = text.replace("\r\n", "\n")
    if len(text) > MAX_CHARS:
        raise ValueError(f"Text is too long ({len(text):,} characters). The limit is {MAX_CHARS:,}; split it into parts.")
    words = [t[0] for t in tu.tokenize(text)]
    if len(words) < 10:
        raise ValueError("Please give at least a few sentences (10+ words) to check.")
    sents = tu.split_sentences(text)
    warnings, srcs, web_stats = [], [], None

    if opts.get("local", True):
        idx = get_index()
        if idx.count():
            log(f"Comparing with your index ({idx.count():,} files)…")
            for d in idx.candidates(tu.shingles(words)):
                s = idx.load(d)
                if opts.get("exclude_path") and os.path.abspath(s.location) == os.path.abspath(opts["exclude_path"]):
                    continue
                srcs.append(s)
            log(f"  {len(srcs)} similar files found")
        else:
            log("Local index is empty (add folders to compare against your own files).")

    if opts.get("web", True):
        web = sources.WebSearch()
        depth = int(opts.get("depth", 15))
        web_srcs, web_stats = sources.gather_web_sources(text, sents, web, max_queries=depth,
                                                         max_pages=max(10, depth + 5), log=log)
        if web_stats["errors"] and web_stats["errors"] == web_stats["queries"]:
            warnings.append("Web search failed for every query (offline, blocked or rate-limited). "
                            "Results only cover your local index.")
        srcs.extend(web_srcs)

    log(f"Comparing against {len(srcs)} sources…")
    res = detector.analyze(text, srcs)
    ai = aiscore.score(text, res["sentences"])

    if opts.get("ai_review"):
        log("Asking Claude for a second opinion on AI writing…")
        try:
            ai["claude"] = rewriter.claude_review(text)
        except rewriter.ClaudeUnavailable as e:
            warnings.append(str(e))

    if opts.get("rewrite", True):
        _add_suggestions(text, res["sentences"], opts.get("claude", False), log, warnings)

    return {
        "text": text,
        "filename": opts.get("filename"),
        "word_count": res["word_count"],
        "scores": {"similarity": res["similarity_pct"], "exact": res["exact_pct"],
                   "paraphrase": res["paraphrase_pct"], "ai": ai["percent"],
                   "original": round(100 - res["similarity_pct"], 1)},
        "ai": ai,
        "sources": res["sources"],
        "sentences": res["sentences"],
        "web": {"provider": sources.WebSearch().label, **web_stats} if web_stats else None,
        "warnings": warnings,
        "seconds": round(time.time() - t0, 1),
        "checked_at": time.strftime("%Y-%m-%d %H:%M"),
    }


def _issue(e):
    if e["plag"]["type"] != "none":
        return e["plag"]["type"]
    return "ai-style" if e["ai"]["flag"] else None


def _add_suggestions(text, sentences, use_claude, log, warnings):
    todo = [e for e in sentences if _issue(e)]
    if not todo:
        return
    done = {}
    if use_claude:
        log(f"Writing rewrites for {len(todo)} sentences with Claude…")
        items = [{"id": e["i"], "text": e["text"], "issue": _issue(e),
                  "source_excerpt": e["plag"].get("excerpt")} for e in todo[:60]]
        try:
            for k in range(0, len(items), 20):
                done.update(rewriter.claude_rewrite(items[k:k + 20], context=text))
        except rewriter.ClaudeUnavailable as e:
            warnings.append(str(e) + " Showing simpler rule-based suggestions instead.")
    for e in todo:
        e["suggestion"] = done.get(e["i"]) or rewriter.rule_rewrite(e["text"], _issue(e))
        e["issue"] = _issue(e)


# ====================================================================== jobs
JOBS = {}


def start_job(fn, *args):
    jid = uuid.uuid4().hex[:12]
    job = {"status": "running", "log": [], "result": None, "error": None, "started": time.time()}
    JOBS[jid] = job

    def log(msg):
        job["log"].append(msg)

    def work():
        try:
            job["result"] = fn(*args, log=log)
            job["status"] = "done"
        except ValueError as e:
            job["error"], job["status"] = str(e), "error"
        except Exception as e:
            traceback.print_exc()
            job["error"], job["status"] = f"Unexpected error: {e}", "error"

    threading.Thread(target=work, daemon=True).start()
    # forget old jobs
    for k in [k for k, j in JOBS.items() if time.time() - j["started"] > 6 * 3600]:
        JOBS.pop(k, None)
    return jid


def check_job(payload, log):
    text = payload.get("text") or ""
    fname = None
    if payload.get("file"):
        fname = payload["file"]["name"]
        log(f"Reading {fname}…")
        text = tu.extract_text(fname, base64.b64decode(payload["file"]["data"]))
    opts = {k: payload.get(k) for k in ("web", "local", "rewrite", "claude", "ai_review", "depth")}
    opts["filename"] = fname
    return run_check(text, opts, log)


def index_job(folder, log):
    log(f"Indexing {folder}…")
    stats = get_index().add_folder(folder, progress=log)
    log(f"Finished: {stats}")
    return {"stats": stats, "total": get_index().count()}


def report_html(result):
    with open(UI_PATH, encoding="utf-8") as f:
        page = f.read()
    data = json.dumps(result, ensure_ascii=False).replace("</", "<\\/")
    return page.replace("/*__REPORT__*/null", data, 1)


# ====================================================================== server
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        if not isinstance(body, (bytes, bytearray)):
            body = (json.dumps(body, ensure_ascii=False) if "json" in ctype else body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 80_000_000:
            raise ValueError("Upload too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            with open(UI_PATH, "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if p == "/api/status":
            try:
                import pypdf  # noqa: F401
                pdf = True
            except ImportError:
                pdf = False
            return self._send(200, {
                "search": sources.WebSearch().label,
                "search_keyed": sources.WebSearch().provider != "duckduckgo",
                "claude_key": bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")),
                "index_docs": get_index().count(), "pdf": pdf, "model": rewriter.MODEL})
        if p.startswith("/api/job/"):
            job = JOBS.get(p.rsplit("/", 1)[-1])
            if not job:
                return self._send(404, {"error": "Unknown job"})
            return self._send(200, {k: job[k] for k in ("status", "log", "result", "error")})
        if p.startswith("/report/"):
            job = JOBS.get(p.rsplit("/", 1)[-1])
            if not job or not job["result"]:
                return self._send(404, "Report not found", "text/plain; charset=utf-8")
            name = os.path.splitext(job["result"].get("filename") or "text")[0] + "-plagcheck-report.html"
            return self._send(200, report_html(job["result"]), "text/html; charset=utf-8",
                              {"Content-Disposition": f'attachment; filename="{name}"'})
        self._send(404, {"error": "Not found"})

    def do_POST(self):
        try:
            data = self._json()
            if self.path == "/api/check":
                return self._send(200, {"job": start_job(check_job, data)})
            if self.path == "/api/index":
                folder = (data.get("folder") or "").strip().strip('"')
                if not os.path.isdir(folder):
                    return self._send(400, {"error": f"Folder not found: {folder}"})
                return self._send(200, {"job": start_job(index_job, folder)})
            if self.path == "/api/rewrite":
                issue = data.get("issue") or "ai-style"
                if data.get("claude"):
                    try:
                        out = rewriter.claude_rewrite([{"id": 0, "text": data["text"], "issue": issue,
                                                        "source_excerpt": data.get("source_excerpt")}],
                                                      context=data.get("context", ""))
                        if 0 in out:
                            return self._send(200, out[0])
                    except rewriter.ClaudeUnavailable as e:
                        return self._send(200, {**rewriter.rule_rewrite(data["text"], issue), "warning": str(e)})
                return self._send(200, rewriter.rule_rewrite(data["text"], issue))
            self._send(404, {"error": "Not found"})
        except Exception as e:
            self._send(400, {"error": str(e)})


def serve(port, open_browser=True):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"PlagCheck is running at {url}  (press Ctrl+C to stop)")
    print(f"  web search: {sources.WebSearch().label}")
    print(f"  Claude rewrites: {'on' if os.environ.get('ANTHROPIC_API_KEY') else 'set ANTHROPIC_API_KEY to enable'}")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


# ====================================================================== CLI
def main():
    ap = argparse.ArgumentParser(description="Plagiarism + AI-writing checker")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("serve", help="start the web app")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-browser", action="store_true")
    i = sub.add_parser("index", help="add a folder of your own files to the comparison index")
    i.add_argument("folders", nargs="+")
    c = sub.add_parser("check", help="check one or more files")
    c.add_argument("files", nargs="+")
    c.add_argument("--no-web", action="store_true", help="don't search the web")
    c.add_argument("--no-local", action="store_true", help="don't use the local index")
    c.add_argument("--claude", action="store_true", help="use Claude for rewrites (needs ANTHROPIC_API_KEY)")
    c.add_argument("--ai-review", action="store_true", help="ask Claude for a second AI-writing opinion")
    c.add_argument("--depth", type=int, default=15, help="number of web phrase searches (default 15)")
    c.add_argument("--out", default=".", help="folder for the HTML reports")
    c.add_argument("--json", action="store_true", help="also write the raw result as JSON")
    a = ap.parse_args()

    if a.cmd == "index":
        for f in a.folders:
            stats = get_index().add_folder(f, progress=lambda m: print("  " + m))
            print(f"{f}: {stats}")
        print(f"Index now holds {get_index().count():,} files ({DB_PATH})")
    elif a.cmd == "check":
        os.makedirs(a.out, exist_ok=True)
        for path in a.files:
            print(f"\n== {path}")
            try:
                with open(path, "rb") as f:
                    text = tu.extract_text(path, f.read())
                r = run_check(text, {"web": not a.no_web, "local": not a.no_local, "rewrite": True,
                                     "claude": a.claude, "ai_review": a.ai_review, "depth": a.depth,
                                     "filename": os.path.basename(path), "exclude_path": path},
                              log=lambda m: print("  " + m))
            except (ValueError, OSError) as e:
                print(f"  ERROR: {e}")
                continue
            sc = r["scores"]
            print(f"  Similarity {sc['similarity']}%  (exact {sc['exact']}%, paraphrased {sc['paraphrase']}%)"
                  f"  |  AI-writing estimate {sc['ai']}% ({r['ai']['label']})")
            for src in r["sources"][:5]:
                print(f"    {src['percent']:5.1f}%  {src['title'][:60]}  {src['location']}")
            for w in r["warnings"]:
                print(f"  note: {w}")
            base = os.path.join(a.out, os.path.splitext(os.path.basename(path))[0] + "-plagcheck-report")
            with open(base + ".html", "w", encoding="utf-8") as f:
                f.write(report_html(r))
            if a.json:
                with open(base + ".json", "w", encoding="utf-8") as f:
                    json.dump(r, f, ensure_ascii=False, indent=1)
            print(f"  report: {base}.html")
    else:
        serve(getattr(a, "port", 8765), not getattr(a, "no_browser", False))


if __name__ == "__main__":
    main()

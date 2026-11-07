"""Where text gets compared: the web (search engines) and your own indexed folders."""
import concurrent.futures as cf
import html
import json
import os
import re
import sqlite3
import threading
import time
import urllib.parse
import urllib.request
import zlib
from collections import Counter

import textutil as tu
from detector import Source

UA = "PlagCheck/1.0 (local plagiarism checker)"
INDEX_EXTS = {".txt", ".md", ".docx", ".pdf", ".html", ".htm", ".tex", ".rst"}
SAMPLE = 4   # keep 1 in 4 fingerprints in the index (enough to find candidates, 4x smaller)


# ====================================================================== local
class LocalIndex:
    """Fingerprint index of your own files (papers, past submissions, notes...).

    Stored in SQLite, so it scales to millions of documents on one machine:
    a check only looks up its own fingerprints, then compares the top candidates in full.
    """

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.con = sqlite3.connect(path, check_same_thread=False)
        with self.con:
            self.con.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS docs(id INTEGER PRIMARY KEY, path TEXT UNIQUE,
                    mtime REAL, words INTEGER, body BLOB);
                CREATE TABLE IF NOT EXISTS fp(h INTEGER NOT NULL, doc INTEGER NOT NULL);
                CREATE INDEX IF NOT EXISTS fp_h ON fp(h);
                CREATE INDEX IF NOT EXISTS fp_doc ON fp(doc);
            """)

    def count(self):
        with self.lock:
            return self.con.execute("SELECT COUNT(*) FROM docs").fetchone()[0]

    def add_folder(self, folder, progress=None):
        folder = os.path.abspath(folder)
        if not os.path.isdir(folder):
            raise ValueError(f"Folder not found: {folder}")
        stats = Counter()
        pending = 0
        for root, dirs, files in os.walk(folder):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "__pycache__")]
            for name in files:
                if os.path.splitext(name)[1].lower() not in INDEX_EXTS:
                    continue
                path = os.path.join(root, name)
                try:
                    st = os.stat(path)
                    if st.st_size > 50_000_000:
                        stats["skipped"] += 1
                        continue
                    with self.lock:
                        row = self.con.execute("SELECT id, mtime FROM docs WHERE path=?", (path,)).fetchone()
                    if row and row[1] == st.st_mtime:
                        stats["unchanged"] += 1
                        continue
                    with open(path, "rb") as f:
                        text = tu.extract_text(name, f.read())
                    words = [t[0] for t in tu.tokenize(text)]
                    if len(words) < 20:
                        stats["skipped"] += 1
                        continue
                    hs = {h for h in tu.shingles(words) if h % SAMPLE == 0}
                    with self.lock:
                        if row:
                            self.con.execute("DELETE FROM fp WHERE doc=?", (row[0],))
                            self.con.execute("UPDATE docs SET mtime=?, words=?, body=? WHERE id=?",
                                             (st.st_mtime, len(words), zlib.compress(text.encode()), row[0]))
                            doc = row[0]
                            stats["updated"] += 1
                        else:
                            cur = self.con.execute("INSERT INTO docs(path, mtime, words, body) VALUES(?,?,?,?)",
                                                   (path, st.st_mtime, len(words), zlib.compress(text.encode())))
                            doc = cur.lastrowid
                            stats["added"] += 1
                        self.con.executemany("INSERT INTO fp(h, doc) VALUES(?,?)", ((h, doc) for h in hs))
                    pending += 1
                    if pending >= 50:
                        with self.lock:
                            self.con.commit()
                        pending = 0
                    if progress:
                        progress(f"Indexed {stats['added'] + stats['updated']} files — {name}")
                except Exception as e:  # keep going on bad files
                    stats["failed"] += 1
                    if progress:
                        progress(f"Could not read {name}: {e}")
        with self.lock:
            self.con.commit()
        return dict(stats)

    def candidates(self, hashes, top=15, min_hits=2):
        hs = list({h for h in hashes if h % SAMPLE == 0})
        counts = Counter()
        with self.lock:
            for i in range(0, len(hs), 500):
                chunk = hs[i:i + 500]
                q = f"SELECT doc, COUNT(*) FROM fp WHERE h IN ({','.join('?' * len(chunk))}) GROUP BY doc"
                for doc, c in self.con.execute(q, chunk):
                    counts[doc] += c
        return [d for d, c in counts.most_common(top) if c >= min_hits]

    def load(self, doc_id):
        with self.lock:
            path, body = self.con.execute("SELECT path, body FROM docs WHERE id=?", (doc_id,)).fetchone()
        return Source("local", os.path.basename(path), path, zlib.decompress(body).decode())


# ====================================================================== web
def http_get(url, data=None, headers=None, timeout=12, max_bytes=3_000_000):
    h = {"User-Agent": UA, "Accept-Language": "en;q=0.9,*;q=0.5"}
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read(max_bytes)
        return body, r.headers.get("Content-Type", ""), r.geturl()


def _strip_tags(s):
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


class WebSearch:
    """Searches the public web. Picks a provider from environment variables:

    SERPER_API_KEY (Google results via serper.dev), BRAVE_API_KEY (Brave Search API),
    GOOGLE_API_KEY + GOOGLE_CSE_ID (Google Programmable Search). Without any key it
    falls back to DuckDuckGo's HTML page, which works but is slower and rate-limited.
    """

    def __init__(self):
        env = os.environ
        if env.get("SERPER_API_KEY"):
            self.provider = "serper"
        elif env.get("BRAVE_API_KEY"):
            self.provider = "brave"
        elif env.get("GOOGLE_API_KEY") and env.get("GOOGLE_CSE_ID"):
            self.provider = "google"
        else:
            self.provider = "duckduckgo"
        self.parallel = 1 if self.provider == "duckduckgo" else 4

    @property
    def label(self):
        return {"serper": "Google (Serper)", "brave": "Brave Search", "google": "Google Programmable Search",
                "duckduckgo": "DuckDuckGo (no API key, slower)"}[self.provider]

    def search(self, q, n=6):
        env = os.environ
        if self.provider == "serper":
            body, _, _ = http_get("https://google.serper.dev/search",
                                  data=json.dumps({"q": q, "num": n}).encode(),
                                  headers={"X-API-KEY": env["SERPER_API_KEY"], "Content-Type": "application/json"})
            items = json.loads(body).get("organic", [])
            return [{"url": i.get("link"), "title": i.get("title", ""), "snippet": i.get("snippet", "")} for i in items]
        if self.provider == "brave":
            url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode({"q": q, "count": n})
            body, _, _ = http_get(url, headers={"X-Subscription-Token": env["BRAVE_API_KEY"],
                                                "Accept": "application/json"})
            items = json.loads(body).get("web", {}).get("results", [])
            return [{"url": i.get("url"), "title": _strip_tags(i.get("title")),
                     "snippet": _strip_tags(i.get("description"))} for i in items]
        if self.provider == "google":
            url = "https://www.googleapis.com/customsearch/v1?" + urllib.parse.urlencode(
                {"key": env["GOOGLE_API_KEY"], "cx": env["GOOGLE_CSE_ID"], "q": q, "num": min(n, 10)})
            body, _, _ = http_get(url)
            items = json.loads(body).get("items", [])
            return [{"url": i.get("link"), "title": i.get("title", ""), "snippet": i.get("snippet", "")} for i in items]
        # DuckDuckGo HTML endpoint
        body, _, _ = http_get("https://html.duckduckgo.com/html/", data=urllib.parse.urlencode({"q": q}).encode(),
                              headers={"Content-Type": "application/x-www-form-urlencoded"})
        page = body.decode("utf-8", "replace")
        out = []
        for m in re.finditer(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>(.*?)'
                             r'(?:class="result__snippet"[^>]*>(.*?)</a>|</div>\s*</div>)', page, re.S):
            href = html.unescape(m.group(1))
            if "uddg=" in href:
                href = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)["uddg"][0]
            if "duckduckgo.com/y.js" in href or not href.startswith("http"):
                continue  # ads
            out.append({"url": href, "title": _strip_tags(m.group(2)), "snippet": _strip_tags(m.group(4))})
            if len(out) >= n:
                break
        time.sleep(1.0)  # be polite
        return out


def fetch_page(url):
    try:
        if not url.startswith(("http://", "https://")):
            return None
        body, ctype, _ = http_get(url, timeout=10)
        ctype = ctype.lower()
        if "pdf" in ctype or url.lower().endswith(".pdf"):
            return tu.extract_text("x.pdf", body)
        charset = re.search(r"charset=([\w-]+)", ctype)
        raw = body.decode(charset.group(1) if charset else "utf-8", "replace")
        text = tu.html_to_text(raw) if ("html" in ctype or "<html" in raw[:2000].lower()) else raw
        return text[:400_000]
    except Exception:
        return None


def pick_queries(text, sents, max_q):
    """Choose distinctive sentences spread across the document and turn them into exact-phrase queries."""
    cands = []
    for idx, (a, b) in enumerate(sents):
        toks = tu.tokenize(text[a:b])
        if len(toks) < 8:
            continue
        distinct = sum(1 for w, _, _ in toks if w not in tu.STOPWORDS and len(w) >= 5)
        cands.append((idx, distinct, toks, a))
    if not cands:
        return []
    if len(cands) > max_q:
        bucket = len(cands) / max_q
        picked = []
        for k in range(max_q):
            group = cands[int(k * bucket):int((k + 1) * bucket)] or [cands[-1]]
            picked.append(max(group, key=lambda c: c[1]))
        cands = picked
    queries = []
    for idx, _, toks, a in cands:
        mid = max(0, (len(toks) - 10) // 2)
        chunk = toks[mid:mid + 10]
        phrase = text[a + chunk[0][1]: a + chunk[-1][2]]
        phrase = re.sub(r"[\"“”]", "", re.sub(r"\s+", " ", phrase))
        queries.append((idx, f'"{phrase}"'))
    return queries


def gather_web_sources(text, sents, web, max_queries=15, max_pages=20, log=print):
    queries = pick_queries(text, sents, max_queries)
    log(f"Searching the web with {web.label}: {len(queries)} phrase searches…")
    hits = {}
    order = Counter()
    errors = 0

    def run(q):
        try:
            res = web.search(q)
            if not res and q.startswith('"'):
                res = web.search(q.strip('"'))  # no exact hit: try a loose search
            return res
        except Exception as e:
            return e

    with cf.ThreadPoolExecutor(web.parallel) as ex:
        for k, res in enumerate(ex.map(run, [q for _, q in queries]), 1):
            if isinstance(res, Exception):
                errors += 1
                if errors == 1:
                    log(f"Search error: {res}")
                continue
            for r in res:
                u = (r.get("url") or "").split("#")[0]
                if not u:
                    continue
                order[u] += 1
                hits.setdefault(u, r)
            if k % 5 == 0:
                log(f"  {k}/{len(queries)} searches done, {len(hits)} candidate pages")
    urls = [u for u, _ in order.most_common(max_pages)]
    log(f"Reading {len(urls)} candidate pages…")
    sources = []
    with cf.ThreadPoolExecutor(8) as ex:
        for u, page in zip(urls, ex.map(fetch_page, urls)):
            r = hits[u]
            if page and len(page) > 200:
                sources.append(Source("web", r.get("title") or u, u, page))
            elif r.get("snippet"):
                sources.append(Source("snippet", r.get("title") or u, u, r["snippet"]))
    for u, r in hits.items():  # snippets of pages we didn't open still help
        if u not in urls and r.get("snippet"):
            sources.append(Source("snippet", r.get("title") or u, u, r["snippet"]))
    return sources, {"queries": len(queries), "pages": len(urls), "errors": errors}

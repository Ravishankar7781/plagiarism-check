"""Text helpers: file extraction, tokenizing, sentence splitting, fingerprints."""
import hashlib
import io
import re
import zipfile
from html.parser import HTMLParser
from xml.etree import ElementTree as ET

# Words: letters/digits (incl. accented + Indic scripts), with an optional apostrophe part.
WORD_RE = re.compile(r"[\wऀ-෿]+(?:['’][\w]+)?", re.UNICODE)

STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have
having he her here hers herself him himself his how i if in into is it its itself just me more most
my myself no nor not now of off on once only or other our ours ourselves out over own same she should
so some such than that the their theirs them themselves then there these they this those through to
too under until up very was we were what when where which while who whom why will with would you your
yours yourself yourselves also may might must shall upon within without however thus therefore
""".split())

ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
                 "fig", "no", "vol", "inc", "ltd", "co", "approx", "al", "u.s", "jan", "feb", "mar",
                 "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"}


# ---------------------------------------------------------------- extraction
def decode_bytes(data: bytes) -> str:
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16", errors="replace")
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


class _HTMLText(HTMLParser):
    SKIP = {"script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg",
            "button", "select", "template", "iframe"}
    BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section",
             "article", "blockquote", "pre", "td", "dd", "dt"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def html_to_text(markup: str) -> str:
    p = _HTMLText()
    try:
        p.feed(markup)
        p.close()
    except Exception:
        pass
    text = "".join(p.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _docx_text(data: bytes) -> str:
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    paras = []
    for p in root.iter(ns + "p"):
        parts = []
        for node in p.iter():
            if node.tag == ns + "t" and node.text:
                parts.append(node.text)
            elif node.tag == ns + "tab":
                parts.append("\t")
            elif node.tag in (ns + "br", ns + "cr"):
                parts.append("\n")
        paras.append("".join(parts))
    return "\n".join(paras)


def _pdf_text(data: bytes) -> str:
    try:
        import pypdf  # optional dependency
    except ImportError:
        raise ValueError("PDF support needs one extra package. Run:  pip install pypdf")
    reader = pypdf.PdfReader(io.BytesIO(data))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    # PDFs wrap lines mid-sentence; join lines that don't end a sentence.
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"(?<![.!?:;\n])\n(?!\n)", " ", text)
    return text


def extract_text(filename: str, data: bytes) -> str:
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext == "docx":
        return _docx_text(data)
    if ext == "pdf":
        return _pdf_text(data)
    if ext in ("html", "htm", "xhtml"):
        return html_to_text(decode_bytes(data))
    if ext in ("doc", "odt", "pages"):
        raise ValueError(f".{ext} files aren't supported. Save the file as .docx, .pdf or .txt first.")
    return decode_bytes(data)


# ---------------------------------------------------------------- tokens
def tokenize(text: str):
    """Return [(lowercase_word, start, end), ...]."""
    return [(m.group(0).lower().replace("’", "'"), m.start(), m.end())
            for m in WORD_RE.finditer(text)]


def h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big") >> 1


def shingles(words, k=5):
    return [h64(" ".join(words[i:i + k])) for i in range(len(words) - k + 1)]


def stem(w: str) -> str:
    for suf in ("ingly", "edly", "ing", "ed", "ies", "es", "s", "ly"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def content_stems(words):
    return {stem(w) for w in words if w not in STOPWORDS and len(w) > 2 and not w.isdigit()}


# ---------------------------------------------------------------- sentences
_BOUNDARY = re.compile(r"([.!?]+[\"'”’)\]]*)(\s+)(?=[A-Z0-9\"'“‘(\[ऀ-෿])"
                       r"|(\n+)|([।॥])")


def split_sentences(text: str):
    """Return a list of (start, end) character spans, one per sentence or line."""
    spans, start = [], 0
    for m in _BOUNDARY.finditer(text):
        if m.group(1):
            end = m.end(1)
            before = re.search(r"([\w.]+)\.+$", text[start:m.start(1) + 1])
            word = (before.group(1) if before else "").lower().rstrip(".")
            if word in ABBREVIATIONS or (len(word) == 1 and word.isalpha()):
                continue  # "Dr. Smith", "J. Doe" -> not a sentence end
        elif m.group(4):
            end = m.end(4)
        else:
            end = m.start(3)
        _add_span(text, start, end, spans)
        start = m.end()
    _add_span(text, start, len(text), spans)
    return spans


def _add_span(text, a, b, spans):
    seg = text[a:b]
    stripped = seg.strip()
    if not stripped:
        return
    a += len(seg) - len(seg.lstrip())
    spans.append((a, a + len(stripped)))

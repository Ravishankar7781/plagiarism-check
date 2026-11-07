"""Plagiarism matching: exact overlap (word 5-gram fingerprints) + close-paraphrase detection."""
from collections import Counter, defaultdict
from dataclasses import dataclass

import textutil as tu

K = 5            # words per fingerprint
MIN_RUN = 8      # ignore matched runs shorter than this many words (common phrases)
COPIED = 0.5     # sentence coverage at/above this = "copied"
PARA_MIN = 0.6   # share of a sentence's content words found in one source sentence = "paraphrased"


@dataclass
class Source:
    kind: str        # "web" | "snippet" | "local"
    title: str
    location: str    # URL or file path
    text: str
    id: int = -1


def _sentence_of_tokens(toks, sents):
    out, si = [], 0
    for _, a, _ in toks:
        while si < len(sents) - 1 and a >= sents[si][1]:
            si += 1
        out.append(si)
    return out


def analyze(text, sources, min_run=MIN_RUN):
    toks = tu.tokenize(text)
    words = [t[0] for t in toks]
    n = len(words)
    sents = tu.split_sentences(text)
    for i, s in enumerate(sources):
        s.id = i

    # ---- fingerprint every source
    index = {}                       # hash -> [(source, token position)]
    src_toks = []
    for s in sources:
        st = tu.tokenize(s.text)
        src_toks.append(st)
        seen = set()
        for p, h in enumerate(tu.shingles([t[0] for t in st], K)):
            if h not in seen:
                seen.add(h)
                index.setdefault(h, []).append((s.id, p))

    # ---- exact matches
    inp = tu.shingles(words, K)
    hits = [index.get(h) for h in inp]
    totals = Counter()
    for m in hits:
        if m:
            for sid, _ in m:
                totals[sid] += 1
    cover = [-1] * n                 # which source covers each input word
    cover_pos = [None] * n           # position of that word in the source
    for i, m in enumerate(hits):
        if not m:
            continue
        sid, p = max(m, key=lambda x: totals[x[0]])
        for j in range(K):
            w = i + j
            if cover[w] == -1 or totals[sid] > totals[cover[w]]:
                cover[w], cover_pos[w] = sid, p + j

    # drop short runs (stock phrases like "on the other hand")
    i = 0
    while i < n:
        if cover[i] == -1:
            i += 1
            continue
        j = i
        while j < n and cover[j] != -1:
            j += 1
        if j - i < min_run:
            for w in range(i, j):
                cover[w], cover_pos[w] = -1, None
        i = j

    # ---- paraphrase index over source sentences
    src_sents = []                   # (source id, start, end, content stems)
    post = defaultdict(list)         # stem -> source sentence numbers
    for s, st in zip(sources, src_toks):
        ti = 0
        for a, b in tu.split_sentences(s.text):
            ws = []
            while ti < len(st) and st[ti][1] < b:
                if st[ti][1] >= a:
                    ws.append(st[ti][0])
                ti += 1
            stems = tu.content_stems(ws)
            for x in stems:
                post[x].append(len(src_sents))
            src_sents.append((s.id, a, b, stems))

    # ---- per-sentence results
    tok_sent = _sentence_of_tokens(toks, sents)
    by_sent = defaultdict(list)
    for ti, si in enumerate(tok_sent):
        by_sent[si].append(ti)

    src_words = Counter()
    para_words = 0
    results = []
    for si, (a, b) in enumerate(sents):
        tis = by_sent.get(si, [])
        covered = [t for t in tis if cover[t] != -1]
        cov = len(covered) / len(tis) if tis else 0.0
        entry = {"i": si, "start": a, "end": b, "text": text[a:b], "words": len(tis),
                 "plag": {"type": "none", "coverage": round(cov, 3)}, "matches": []}
        # character ranges of exact matches inside this sentence
        run = None
        for t in tis:
            if cover[t] != -1:
                if run and run[2] == cover[t] and run[3] == t - 1:
                    run[1], run[3] = toks[t][2], t
                else:
                    if run:
                        entry["matches"].append(run[:3])
                    run = [toks[t][1], toks[t][2], cover[t], t]
            elif run:
                entry["matches"].append(run[:3])
                run = None
        if run:
            entry["matches"].append(run[:3])

        if covered:
            best = Counter(cover[t] for t in covered).most_common(1)[0][0]
            for t in covered:
                src_words[cover[t]] += 1
            positions = [cover_pos[t] for t in covered if cover[t] == best]
            entry["plag"].update(type="copied" if cov >= COPIED else "partial", source_id=best,
                                 excerpt=_excerpt(sources[best].text, src_toks[best],
                                                  min(positions), max(positions)))
        if cov < COPIED and tis:
            stems = tu.content_stems(words[t] for t in tis)
            if len(stems) >= 6:
                c = Counter()
                for x in stems:
                    plist = post.get(x, ())
                    if len(plist) <= 400:
                        c.update(plist)
                if c:
                    k_best, overlap = max(c.items(), key=lambda kv: (kv[1], -len(src_sents[kv[0]][3])))
                    ratio = overlap / len(stems)
                    if ratio >= PARA_MIN and overlap >= 5:
                        sid, sa, sb, _ = src_sents[k_best]
                        entry["plag"].update(type="paraphrased", source_id=sid,
                                             similarity=round(ratio, 3),
                                             excerpt=sources[sid].text[sa:sb].strip()[:600])
                        para_words += len(tis) - len(covered)
                        src_words[sid] += len(tis) - len(covered)
        results.append(entry)

    exact = sum(1 for c in cover if c != -1)
    pct = lambda x: round(100.0 * x / n, 1) if n else 0.0
    src_list = []
    for s in sources:
        if src_words[s.id]:
            src_list.append({"id": s.id, "kind": s.kind, "title": s.title, "location": s.location,
                             "words": src_words[s.id], "percent": pct(src_words[s.id])})
    src_list.sort(key=lambda x: -x["words"])
    return {
        "word_count": n,
        "exact_pct": pct(exact),
        "paraphrase_pct": pct(para_words),
        "similarity_pct": min(100.0, pct(exact + para_words)),
        "sentences": results,
        "sources": src_list,
    }


def _excerpt(text, st, p0, p1, pad=12):
    a = st[max(0, p0 - pad)][1]
    b = st[min(len(st) - 1, p1 + pad)][2]
    out = text[a:b].replace("\n", " ").strip()
    if p0 - pad > 0:
        out = "…" + out
    if p1 + pad < len(st) - 1:
        out += "…"
    return out[:700]

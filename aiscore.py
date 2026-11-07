"""AI-writing estimate from style signals.

No detector can prove who wrote a text. This scores patterns that are common in
machine-generated prose (uniform sentence length, stock phrases, heavy
transitions...). Treat the number as a hint for where to revise, never as proof.
"""
import re
import statistics

import textutil as tu

STRONG = ["delve", "delves", "delving", "tapestry", "testament to", "in today's fast-paced",
          "in today's digital", "in today's world", "ever-evolving", "ever-changing", "realm of",
          "navigate the complexities", "navigating the complexities", "it is important to note",
          "it's important to note", "it is worth noting", "it's worth noting", "plays a crucial role",
          "plays a pivotal role", "plays a vital role", "a pivotal role", "a crucial role",
          "multifaceted", "harness the power", "unlock the potential", "unlocking the potential",
          "a myriad of", "a plethora of", "intricacies", "underscores the", "stands as a",
          "serves as a testament", "rich tapestry", "embark on", "game-changer", "in the realm",
          "at its core", "in essence", "a beacon", "nestled", "seamlessly", "meticulously",
          "commendable", "fostering", "foster a", "paramount", "key takeaways", "let's dive",
          "dive into", "deep dive", "shed light on", "sheds light on", "a testament",
          "landscape of", "transformative", "elevate your", "revolutionize", "unwavering"]
WEAK = ["furthermore", "moreover", "additionally", "in conclusion", "in summary", "overall",
        "ultimately", "notably", "consequently", "comprehensive", "robust", "seamless", "leverage",
        "leveraging", "holistic", "nuanced", "pivotal", "crucial", "vital", "vibrant", "intricate",
        "showcase", "showcasing", "dynamic", "empower", "streamline", "insights", "journey",
        "not only", "when it comes to", "a wide range of", "a variety of", "it can be argued",
        "in addition", "enhance", "enhancing", "significant", "significantly", "various"]
TRANSITIONS = re.compile(r"^(furthermore|moreover|additionally|in addition|however|overall|"
                         r"ultimately|consequently|therefore|thus|in conclusion|in summary|"
                         r"notably|importantly|similarly|likewise|as a result|on the other hand|"
                         r"by contrast|to summarize|in essence|firstly|secondly|lastly|finally)\b",
                         re.I)
TRICOLON = re.compile(r"\b[\w-]+(?: [\w-]+)?, [\w-]+(?: [\w-]+)?,? and [\w-]+", re.I)
CONTRACTION = re.compile(r"\b\w+'(?:t|s|re|ve|ll|d|m)\b", re.I)


def _phrase_hits(low):
    strong = [p for p in STRONG if re.search(r"(?<!\w)" + re.escape(p) + r"(?!\w)", low)]
    weak = [p for p in WEAK if re.search(r"(?<!\w)" + re.escape(p) + r"(?!\w)", low)]
    return strong, weak


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def score(text, sentence_entries):
    """sentence_entries: list of dicts with 'text' and 'words'. Adds entry['ai']. Returns summary."""
    total_words = sum(e["words"] for e in sentence_entries) or 1
    lengths = [e["words"] for e in sentence_entries if e["words"] >= 3]
    mean = statistics.mean(lengths) if lengths else 0
    flagged = 0
    strong_all = weak_all = trans = 0

    for e in sentence_entries:
        s = e["text"].strip()
        low = s.lower().replace("’", "'")
        strong, weak = _phrase_hits(low)
        strong_all += len(strong)
        weak_all += len(weak)
        reasons, sc = [], 0.0
        pw = len(strong) + 0.4 * len(weak)
        if pw:
            sc += 0.5 * clamp(pw / 1.4)
            reasons.append("stock phrases: " + ", ".join((strong + weak)[:4]))
        if TRANSITIONS.match(s):
            sc += 0.2
            trans += 1
            reasons.append("opens with a formal transition")
        if "—" in s:
            sc += 0.1
            reasons.append("em dash")
        if TRICOLON.search(s):
            sc += 0.08
            reasons.append("list of three")
        if mean and 14 <= e["words"] <= 35 and abs(e["words"] - mean) <= 0.2 * mean:
            sc += 0.12
            reasons.append("same length as most sentences")
        if e["words"] < 4:
            sc = 0.0
        sc = round(clamp(sc), 2)
        flag = sc >= 0.45
        flagged += flag
        e["ai"] = {"score": sc, "flag": flag, "reasons": reasons if sc > 0 else []}

    # ---- document-level signals (each 0..1, higher = more machine-like)
    cv = (statistics.pstdev(lengths) / mean) if len(lengths) >= 5 and mean else None
    burst = clamp((0.65 - cv) / 0.4) if cv is not None else 0.5
    density = (strong_all + 0.4 * weak_all) / (total_words / 100)
    phrases = clamp(density / 2.0)
    n_sent = max(1, len(sentence_entries))
    transitions = clamp((trans / n_sent) / 0.3)
    emdash = clamp(text.count("—") / (total_words / 100) / 0.8)
    contr = len(CONTRACTION.findall(text.replace("’", "'")))
    no_contractions = 1 - clamp(contr / (total_words / 100) / 1.0)
    flagged_share = flagged / n_sent

    signals = {
        "uniform_sentence_length": round(burst, 2),
        "stock_phrases": round(phrases, 2),
        "formal_transitions": round(transitions, 2),
        "em_dashes": round(emdash, 2),
        "no_contractions": round(no_contractions, 2),
        "flagged_sentences": round(flagged_share, 2),
    }
    raw = (0.28 * burst + 0.30 * phrases + 0.14 * transitions + 0.08 * emdash +
           0.08 * no_contractions + 0.12 * flagged_share)
    pct = round(100 * clamp(raw), 1)
    if total_words < 80:
        label = "Too short to judge"
    elif pct >= 60:
        label = "Likely AI-written"
    elif pct >= 35:
        label = "Mixed / unclear"
    else:
        label = "Likely human-written"
    return {"percent": pct, "label": label, "signals": signals,
            "sentence_cv": round(cv, 2) if cv is not None else None,
            "note": "Estimate from writing-style patterns. It can be wrong both ways "
                    "(formal human writing can score high), so use it to find sentences to revise, not as proof."}

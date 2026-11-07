"""Rewrite suggestions: Claude when an API key is available, simple rule-based edits otherwise."""
import json
import os
import re

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None

MODEL = os.environ.get("PLAGCHECK_MODEL", "claude-opus-5-5")

SYSTEM = """You help writers revise their own drafts so the wording is original and reads like a person wrote it.

For each sentence you get, write one revised version that:
- keeps the meaning, facts, numbers, names and technical terms exactly;
- matches the register of the surrounding text (an academic essay stays academic, a blog stays casual);
- uses plain, specific words and a natural rhythm: vary sentence length, drop filler and stock phrases
  ("delve", "it is important to note", "plays a crucial role", "in today's fast-paced world", "furthermore"...);
- never invents facts, sources, quotes or citations.

Issue types:
- "copied" / "partial": the wording matches a published source. Rebuild the sentence with a new structure and
  your own words (not synonym swapping). The idea still came from that source, so the note must remind the
  writer to cite it.
- "paraphrased": too close to a source sentence. Change the structure and wording more, and remind them to cite.
- "ai-style": reads like generated text. Make it sound like a thoughtful human writer.

The note is one short sentence telling the writer what you changed or what they still need to do."""

SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "rewrite": {"type": "string"},
                               "note": {"type": "string"}},
                "required": ["id", "rewrite", "note"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "ai_likelihood": {"type": "integer"},
        "verdict": {"type": "string"},
        "signals": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["ai_likelihood", "verdict", "signals"],
    "additionalProperties": False,
}


class ClaudeUnavailable(Exception):
    pass


def _client():
    if anthropic is None:
        raise ClaudeUnavailable("The 'anthropic' package is not installed (pip install anthropic).")
    # Uses ANTHROPIC_API_KEY, or a saved `ant auth login` profile.
    return anthropic.Anthropic()


def _call(system, user, schema, effort="medium"):
    try:
        client = _client()
        resp = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        )
    except ClaudeUnavailable:
        raise
    except anthropic.AuthenticationError:
        raise ClaudeUnavailable("No valid Anthropic API key. Set ANTHROPIC_API_KEY to enable Claude rewrites.")
    except anthropic.RateLimitError:
        raise ClaudeUnavailable("Claude API rate limit reached. Try again in a minute.")
    except anthropic.APIConnectionError:
        raise ClaudeUnavailable("Could not reach the Claude API (network error).")
    except anthropic.APIStatusError as e:
        raise ClaudeUnavailable(f"Claude API error {e.status_code}: {e.message}")
    except Exception as e:  # e.g. missing credentials raised by the client
        raise ClaudeUnavailable(f"Claude is not available: {e}")
    if resp.stop_reason == "refusal":
        raise ClaudeUnavailable("Claude declined this request.")
    if resp.stop_reason == "max_tokens":
        raise ClaudeUnavailable("Claude's answer was cut off; try fewer sentences at once.")
    text = next((b.text for b in resp.content if b.type == "text"), "")
    return json.loads(text)


def claude_rewrite(items, context=""):
    """items: [{id, text, issue, source_excerpt?}] -> {id: {rewrite, note, by}}"""
    lines = []
    for it in items:
        block = {"id": it["id"], "issue": it["issue"], "sentence": it["text"]}
        if it.get("source_excerpt"):
            block["matching_source_text"] = it["source_excerpt"]
        lines.append(block)
    user = ("Surrounding text (for tone only, do not rewrite it):\n<context>\n" + context[:6000] +
            "\n</context>\n\nSentences to revise:\n" + json.dumps(lines, ensure_ascii=False, indent=1))
    data = _call(SYSTEM, user, SCHEMA)
    return {int(x["id"]): {"rewrite": x["rewrite"].strip(), "note": x["note"].strip(), "by": "claude"}
            for x in data.get("items", [])}


def claude_review(text):
    system = ("You estimate how likely a text is to be machine-generated. Be calibrated and honest: no one can "
              "prove authorship from text alone, and formal human writing often looks 'AI-like'. Give a 0-100 "
              "likelihood, a one-sentence verdict, and up to five concrete signals you noticed.")
    return _call(system, "<text>\n" + text + "\n</text>", REVIEW_SCHEMA, effort="low")


# ---------------------------------------------------------------- no-API fallback
_SWAPS = [
    (r"\bit is (important|worth|crucial) to (note|mention|remember) that\s*", ""),
    (r"\bit'?s (important|worth) (noting|to note) that\s*", ""),
    (r"\bin today'?s (fast-paced|digital|modern|ever-changing) (world|age|era|landscape),?\s*", ""),
    (r"\bplays an? (crucial|pivotal|vital|key|significant|important) role in\b", "helps with"),
    (r"\bnavigat(e|es|ing) the complexities of\b", "deal with"),
    (r"\bdelves? into\b", "looks at"), (r"\bdelving into\b", "looking at"),
    (r"\ba (myriad|plethora) of\b", "many"), (r"\bmyriad\b", "many"),
    (r"\butili[sz]e[sd]?\b", "use"), (r"\butili[sz]ing\b", "using"),
    (r"\bleverag(e|es)\b", "use"), (r"\bleveraging\b", "using"),
    (r"\bin order to\b", "to"), (r"\bdue to the fact that\b", "because"),
    (r"\bfacilitate[sd]?\b", "help"), (r"\bcommence[sd]?\b", "start"),
    (r"\bseamlessly\b", "smoothly"), (r"\bmeticulously\b", "carefully"),
    (r"\bharness(es)? the power of\b", "use"), (r"\bunlock(s)? the potential of\b", "get more from"),
    (r"\bembark(s)? on\b", "start"), (r"\bshed(s)? light on\b", "explain"),
    (r"\bserves as a testament to\b", "shows"), (r"\bis a testament to\b", "shows"),
    (r"\bin the realm of\b", "in"), (r"\bat its core,?\s*", ""), (r"\bin essence,?\s*", ""),
    (r"\bmultifaceted\b", "complex"), (r"\bparamount\b", "essential"),
]
_LEADING = re.compile(r"^(Furthermore|Moreover|Additionally|In addition|Notably|Importantly|Ultimately|"
                      r"Overall|Consequently),\s+")


def rule_rewrite(sentence, issue):
    if issue in ("copied", "partial", "paraphrased"):
        return {"rewrite": "", "by": "rules",
                "note": ("This wording matches a source. Put it in quotation marks and cite the source, or "
                         "explain the idea in your own words and still cite it. Turn on Claude for a full rewrite.")}
    s = sentence
    for pat, rep in _SWAPS:
        s = re.sub(pat, rep, s, flags=re.I)
    s = _LEADING.sub("", s)
    words = s.split()
    if len(words) > 32:
        for sep in ("; ", ", and ", ", which ", ", but "):
            k = s.find(sep, len(s) // 3)
            if 0 < k < len(s) * 0.75:
                head, tail = s[:k], s[k + len(sep):]
                tail = {"; ": "", ", and ": "", ", which ": "This ", ", but ": "But "}[sep] + tail
                s = head.rstrip(",") + ". " + tail[:1].upper() + tail[1:]
                break
    s = re.sub(r"\s{2,}", " ", s).strip()
    if s:
        s = s[0].upper() + s[1:]
    changed = s != sentence.strip()
    return {"rewrite": s if changed else "", "by": "rules",
            "note": ("Removed stock phrases and filler." if changed else
                     "Reads as formulaic. Try adding a concrete detail, example, or your own view.")}

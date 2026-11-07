# PlagCheck

Plagiarism + AI-writing checker for text and files (.docx, .pdf, .txt, .md, .html).

- **Similarity %**: how much of your text matches the web or your own files, split into exact matches and close paraphrases, with every matching source listed.
- **AI-writing %**: an estimate from writing-style signals (even sentence lengths, stock phrases like "delve" or "plays a crucial role", heavy "Moreover/Furthermore" transitions, and more).
- **Where to change**: flagged sentences are highlighted. Click one to see the matching source text, why it was flagged, and a suggested rewrite you can edit and apply.
- **Revised text**: apply rewrites, then copy or download the revised text and an HTML report.

## Start

Double-click `run.bat`, or:

```
python app.py serve
```

It opens http://127.0.0.1:8765 in your browser. Everything runs on your computer; text is only sent out as short search phrases (and to Claude if you turn rewrites on).

## Make it better (optional keys)

Set these as environment variables before starting:

| Variable | What it does |
|---|---|
| `ANTHROPIC_API_KEY` | Claude writes natural, human-sounding rewrites and can give a second opinion on AI writing. Without it, simple built-in rules are used. |
| `SERPER_API_KEY` | Search Google's index (serper.dev, free tier available). Best coverage. |
| `BRAVE_API_KEY` | Search with the Brave Search API instead. |
| `GOOGLE_API_KEY` + `GOOGLE_CSE_ID` | Google Programmable Search instead. |

With no search key it uses DuckDuckGo, which works but is slower and can get rate-limited.

PowerShell example:

```
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:SERPER_API_KEY = "..."
python app.py serve
```

## Your own files ("billions of files")

No tool can open billions of files for every check. PlagCheck does what commercial checkers do:

1. **The web**: it picks distinctive phrases from your text, searches them in a search engine that already indexes billions of pages, then downloads and compares the top pages.
2. **Your collections**: add folders (papers, past submissions, books, notes). Each file is fingerprinted once into a local database (`plagcheck_index.db`), so a check only looks up its own fingerprints and compares the few files that match. This scales to millions of files on one machine.

```
python app.py index "D:\papers" "D:\submissions"
```

You can also add folders from the web app. Re-running skips unchanged files.

Tip: index a folder of student submissions to catch students copying from each other.

## Command line

```
python app.py check essay.docx report2.pdf --claude --out reports
```

Prints the scores and top sources, and writes an HTML report per file. Options: `--no-web`, `--no-local`, `--depth 30` (more web searches), `--ai-review`, `--json`.

## Things to know

- **AI detection is an estimate.** No tool can prove who wrote a text. Formal human writing can score high and edited AI text can score low. Use the score to find sentences to revise, not as proof.
- **Rewriting does not remove the need to cite.** If an idea came from a source, cite it even after rewording. Suggestions remind you of this.
- Similarity only covers what was searched: the web phrases picked (raise Depth for long documents) and your indexed files. Paywalled journals and private databases aren't searchable.
- Matches under 8 words in a row are ignored (common phrases).

## Files

`app.py` (server + CLI), `detector.py` (matching), `aiscore.py` (AI-writing estimate), `sources.py` (web search + local index), `rewriter.py` (Claude + rule-based rewrites), `textutil.py` (file reading, sentences), `static/index.html` (the app), `samples/` (test files).

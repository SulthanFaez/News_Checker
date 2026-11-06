"""Lean fact-checker: DuckDuckGo search snippets + two small Groq LLM calls.

Why this uses far fewer tokens than Groq's built-in browser search:
the model only ever sees short search snippets (a few hundred tokens), never
whole web pages. Source URLs are attached by this code from the real search
results, so the model cannot invent a citation.

Flow:  extract claims (LLM) -> search each claim (ddgs) -> judge from snippets (LLM)
"""
import json
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

# --------------------------------------------------------------------------- #
# Settings (the Streamlit app overrides these at run time)
# --------------------------------------------------------------------------- #
LLM_MODEL = "qwen/qwen3.8-27b"           # no browsing needed, so any good Groq chat model works
FALLBACK_MODEL = "openai/gpt-oss-20b"    # used if the main model is rate-limited or failing
MAX_CLAIMS = 5
RESULTS_PER_CLAIM = 5
MAX_CONTEXT_CHARS = 8000                 # article text sent for claim extraction
SNIPPET_CHARS = 300                      # per search result sent to the judge
MAX_COMPLETION_TOKENS = 2500
MAX_RETRIES = 4
MAX_WAIT_SECONDS = 60                    # never sleep longer; switch model instead
SEARCH_WORKERS = 3                       # keep low: DuckDuckGo throttles heavy parallel use

VERDICTS = ("supported", "contradicted", "mixed", "unverified")


# --------------------------------------------------------------------------- #
# LLM helpers
# --------------------------------------------------------------------------- #
def _parse_json(text: str) -> dict:
    """Pull one JSON object out of a model reply (handles <think> blocks and code fences)."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in reply")
    return json.loads(text[start:end + 1])


def _parse_retry_delay(msg: str) -> float | None:
    """Parse Groq's 'try again in 6.5s' / '1m12s' / '350ms' hints."""
    m = re.search(r"try again in\s*(?:(\d+)m)?\s*(\d+(?:\.\d+)?)(ms|s)", msg, re.I)
    if not m:
        return None
    seconds = float(m.group(2)) / (1000 if m.group(3).lower() == "ms" else 1)
    return int(m.group(1) or 0) * 60 + seconds


def _status_code(msg: str) -> int | None:
    m = re.search(r"Error code:\s*(\d{3})", msg)
    return int(m.group(1)) if m else None


def ask_json(client, prompt: str) -> tuple[dict, str]:
    """Send one prompt, return (parsed_json, model_used). Fails over to FALLBACK_MODEL."""
    models = [LLM_MODEL]
    if FALLBACK_MODEL and FALLBACK_MODEL != LLM_MODEL:
        models.append(FALLBACK_MODEL)

    last_err = "unknown error"
    for model in models:
        json_mode = True
        retries = 0
        parse_failures = 0
        while True:
            kwargs = dict(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
            )
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if model.startswith("openai/gpt-oss"):
                kwargs["reasoning_effort"] = "low"

            try:
                resp = client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                try:
                    return _parse_json(text), model
                except ValueError:
                    parse_failures += 1
                    last_err = f"{model} returned invalid JSON"
                    if parse_failures >= 2:
                        break  # give up on this model, try the next one
                    continue
            except Exception as e:
                msg = str(e)
                low = msg.lower()
                status = _status_code(msg)
                last_err = msg[:300]

                if status in (401, 402, 403) or "invalid_api_key" in low:
                    raise RuntimeError(f"Groq auth/billing problem - check your key and plan: {msg[:300]}")

                # Some models reject JSON mode; fall back to parsing the text ourselves.
                if json_mode and status == 400 and ("response_format" in low or "json" in low):
                    json_mode = False
                    continue

                rate_limited = status == 429 or "rate_limit" in low
                hinted = _parse_retry_delay(msg)
                if rate_limited:
                    daily = any(k in low for k in ("tokens per day", "(tpd)", "requests per day", "(rpd)"))
                    if daily or (hinted is not None and hinted > MAX_WAIT_SECONDS):
                        break  # sleeping won't help; try the next model

                transient = rate_limited or status in (500, 502, 503, 504) or "overloaded" in low
                if transient and retries < MAX_RETRIES:
                    retries += 1
                    wait = (hinted + 1) if hinted else min(30, 3 * 2 ** retries)
                    print(f"  Groq retry in {wait:.0f}s ({msg[:120]})")
                    time.sleep(wait + random.uniform(0, 1))
                    continue
                break  # non-retryable: try the next model

    raise RuntimeError(f"All models failed. Last error: {last_err}")


# --------------------------------------------------------------------------- #
# Step 1: extract claims
# --------------------------------------------------------------------------- #
def extract_claims(client, article: dict) -> tuple[list[dict], str]:
    body = "\n\n".join(article["paragraphs"])[:MAX_CONTEXT_CHARS]
    prompt = (
        f"Article title: {article.get('title')}\n\n"
        f"Article text:\n{body}\n\n"
        f"Extract the {MAX_CLAIMS} most important CHECKABLE factual claims from this article: "
        "specific statistics, dates, named studies or experiments, historical facts, and "
        "cause-and-effect statements presented as fact. Skip opinions, advice, and personal anecdotes.\n"
        'For each claim give: "claim" (self-contained, max 25 words) and "query" '
        "(a web search query of at most 12 words likely to find evidence about it).\n"
        'Return ONLY JSON: {"claims": [{"claim": "...", "query": "..."}]}'
    )
    data, model = ask_json(client, prompt)
    claims = []
    for c in (data.get("claims") or [])[:MAX_CLAIMS]:
        if not isinstance(c, dict):
            continue
        claim = str(c.get("claim", "")).strip()
        query = str(c.get("query", "")).strip() or claim
        if claim:
            claims.append({"claim": claim, "query": query})
    return claims, model


# --------------------------------------------------------------------------- #
# Step 2: search
# --------------------------------------------------------------------------- #
def search_one(query: str, exclude_domain: str = "") -> list[dict]:
    """DuckDuckGo search. Skips the article's own site so results are independent sources."""
    try:
        from ddgs import DDGS
    except ImportError:
        raise RuntimeError("The search package is missing. Run: pip install ddgs")

    for attempt in range(3):
        try:
            raw = DDGS().text(query, max_results=RESULTS_PER_CLAIM + 4)
            out = []
            for r in raw or []:
                url = r.get("href") or r.get("url") or ""
                if not url:
                    continue
                host = urlparse(url).netloc.lower().removeprefix("www.")
                if exclude_domain and (host == exclude_domain or host.endswith("." + exclude_domain)):
                    continue
                out.append({
                    "title": r.get("title") or url,
                    "url": url,
                    "snippet": re.sub(r"\s+", " ", r.get("body") or "")[:SNIPPET_CHARS],
                })
                if len(out) >= RESULTS_PER_CLAIM:
                    break
            return out
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return []


# --------------------------------------------------------------------------- #
# Step 3: judge claims from snippets
# --------------------------------------------------------------------------- #
JUDGE_FORMAT = (
    'Return ONLY JSON: {"overall": "...", "verdicts": [{"id": 1, "verdict": "supported", '
    '"explanation": "...", "evidence": ["1.1", "1.3"]}]}'
)


def judge(client, article: dict, claims: list[dict]) -> tuple[list[dict], str, str]:
    blocks = []
    for i, c in enumerate(claims, 1):
        lines = [f"CLAIM {i}: {c['claim']}"]
        if c["results"]:
            for j, r in enumerate(c["results"], 1):
                host = urlparse(r["url"]).netloc
                lines.append(f"  [{i}.{j}] {r['title']} ({host}): {r['snippet']}")
        else:
            lines.append("  (no search results found)")
        blocks.append("\n".join(lines))

    prompt = (
        f'You are fact-checking claims from the article "{article.get("title")}". '
        "Each claim is followed by web search snippets.\n\n"
        "Rules:\n"
        "- Base every verdict ONLY on the snippets shown. Do not use your own knowledge.\n"
        '- verdict is one of: "supported" (snippets directly back the claim), '
        '"contradicted" (snippets directly conflict with it), "mixed" (partly), '
        '"unverified" (snippets do not address it, or there are none).\n'
        "- Use \"contradicted\" ONLY if a snippet measures the SAME thing in the SAME sample, population, "
        "country and time period as the claim. A different figure for a subset (e.g. only milks, when the "
        "claim covers milks, yoghurts and cheeses), or from a different study or year, does NOT contradict "
        "the claim: use \"unverified\" or \"mixed\" and say the figures are not comparable.\n"
        "- explanation: one or two sentences, max 40 words.\n"
        '- evidence: snippet ids you relied on (like "2.1"). Use only ids shown. '
        "Empty list when unverified.\n"
        "- overall: 2 or 3 plain sentences saying what these checked claims suggest about the "
        "article. Mention that only these claims were checked and only from search snippets. "
        "Do not judge claims you marked unverified.\n"
        f"{JUDGE_FORMAT}\n\n" + "\n\n".join(blocks)
    )
    data, model = ask_json(client, prompt)

    by_id = {}
    for v in data.get("verdicts") or []:
        try:
            by_id[int(v.get("id"))] = v
        except (TypeError, ValueError):
            continue

    judged = []
    for i, c in enumerate(claims, 1):
        v = by_id.get(i, {})
        verdict = str(v.get("verdict", "unverified")).lower()
        if verdict not in VERDICTS:
            verdict = "unverified"

        # Attach real URLs ourselves; ignore any id the model made up.
        sources, seen = [], set()
        for ev in v.get("evidence") or []:
            m = re.fullmatch(r"\s*(\d+)\.(\d+)\s*", str(ev))
            if not m or int(m.group(1)) != i:
                continue
            j = int(m.group(2))
            if 1 <= j <= len(c["results"]):
                r = c["results"][j - 1]
                if r["url"] not in seen:
                    seen.add(r["url"])
                    sources.append({"title": r["title"], "url": r["url"]})

        judged.append({
            "claim": c["claim"],
            "query": c["query"],
            "verdict": verdict,
            "explanation": str(v.get("explanation", "")).strip() or "No explanation returned.",
            "sources": sources,
            "results": c["results"],
        })
    overall = data.get("overall")
    overall = overall.strip() if isinstance(overall, str) else ""
    return judged, model, overall


# --------------------------------------------------------------------------- #
# Overall summary
# --------------------------------------------------------------------------- #
def summarize(judged: list[dict], summary_text: str) -> dict:
    """Combine the verdicts into an overall label (computed in code, not by the model)."""
    total = len(judged)
    counts = Counter(c["verdict"] for c in judged)
    sup, con, mix, unv = (counts.get(v, 0) for v in VERDICTS)

    if con and con / total >= 0.34:
        label, tone = "Several claims disputed", "bad"
    elif con:
        label, tone = "Some claims disputed", "mid"
    elif sup / total >= 0.6:
        label, tone = "Mostly supported", "good"
    elif sup or mix:
        label, tone = "Partly supported", "mid"
    else:
        label, tone = "Mostly unverified", "none"

    if not summary_text:
        summary_text = (
            f"Of {total} claims checked, {sup} were supported, {con} contradicted, {mix} mixed, "
            f"and {unv} could not be verified from the search snippets."
        )
    return {
        "label": label,
        "tone": tone,
        "summary": summary_text,
        "counts": {v: counts.get(v, 0) for v in VERDICTS},
        "total": total,
    }


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def fact_check(client, article: dict) -> dict:
    """Returns {"claims": [...], "model": str | None, "error": str | None}. Never raises."""
    try:
        claims, model = extract_claims(client, article)
        if not claims:
            return {"claims": [], "model": model, "error": "No checkable factual claims were found."}

        domain = urlparse(article.get("url", "")).netloc.lower().removeprefix("www.")
        with ThreadPoolExecutor(max_workers=SEARCH_WORKERS) as pool:
            results = list(pool.map(lambda c: search_one(c["query"], domain), claims))
        for c, r in zip(claims, results):
            c["results"] = r

        judged, model, summary_text = judge(client, article, claims)
        return {
            "claims": judged,
            "overall": summarize(judged, summary_text),
            "model": model,
            "error": None,
        }
    except Exception as e:
        return {"claims": [], "model": None, "error": str(e)}
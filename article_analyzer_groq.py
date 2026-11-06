import copy
import io
import os
import re
import time
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from PIL import Image as PILImage
from groq import Groq

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

NOISE_TAGS = ["script", "style", "noscript", "nav", "footer", "aside", "form",
              "header", "iframe", "svg", "button"]
NOISE_HINTS = re.compile(
    r"(comment|share|social|related|promo|advert|sponsor|newsletter|"
    r"subscribe|sidebar|breadcrumb|cookie|popup|modal|recommend)", re.I)
IMG_SKIP_HINTS = re.compile(
    r"(logo|icon|avatar|sprite|pixel|tracking|badge|banner-ad|spacer|blank|1x1)", re.I)

MIN_PARAGRAPH_CHARS = 40
MIN_IMAGE_DIM = 150

# --- Groq (text + sources via built-in browser search) ---
# Get a key at https://console.groq.com/keys, then set it in your terminal:
#   PowerShell:  $env:GROQ_API_KEY="your-key"
# Never paste a real key into this file or into a chat.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Built-in browser search only works on the gpt-oss models.
GROQ_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"
REASONING_EFFORT = "medium"      # "low" uses fewer tokens; "high" browses much longer
MAX_COMPLETION_TOKENS = 4096     # reasoning tokens count toward this
MAX_CONTEXT_CHARS = 6000         # keep modest: browsing adds a lot of tokens on top
MAX_RETRIES = 6
MAX_WAIT_SECONDS = 90            # never sleep longer than this; switch model or stop instead

# --- Local AI-image detector ---
DETECTOR_MODEL = "umm-maybe/AI-image-detector"   # alternative: "Organika/sdxl-detector"
DETECTOR_BATCH_SIZE = 8
DOWNLOAD_WORKERS = 8

session = requests.Session()
session.headers.update(HEADERS)


# --------------------------------------------------------------------------- #
# Scraper Helper Functions
# --------------------------------------------------------------------------- #
def fetch_html(url: str) -> str:
    resp = session.get(url, timeout=20)
    resp.raise_for_status()
    if resp.encoding is None or resp.encoding.lower() == "iso-8859-1":
        resp.encoding = resp.apparent_encoding
    return resp.text


def _meta(soup, *names):
    for n in names:
        tag = soup.find("meta", attrs={"property": n}) or soup.find("meta", attrs={"name": n})
        if tag and tag.get("content"):
            return tag["content"].strip()
    return None


def extract_metadata(soup: BeautifulSoup) -> dict:
    title = _meta(soup, "og:title", "twitter:title")
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else (
            soup.title.string.strip() if soup.title and soup.title.string else None
        )
    return {
        "title": title,
        "author": _meta(soup, "author", "article:author"),
        "published": _meta(soup, "article:published_time", "date", "pubdate"),
        "description": _meta(soup, "og:description", "description"),
        "og_image": _meta(soup, "og:image", "twitter:image"),
    }


def _text_len(tag) -> int:
    return sum(len(p.get_text(" ", strip=True)) for p in tag.find_all("p"))


def find_article_container(soup: BeautifulSoup):
    candidates = soup.select(
        "article, [itemprop=articleBody], [role=main] article, "
        ".article-body, .post-content, .entry-content, .article-content, "
        ".story-body, #article-body, main"
    )
    candidates = [c for c in candidates if _text_len(c) > 200]
    if candidates:
        return max(candidates, key=_text_len)

    scores = defaultdict(int)
    for p in soup.find_all("p"):
        text = p.get_text(" ", strip=True)
        if len(text) < MIN_PARAGRAPH_CHARS:
            continue
        for parent in (p.parent, p.parent.parent if p.parent else None):
            if parent is not None:
                scores[parent] += len(text)
    if scores:
        return max(scores, key=scores.get)

    return soup.body or soup


def clean_container(container):
    container = copy.copy(container)
    for tag in container.find_all(NOISE_TAGS):
        tag.decompose()
    for tag in container.find_all(True):
        if tag.attrs is None:
            continue
        ident = " ".join(tag.get("class", [])) + " " + (tag.get("id") or "")
        if ident.strip() and NOISE_HINTS.search(ident):
            tag.decompose()
    return container


def extract_paragraphs(container) -> list[str]:
    paragraphs = []
    seen = set()
    for el in container.find_all(["p", "h2", "h3", "h4", "blockquote", "li"]):
        if el.name in ("blockquote", "li") and el.find("p"):
            continue
        text = re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()
        is_heading = el.name in ("h2", "h3", "h4")
        if not text or text in seen:
            continue
        if not is_heading and len(text) < MIN_PARAGRAPH_CHARS:
            continue
        seen.add(text)
        paragraphs.append(text)
    return paragraphs


def _best_from_srcset(srcset: str) -> str | None:
    best, best_w = None, -1
    for part in srcset.split(","):
        bits = part.strip().split()
        if not bits:
            continue
        url = bits[0]
        w = 0
        if len(bits) > 1:
            m = re.match(r"(\d+(?:\.\d+)?)[wx]", bits[1])
            if m:
                w = float(m.group(1))
        if w >= best_w:
            best, best_w = url, w
    return best


def _img_url(img) -> str | None:
    for attr in ("data-srcset", "srcset"):
        if img.get(attr):
            u = _best_from_srcset(img[attr])
            if u:
                return u
    for attr in ("data-src", "data-original", "data-lazy-src", "data-full-url", "src"):
        if img.get(attr):
            return img[attr]
    pic = img.find_parent("picture")
    if pic:
        for source in pic.find_all("source"):
            if source.get("srcset"):
                u = _best_from_srcset(source["srcset"])
                if u:
                    return u
    return None


def _to_int(v):
    try:
        return int(re.sub(r"[^\d]", "", str(v)))
    except (ValueError, TypeError):
        return None


def extract_image_refs(container, base_url: str, og_image: str | None) -> list[dict]:
    refs, seen = [], set()
    for img in container.find_all("img"):
        raw = _img_url(img)
        if not raw or raw.startswith("data:"):
            continue
        url = urljoin(base_url, raw.strip())
        if url in seen or urlparse(url).path.lower().endswith(".svg"):
            continue
        ident = url + " " + " ".join(img.get("class", [])) + " " + (img.get("alt") or "")
        if IMG_SKIP_HINTS.search(ident):
            continue
        w, h = _to_int(img.get("width")), _to_int(img.get("height"))
        if (w and w < MIN_IMAGE_DIM) or (h and h < MIN_IMAGE_DIM):
            continue

        caption = None
        fig = img.find_parent("figure")
        if fig and fig.find("figcaption"):
            caption = fig.find("figcaption").get_text(" ", strip=True)

        seen.add(url)
        refs.append({"url": url, "alt": (img.get("alt") or "").strip(), "caption": caption})

    if og_image:
        url = urljoin(base_url, og_image)
        if url not in seen:
            refs.insert(0, {"url": url, "alt": "", "caption": None, "source": "og:image"})
    return refs


def scrape_article(url: str) -> dict:
    html = fetch_html(url)
    soup = BeautifulSoup(html, "lxml")
    meta = extract_metadata(soup)
    container = clean_container(find_article_container(soup))
    paragraphs = extract_paragraphs(container)
    image_refs = extract_image_refs(container, url, meta["og_image"])

    return {
        "url": url,
        **meta,
        "paragraphs": paragraphs,
        "images": image_refs,
    }


# --------------------------------------------------------------------------- #
# Part 1: Text analysis + sources (Groq gpt-oss with built-in browser search)
# --------------------------------------------------------------------------- #
URL_RE = re.compile(r"https?://[^\s)\]>\"']+")


def extract_sources(resp, text: str) -> list[tuple[str, str]]:
    """Collect source URLs: first from Groq's executed browser-search results (best effort),
    then any URLs the model wrote in its answer."""
    sources, seen = [], set()

    def add(title, url):
        url = url.rstrip(".,;")
        if url and url not in seen:
            seen.add(url)
            sources.append((title or url, url))

    try:
        msg = resp.choices[0].message
        for tool in getattr(msg, "executed_tools", None) or []:
            sr = getattr(tool, "search_results", None)
            for r in (getattr(sr, "results", None) or []):
                add(getattr(r, "title", None), getattr(r, "url", "") or "")
    except Exception:
        pass  # response shape can vary; fall back to URLs in the text

    for url in URL_RE.findall(text):
        add(None, url)
    return sources


def parse_retry_delay(msg: str) -> float | None:
    """Parse Groq's 'Please try again in 6.5s' / '1m12s' / '350ms' hints."""
    m = re.search(r"try again in\s*(?:(\d+)m)?\s*(\d+(?:\.\d+)?)(ms|s)", msg, re.I)
    if not m:
        return None
    minutes = int(m.group(1) or 0)
    value = float(m.group(2))
    seconds = value / 1000 if m.group(3).lower() == "ms" else value
    return minutes * 60 + seconds


def build_prompt(article: dict, ctx_chars: int, max_claims: int) -> str:
    body = "\n\n".join(article["paragraphs"])[:ctx_chars]
    return (
        f"Article title: {article.get('title')}\n\n"
        f"Article text:\n{body}\n\n"
        "Task: fact-check this article.\n"
        f"1. Pick its {max_claims} most important checkable factual claims.\n"
        "2. Use web search sparingly (at most 6 searches total, and do not open more "
        "than 6 pages) to find sources, then say for each claim whether they support or "
        "contradict it, or that no reliable source was found.\n"
        "3. Keep the answer concise. End with a 'Sources' list of the full URLs you "
        "actually visited. Never invent or guess a URL."
    )


# On a context_length_exceeded error, retry with progressively smaller requests.
# (max_claims, article-text divisor) for each shrink step.
SHRINK_STEPS = [(3, 2), (2, 4), (2, 8)]


def analyze_text(client: Groq, article: dict) -> tuple[str, list[tuple[str, str]]]:
    """One Groq request for the whole article. Returns (analysis_text, [(title, url), ...])."""
    model = GROQ_MODEL
    effort = REASONING_EFFORT
    ctx_chars = MAX_CONTEXT_CHARS
    max_claims = 5
    retries = 0
    shrinks = 0
    tool_failures = 0
    temperature = 0.2

    while True:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": build_prompt(article, ctx_chars, max_claims)}],
                tools=[{"type": "browser_search"}],
                tool_choice="required",
                reasoning_effort=effort,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                temperature=temperature,
            )
            text = (resp.choices[0].message.content or "[No text returned]").strip()
            return text, extract_sources(resp, text)
        except Exception as e:
            msg = str(e)

            # gpt-oss sometimes emits a malformed tool call (e.g. a leaked
            # "<|channel|>analysis" token). It is a random generation glitch, so
            # retry with a slightly different temperature, then try the other model.
            if "tool_use_failed" in msg or "tool call validation failed" in msg.lower():
                tool_failures += 1
                if tool_failures > 6:
                    return ("[The model repeatedly produced malformed tool calls. "
                            f"Try again, or switch model in the sidebar: {msg[:200]}]"), []
                temperature = min(0.2 + 0.15 * tool_failures, 0.8)
                if tool_failures >= 3 and model != FALLBACK_MODEL:
                    model = FALLBACK_MODEL
                print(f"  Malformed tool call (attempt {tool_failures}); retrying "
                      f"on {model} at temperature {temperature:.2f}")
                time.sleep(1)
                continue

            # Browsing pulled in too much page text for the model's context window.
            # Retry with fewer claims, less article text, and low reasoning effort.
            if "context_length_exceeded" in msg or "reduce the length" in msg.lower():
                if shrinks >= len(SHRINK_STEPS):
                    return ("[Context window exceeded even with the smallest settings. "
                            f"Try a shorter article: {msg[:200]}]"), []
                max_claims, divisor = SHRINK_STEPS[shrinks]
                ctx_chars = max(1500, MAX_CONTEXT_CHARS // divisor)
                effort = "low"
                shrinks += 1
                print(f"  Context too long; retrying smaller (claims={max_claims}, "
                      f"chars={ctx_chars}, effort=low)")
                continue

            # Errors that retrying will never fix: fail fast with a clear message.
            if any(code in msg for code in ("401", "402", "403")) or "invalid_api_key" in msg:
                return f"[Groq auth/billing problem - check your key and plan: {msg[:300]}]", []
            if "413" in msg or "too large" in msg.lower():
                return ("[Request too large for your Groq rate limits. Lower MAX_CONTEXT_CHARS "
                        f"or REASONING_EFFORT and retry: {msg[:300]}]"), []

            overloaded = "503" in msg or "502" in msg or "overloaded" in msg.lower()
            rate_limited = "429" in msg or "rate_limit" in msg.lower()

            # Daily quota (TPD/RPD) or a very long wait: sleeping won't help.
            # Groq limits are per model, so switch to the fallback model right away.
            if rate_limited:
                low = msg.lower()
                daily = any(k in low for k in ("tokens per day", "(tpd)", "requests per day", "(rpd)"))
                hinted = parse_retry_delay(msg)
                if daily or (hinted is not None and hinted > MAX_WAIT_SECONDS):
                    if model != FALLBACK_MODEL:
                        print(f"  {model} daily/long rate limit hit (resets in ~{(hinted or 0) / 60:.0f} min); "
                              f"switching to {FALLBACK_MODEL}")
                        model = FALLBACK_MODEL
                        continue
                    reset = f"about {(hinted or 0) / 60:.0f} minutes" if hinted else "later"
                    return ("[Groq daily limit reached on both gpt-oss models. It resets in "
                            f"{reset}. Try again then, use fewer/shorter articles, or upgrade your "
                            f"Groq tier: {msg[:200]}]"), []

            if (overloaded or rate_limited or "500" in msg or "504" in msg) and retries < MAX_RETRIES:
                retries += 1
                hinted = parse_retry_delay(msg)
                wait = (hinted + 1) if hinted else min(60, 5 * 2 ** retries)
                wait += random.uniform(0, 2)
                if overloaded and retries >= 2:
                    model = FALLBACK_MODEL  # try the smaller model while this one is busy
                print(f"  Groq retry in {wait:.0f}s ({msg[:300]})")
                time.sleep(wait)
                continue
            return f"[Groq request failed: {msg[:300]}]", []


# --------------------------------------------------------------------------- #
# Part 2: AI-generated image detection (local classifier, no API quota)
# --------------------------------------------------------------------------- #
def load_detector():
    import torch
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("image-classification", model=DETECTOR_MODEL, device=device, use_fast=True)


def download_image(info: dict):
    try:
        r = session.get(info["url"], timeout=15)
        r.raise_for_status()
        img = PILImage.open(io.BytesIO(r.content)).convert("RGB")
        if min(img.size) < MIN_IMAGE_DIM:
            return None
        return info, img
    except Exception as e:
        print(f"Skipping {info['url']}: {e}")
        return None


def detect_images(detector, loaded: list) -> list[dict]:
    """Run the classifier in batches. Returns [{url, scores}, ...] in the same order."""
    results = []
    for i in range(0, len(loaded), DETECTOR_BATCH_SIZE):
        chunk = loaded[i:i + DETECTOR_BATCH_SIZE]
        outputs = detector([img for _, img in chunk], top_k=None)
        for (info, _), preds in zip(chunk, outputs):
            scores = {p["label"]: round(p["score"], 3) for p in preds}
            results.append({"url": info["url"], "scores": scores})
    return results


# --------------------------------------------------------------------------- #
# Main Execution Pipeline
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    TARGET_URL = "https://www.scotthyoung.com/blog/2019/06/07/apprenticeships/"

    if not GROQ_API_KEY:
        raise SystemExit(
            "GROQ_API_KEY is not set.\n"
            'PowerShell:  $env:GROQ_API_KEY="your-key"   then run the script again.'
        )

    print(f"Scraping content from: {TARGET_URL} ...\n")
    article = scrape_article(TARGET_URL)

    print(f"TITLE: {article.get('title')}")
    print(f"PARAGRAPHS EXTRACTED: {len(article['paragraphs'])}")
    print(f"IMAGES FOUND: {len(article['images'])}\n")

    client = Groq(api_key=GROQ_API_KEY)

    # Run everything that can overlap: Groq text call, detector load, image downloads
    print("Running text analysis, loading detector, and downloading images in parallel...\n")
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS + 2) as pool:
        text_future = pool.submit(analyze_text, client, article)
        detector_future = pool.submit(load_detector)
        dl_results = list(pool.map(download_image, article["images"]))
        analysis, sources = text_future.result()
        detector = detector_future.result()

    # ---- Text + sources ----
    print("=" * 60)
    print(f"TEXT ANALYSIS ({GROQ_MODEL} + browser search)")
    print("=" * 60)
    print(analysis, "\n")

    if sources:
        print("SOURCES:")
        for i, (title, url) in enumerate(sources, 1):
            print(f"  [{i}] {title}\n      {url}")
        print()
    else:
        print("SOURCES: none returned\n")

    # ---- Image detection ----
    loaded = [r for r in dl_results if r]
    print("=" * 60)
    print(f"AI-IMAGE DETECTION ({DETECTOR_MODEL})")
    print("=" * 60)
    if not loaded:
        print("No valid images to analyze.")
    else:
        for res in detect_images(detector, loaded):
            top_label = max(res["scores"], key=res["scores"].get)
            print(f"- {res['url']}")
            print(f"    scores: {res['scores']}  ->  top: {top_label}")
        print("\nNote: detector scores are unreliable hints, especially on diagrams and charts.")

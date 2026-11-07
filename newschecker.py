import html
import os
import tomllib
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
from groq import Groq

# Keep these files in the same folder:
#   app.py, article_analyzer_groq.py (scraper + image detector + browse mode), factcheck_lean.py
import article_analyzer_groq as core
import factcheck_lean as lean

st.set_page_config(
    page_title="FactChecked!",
    page_icon="🔎",
    layout="centered",
    initial_sidebar_state="collapsed",
)

VERDICT_ICONS = {"supported": "✅", "contradicted": "❌", "mixed": "⚠️", "unverified": "❔"}

# --------------------------------------------------------------------------- #
# Styling (fonts, colours, cards). Edit the variables at the top to re-theme.
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
  --accent: #8b5cf6;
  --accent-2: #6366f1;
  --bg: #0b0d14;
  --card: rgba(255,255,255,0.04);
  --border: rgba(255,255,255,0.10);
  --muted: #9aa3b5;
  --good: #22c55e; --bad: #ef4444; --mid: #f59e0b; --none: #94a3b8;
}

.stApp {
  background:
    radial-gradient(1100px 520px at 8% -8%, rgba(99,102,241,0.20), transparent 60%),
    radial-gradient(900px 480px at 100% 0%, rgba(236,72,153,0.12), transparent 60%),
    var(--bg);
}
.stApp, .stMarkdown, p, label, li, input, textarea, button, [data-baseweb="select"] {
  font-family: 'Inter', sans-serif !important;
}
h1, h2, h3, h4, .hero h1, .stTabs [data-baseweb="tab"] {
  font-family: 'Space Grotesk', sans-serif !important;
  letter-spacing: -0.02em;
}
/* keep Streamlit's icon font intact */
[data-testid="stIconMaterial"] { font-family: "Material Symbols Rounded" !important; }

/* hide default chrome */
#MainMenu, footer, [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 980px; padding-top: 2.2rem; padding-bottom: 4rem; }

/* hero */
.hero .eyebrow { color: var(--accent); font-weight: 600; font-size: .78rem; letter-spacing: .14em; text-transform: uppercase; }
.hero h1 { font-size: 2.5rem; line-height: 1.1; margin: .25rem 0 .5rem 0; font-weight: 700; }
.hero h1 .grad { background: linear-gradient(90deg, #a78bfa, #f472b6); -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero p { color: var(--muted); margin: 0; max-width: 560px; }

/* settings icon (top right) */
[data-testid="stPopover"] { display: flex; justify-content: flex-end; }
[data-testid="stPopover"] > div > button, [data-testid="stPopover"] button {
  border-radius: 999px; width: 46px; height: 46px; padding: 0;
  border: 1px solid var(--border); background: var(--card); font-size: 1.15rem;
  transition: transform .15s ease, border-color .15s ease;
}
[data-testid="stPopover"] button:hover { transform: rotate(40deg); border-color: var(--accent); }

/* inputs + primary button */
.stTextInput input { border-radius: 12px; padding: .7rem .9rem; border: 1px solid var(--border); background: var(--card); }
.stTextInput input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(139,92,246,.25); }
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, var(--accent-2), var(--accent)); border: 0; color: #fff;
  border-radius: 12px; font-weight: 600; padding: .6rem 1.6rem;
  box-shadow: 0 6px 20px rgba(99,102,241,.35); transition: transform .15s ease, box-shadow .15s ease;
}
.stButton > button[kind="primary"]:hover { transform: translateY(-1px); box-shadow: 0 10px 26px rgba(99,102,241,.45); }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap: .4rem; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { font-weight: 600; padding: .6rem 1rem; }

/* article header */
.article-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; margin: 1.4rem 0 .15rem 0; letter-spacing: -0.02em; }
.article-meta { color: var(--muted); font-size: .9rem; margin-bottom: .6rem; }

/* summary pills */
.pills { display: flex; flex-wrap: wrap; gap: .5rem; margin: .4rem 0 1rem 0; }
.pill { padding: .25rem .7rem; border-radius: 999px; border: 1px solid var(--border); background: var(--card); font-size: .85rem; }

/* claim cards */
.claim { border: 1px solid var(--border); border-left: 4px solid var(--none); background: var(--card);
  border-radius: 14px; padding: 1rem 1.15rem; margin: .8rem 0 .3rem 0; }
.claim.supported { border-left-color: var(--good); }
.claim.contradicted { border-left-color: var(--bad); }
.claim.mixed { border-left-color: var(--mid); }
.claim-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: .45rem; }
.badge { font-size: .78rem; font-weight: 600; padding: .18rem .6rem; border-radius: 999px; background: rgba(148,163,184,.15); color: var(--none); }
.badge.supported { background: rgba(34,197,94,.15); color: var(--good); }
.badge.contradicted { background: rgba(239,68,68,.15); color: var(--bad); }
.badge.mixed { background: rgba(245,158,11,.15); color: var(--mid); }
.num { color: var(--muted); font-size: .78rem; }
.claim-text { font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; font-weight: 600; margin-bottom: .35rem; }
.claim-expl { color: #c9cfdd; font-size: .93rem; line-height: 1.5; }
.chips { margin-top: .65rem; display: flex; flex-wrap: wrap; gap: .4rem; }
.chip { font-size: .78rem; padding: .2rem .6rem; border-radius: 8px; border: 1px solid var(--border);
  background: rgba(255,255,255,.03); color: #c4b5fd !important; text-decoration: none !important; }
.chip:hover { border-color: var(--accent); background: rgba(139,92,246,.12); }
.muted { color: var(--muted); font-size: .8rem; }
/* overall summary card */
.overall { border: 1px solid var(--border); border-radius: 16px; padding: 1.1rem 1.25rem; margin: .6rem 0 1.1rem 0;
  background: linear-gradient(135deg, rgba(99,102,241,.14), rgba(236,72,153,.06)); }
.overall-top { display: flex; justify-content: space-between; align-items: center; gap: .6rem; flex-wrap: wrap; }
.overall-label { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.1rem; }
.overall-badge { font-size: .82rem; font-weight: 600; padding: .22rem .75rem; border-radius: 999px;
  background: rgba(148,163,184,.15); color: var(--none); }
.overall-badge.good { background: rgba(34,197,94,.15); color: var(--good); }
.overall-badge.mid { background: rgba(245,158,11,.15); color: var(--mid); }
.overall-badge.bad { background: rgba(239,68,68,.15); color: var(--bad); }
.overall-sum { color: #d5daea; line-height: 1.6; margin: .65rem 0 .9rem 0; }
.bar { display: flex; height: 8px; border-radius: 999px; overflow: hidden; background: rgba(255,255,255,.08); }
.seg.supported { background: var(--good); } .seg.contradicted { background: var(--bad); }
.seg.mixed { background: var(--mid); } .seg.unverified { background: var(--none); }
.overall .pills { margin: .8rem 0 .2rem 0; }
.overall-note { color: var(--muted); font-size: .78rem; margin-top: .4rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def esc(text) -> str:
    return html.escape(str(text), quote=True)


def safe_url(url: str) -> bool:
    return str(url).lower().startswith(("http://", "https://"))


def short(text: str, n: int = 38) -> str:
    text = str(text)
    return text if len(text) <= n else text[: n - 1] + "…"


def claim_card(i: int, c: dict) -> str:
    v = c["verdict"] if c["verdict"] in VERDICT_ICONS else "unverified"
    chips = "".join(
        f'<a class="chip" href="{esc(s["url"])}" target="_blank" rel="noopener noreferrer">{esc(short(s["title"]))}</a>'
        for s in c["sources"] if safe_url(s["url"])
    ) or '<span class="muted">No source cited</span>'
    return (
        f'<div class="claim {v}">'
        f'<div class="claim-top"><span class="badge {v}">{VERDICT_ICONS[v]} {esc(v.title())}</span>'
        f'<span class="num">Claim {i}</span></div>'
        f'<div class="claim-text">{esc(c["claim"])}</div>'
        f'<div class="claim-expl">{esc(c["explanation"])}</div>'
        f'<div class="chips">{chips}</div></div>'
    )


def overall_card(o: dict) -> str:
    tone = o.get("tone") if o.get("tone") in ("good", "mid", "bad", "none") else "none"
    total = max(int(o.get("total") or 0), 1)
    counts = o.get("counts") or {}
    segs = "".join(
        f'<span class="seg {v}" style="width:{counts[v] / total * 100:.1f}%"></span>'
        for v in VERDICT_ICONS if counts.get(v)
    )
    pills = "".join(
        f'<span class="pill">{VERDICT_ICONS[v]} {counts[v]} {v}</span>'
        for v in VERDICT_ICONS if counts.get(v)
    )
    return (
        f'<div class="overall"><div class="overall-top"><span class="overall-label">Overall</span>'
        f'<span class="overall-badge {tone}">{esc(o.get("label", ""))}</span></div>'
        f'<div class="overall-sum">{esc(o.get("summary", ""))}</div>'
        f'<div class="bar">{segs}</div><div class="pills">{pills}</div>'
        f'<div class="overall-note">Rates only the {total} claims checked, using short search snippets. '
        f'It is not a verdict on the whole article.</div></div>'
    )


APP_DIR = Path(__file__).resolve().parent


def _read_toml_key(path: Path) -> tuple[str | None, str | None]:
    """Read GROQ_API_KEY from a toml file. Tolerates BOM and UTF-16 (common on Windows)."""
    try:
        raw = path.read_bytes()
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            text = raw.decode("utf-16")
        else:
            text = raw.decode("utf-8-sig")
        key = str(tomllib.loads(text).get("GROQ_API_KEY", "")).strip()
        if key:
            return key, None
        return None, f"{path.name} was found but has no GROQ_API_KEY line"
    except Exception as e:
        return None, f"{path} could not be read: {e}"


def find_default_key() -> tuple[str | None, str]:
    """Look for a saved key. Returns (key, where_it_came_from_or_why_not)."""
    try:
        key = st.secrets["GROQ_API_KEY"]
        if key:
            return str(key), "Streamlit secrets"
    except Exception:
        pass
    if os.getenv("GROQ_API_KEY"):
        return os.getenv("GROQ_API_KEY"), "the GROQ_API_KEY environment variable"

    problems = []
    for path in (APP_DIR / ".streamlit" / "secrets.toml", APP_DIR / "secrets.toml"):
        if path.exists():
            key, problem = _read_toml_key(path)
            if key:
                return key, str(path)
            problems.append(problem)
    if problems:
        return None, " | ".join(problems)
    try:
        seen = sorted(f.name for f in APP_DIR.iterdir())
    except Exception:
        seen = []
    nested = APP_DIR / ".streamlit"
    nested_seen = sorted(f.name for f in nested.iterdir()) if nested.is_dir() else "folder does not exist"
    return None, (f"no secrets.toml found. Folder checked: {APP_DIR} | files there: {seen} | "
                  f".streamlit contents: {nested_seen}")


def resolve_api_key(typed_value: str) -> str | None:
    """Priority: key typed in settings -> saved default key."""
    if typed_value:
        return typed_value
    return find_default_key()[0]


@st.cache_resource(show_spinner=False)
def get_detector(model_name: str):
    """Load the image classifier once and reuse it across reruns."""
    core.DETECTOR_MODEL = model_name
    return core.load_detector()


def ai_score(scores: dict) -> float | None:
    """Pull out the 'AI-generated' probability if the label is recognisable."""
    for label, value in scores.items():
        if any(key in label.lower() for key in ("artificial", "ai", "fake", "generated")):
            return value
    return None


# --------------------------------------------------------------------------- #
# Header: title on the left, settings icon (popover) on the right
# --------------------------------------------------------------------------- #
head_left, head_right = st.columns([12, 1])

with head_left:
    st.markdown(
        '<div class="hero"><div class="eyebrow">Reading assistant</div>'
        '<h1>FactChecked!</h1>'
        "<p>Paste an article link. Get its key claims checked against web sources, "
        "plus an AI-generation score for every image.</p></div>",
        unsafe_allow_html=True,
    )

with head_right:
    with st.popover("⚙️"):
        st.markdown("**Settings**")
        if hasattr(core, "MAX_WAIT_SECONDS") and hasattr(lean, "SEARCH_WORKERS"):
            st.caption("✅ Latest module files loaded")
        else:
            st.error("Old module files loaded. Replace the files, then fully restart Streamlit.")

        key_input = st.text_input(
            "Groq API key",
            type="password",
            help="Or set GROQ_API_KEY in the environment / .streamlit/secrets.toml",
        )
        _default_key, _key_info = find_default_key()
        if _default_key:
            st.caption(f"🔑 Default key loaded from {_key_info}. Leave the field blank to use it.")
        else:
            st.caption(f"No default key found: {_key_info}")
        do_text = st.checkbox("Fact-check text with sources", value=True)
        do_images = st.checkbox("Check images for AI generation", value=True)

        method = st.radio(
            "Fact-check method",
            ["Lean: DuckDuckGo + LLM (recommended)", "Groq browser search (token-heavy)"],
        )
        lean_mode = method.startswith("Lean")

        if lean_mode:
            lean_model = st.selectbox(
                "LLM", ["qwen/qwen3.8-27b", "openai/gpt-oss-20b", "openai/gpt-oss-120b"],
                help="Only reads short snippets, so any of these works. Each model has its own daily quota.",
            )
            max_claims = st.slider("Claims to check", 3, 6, 5, help="Capped at 6 to stay under the 8K tokens-per-minute limit")
            results_per_claim = st.slider("Search results per claim", 3, 6, 5, help="Capped at 6 to stay under the 8K tokens-per-minute limit")
        else:
            model = st.selectbox("Model (must support browser search)", ["openai/gpt-oss-120b", "openai/gpt-oss-20b"])
            effort = st.select_slider("Reasoning effort", ["low", "medium", "high"], value="low")
            context_chars = st.slider("Max article characters sent", 2000, 12000, 6000, step=500)

        detector_model = st.selectbox(
            "Image detector model",
            ["umm-maybe/AI-image-detector", "Organika/sdxl-detector"],
        )

# --------------------------------------------------------------------------- #
# Input row
# --------------------------------------------------------------------------- #
st.write("")
url = st.text_input(
    "Article URL",
    value="https://www.scotthyoung.com/blog/2019/06/07/apprenticeships/",
    label_visibility="collapsed",
    placeholder="https://example.com/article",
)
run = st.button("Analyze article", type="primary", disabled=not url)

if run:
    api_key = resolve_api_key(key_input)
    if do_text and not api_key:
        st.error("Click the ⚙️ icon (top right) and add a Groq API key, or set GROQ_API_KEY, to run the fact-check.")
        st.stop()
    if not (do_text or do_images):
        st.warning("Turn on at least one analysis in ⚙️ Settings.")
        st.stop()

    # Push settings into the modules (their functions read these globals at call time)
    if lean_mode:
        lean.LLM_MODEL = lean_model
        lean.FALLBACK_MODEL = "openai/gpt-oss-20b"
        lean.MAX_CLAIMS = max_claims
        lean.RESULTS_PER_CLAIM = results_per_claim
        text_fn, text_model = lean.fact_check, lean_model
    else:
        core.GROQ_MODEL = model
        core.FALLBACK_MODEL = "openai/gpt-oss-20b"
        core.REASONING_EFFORT = effort
        core.MAX_CONTEXT_CHARS = context_chars
        text_fn, text_model = core.analyze_text, model

    with st.status("Working...", expanded=True) as status:
        try:
            st.write("Scraping article...")
            article = core.scrape_article(url)
        except Exception as e:
            status.update(label="Scraping failed", state="error")
            st.error(f"Could not scrape that URL: {e}")
            st.stop()

        st.write(f"Found {len(article['paragraphs'])} paragraphs and {len(article['images'])} images.")
        st.write("Running fact-check, downloading images, and loading detector in parallel...")

        client = Groq(api_key=api_key) if do_text else None
        with ThreadPoolExecutor(max_workers=core.DOWNLOAD_WORKERS + 2) as pool:
            text_future = pool.submit(text_fn, client, article) if do_text else None
            dl_futures = (
                [pool.submit(core.download_image, info) for info in article["images"]]
                if do_images else []
            )
            # Streamlit calls (cache_resource) must stay on the main thread
            detector = get_detector(detector_model) if do_images else None

            text_out = text_future.result() if text_future else None
            loaded = [r for r in (f.result() for f in dl_futures) if r]

        detections = core.detect_images(detector, loaded) if (do_images and loaded) else []
        status.update(label="Done", state="complete", expanded=False)

    res = {
        "article": article,
        "mode": "off",
        "text_model": text_model,
        "loaded": loaded,
        "detections": detections,
        "detector_model": detector_model,
    }
    if text_out is not None:
        if lean_mode:
            res.update(mode="lean", lean=text_out, text_model=text_out.get("model") or text_model)
        else:
            res.update(mode="browse", analysis=text_out[0], sources=text_out[1])
    st.session_state["result"] = res

# --------------------------------------------------------------------------- #
# Results (kept in session_state so they survive widget interactions)
# --------------------------------------------------------------------------- #
res = st.session_state.get("result")
if res:
    article = res["article"]
    st.markdown(f'<div class="article-title">{esc(article.get("title") or "Untitled article")}</div>', unsafe_allow_html=True)
    meta_bits = [b for b in (article.get("author"), article.get("published")) if b]
    if meta_bits:
        st.markdown(f'<div class="article-meta">{esc(" · ".join(meta_bits))}</div>', unsafe_allow_html=True)

    tab_text, tab_images = st.tabs(["📝 Fact-check", "🖼️ Images"])

    with tab_text:
        if res["mode"] == "off":
            st.info("Text analysis was turned off.")

        elif res["mode"] == "lean":
            fc = res["lean"]
            if fc.get("error"):
                st.error(fc["error"])
            else:
                if fc.get("overall"):
                    st.markdown(overall_card(fc["overall"]), unsafe_allow_html=True)
                st.caption(f"Model: {res['text_model']} · evidence from DuckDuckGo search snippets")

                for i, c in enumerate(fc["claims"], 1):
                    st.markdown(claim_card(i, c), unsafe_allow_html=True)
                    with st.expander(f"Search used: “{c['query']}” ({len(c['results'])} results)"):
                        for r in c["results"]:
                            st.markdown(f"- [{r['title']}]({r['url']})  \n  {r['snippet']}")

                st.caption("Verdicts are based only on short search snippets, so treat them as a "
                           "starting point and open the sources to confirm.")

        else:  # browse mode
            st.caption(f"Model: {res['text_model']} with built-in browser search")
            st.markdown(res["analysis"])
            st.markdown("#### Sources")
            if res["sources"]:
                for i, (title, link) in enumerate(res["sources"], 1):
                    st.markdown(f"{i}. [{title}]({link})")
            else:
                st.write("No sources returned.")

    with tab_images:
        if not res["detections"]:
            st.info("No image results (detection was off, or no valid images were found).")
        else:
            st.caption(f"Detector: {res['detector_model']}")
            st.warning(
                "Detector scores are hints, not proof, and are especially unreliable on diagrams and charts.",
                icon="⚠️",
            )
            cols = st.columns(3)
            for idx, ((info, img), det) in enumerate(zip(res["loaded"], res["detections"])):
                with cols[idx % 3]:
                    with st.container(border=True):
                        st.image(img, use_container_width=True)
                        score = ai_score(det["scores"])
                        if score is not None:
                            st.progress(min(max(score, 0.0), 1.0), text=f"AI-generated score: {score:.1%}")
                        st.json(det["scores"], expanded=False)
                        st.caption(info["url"])

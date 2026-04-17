"""
Qafiya Annotator — compact two-column annotation UI

Layout: poem (left) | classification + verdict (right)
"""

import os
import uuid

import streamlit as st
from sqlalchemy import create_engine, text

# ── Config ────────────────────────────────────────────────────────────────────

DB = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://alshaer:alshaer@localhost:5433/alshaer",
).replace("+asyncpg", "+psycopg")

st.set_page_config(
    page_title="Qafiya Annotator",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* Global */
.block-container { padding: 1rem 1.5rem 0.5rem; }
section[data-testid="stSidebar"] { width: 220px !important; }

/* Poem display */
.poem-header { font-size: 1rem; font-weight: 700; direction: rtl; text-align: right;
               margin-bottom: 2px; color: #1a1a2e; }
.poem-meta   { font-size: 0.72rem; color: #888; direction: rtl; text-align: right;
               margin-bottom: 8px; }
.verse-list  { direction: rtl; text-align: right; font-size: 1.0rem;
               line-height: 1.9; font-family: 'Amiri', 'Arial Unicode MS', serif;
               background: #fdf8f0; border-radius: 6px; padding: 10px 14px;
               border-right: 3px solid #b8860b; max-height: 72vh;
               overflow-y: auto; }

/* Classification panel */
.cls-block   { background: #f5f5f5; border-radius: 8px; padding: 12px 14px;
               margin-bottom: 10px; }
.cls-row     { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 6px; }
.cls-field   { background: white; border: 1px solid #ddd; border-radius: 5px;
               padding: 4px 10px; font-size: 0.82rem; }
.cls-label   { font-size: 0.7rem; color: #888; display: block; }
.cls-value   { font-size: 1.05rem; font-weight: 700; color: #1a1a2e; }
.cls-value.none { color: #bbb; }

/* Confidence badges */
.conf-high   { background:#2d6a4f; color:white; padding:2px 8px; border-radius:10px;
               font-size:0.75rem; font-weight:600; }
.conf-medium { background:#e9c46a; color:#333; padding:2px 8px; border-radius:10px;
               font-size:0.75rem; font-weight:600; }
.conf-low    { background:#e76f51; color:white; padding:2px 8px; border-radius:10px;
               font-size:0.75rem; font-weight:600; }
.conf-none   { background:#ccc; color:#555; padding:2px 8px; border-radius:10px;
               font-size:0.75rem; font-weight:600; }

/* Verdict buttons */
div[data-testid="stHorizontalBlock"] button { font-size: 0.85rem !important; }

/* Progress bar area */
.prog-bar { font-size: 0.75rem; color: #888; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)


# ── DB ────────────────────────────────────────────────────────────────────────

@st.cache_resource
def engine():
    return create_engine(DB, pool_pre_ping=True)

eng = engine()


def fetch_poem(mode: str, conf_filter: str) -> dict | None:
    where = ["p.verse_count >= 6"]
    if mode == "classified":
        where.append("p.qafiya_confidence NOT IN ('none') AND p.qafiya_confidence IS NOT NULL")
        if conf_filter != "all":
            where.append(f"p.qafiya_confidence = '{conf_filter}'")
    elif mode == "unclassified":
        where.append("(p.qafiya_confidence IS NULL OR p.qafiya_confidence = 'none')")
    where.append("NOT EXISTS (SELECT 1 FROM qafiya_annotations qa WHERE qa.poem_id = p.id)")
    sql = f"""
        SELECT p.id, p.title, p.poet_name, p.meter, p.rhyme_letter,
               p.verse_count, p.era, p.qafiya_rawiy, p.qafiya_radf,
               p.qafiya_wasl, p.qafiya_harakah, p.qafiya_type,
               p.qafiya_pattern, p.qafiya_confidence,
               array_agg(v.text ORDER BY v.position) as verses
        FROM poems p JOIN verses v ON v.poem_id = p.id
        WHERE {' AND '.join(where)}
        GROUP BY p.id ORDER BY random() LIMIT 1
    """
    with eng.connect() as conn:
        row = conn.execute(text(sql)).fetchone()
    return dict(row._mapping) if row else None


def save_annotation(poem_id, verdict, notes, guideline,
                    cor_rawiy, cor_radf, cor_harakah, cor_type):
    with eng.begin() as conn:
        conn.execute(text("""
            INSERT INTO qafiya_annotations
              (id, poem_id, verdict, correct_rawiy, correct_radf,
               correct_harakah, correct_type, notes)
            VALUES (:id,:pid,:v,:r,:rd,:h,:t,:n)
        """), {"id": str(uuid.uuid4()), "pid": str(poem_id),
               "v": verdict, "r": cor_rawiy or None, "rd": cor_radf or None,
               "h": cor_harakah or None, "t": cor_type or None, "n": notes or None})
        if guideline and guideline.strip():
            conn.execute(text("""
                INSERT INTO classifier_guidelines (id, guideline, example_poem_id)
                VALUES (:id,:g,:pid)
            """), {"id": str(uuid.uuid4()), "g": guideline.strip(), "pid": str(poem_id)})


@st.cache_data(ttl=30)
def get_stats() -> dict:
    with eng.connect() as conn:
        r = conn.execute(text("""
            SELECT
              count(*) FILTER (WHERE verse_count >= 6) as total,
              count(*) FILTER (WHERE qafiya_confidence='high') as high,
              count(*) FILTER (WHERE qafiya_confidence='medium') as medium,
              count(*) FILTER (WHERE qafiya_confidence='low') as low,
              count(*) FILTER (WHERE qafiya_confidence='none'
                               OR qafiya_confidence IS NULL) as unclassified
            FROM poems
        """)).fetchone()
        done = conn.execute(text("SELECT count(*) FROM qafiya_annotations")).scalar()
        pending = conn.execute(
            text("SELECT count(*) FROM classifier_guidelines WHERE NOT applied")
        ).scalar()
    return {**dict(r._mapping), "done": done, "pending_guidelines": pending}


def field_html(label: str, value: str | None, wide: bool = False) -> str:
    v = value or "—"
    cls = "none" if not value else ""
    w = "min-width:90px" if wide else ""
    return (f'<div class="cls-field" style="{w}">'
            f'<span class="cls-label">{label}</span>'
            f'<span class="cls-value {cls}">{v}</span></div>')


def conf_badge(conf: str | None) -> str:
    c = conf or "none"
    labels = {"high": "● high", "medium": "◑ medium", "low": "○ low", "none": "✕ none"}
    return f'<span class="conf-{c}">{labels.get(c, c)}</span>'


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Filters")
    mode = st.selectbox("Sample", ["classified", "unclassified", "both"],
                        label_visibility="collapsed")
    conf_filter = st.selectbox(
        "Confidence", ["all", "high", "medium", "low"],
        disabled=(mode != "classified"), label_visibility="collapsed"
    )
    st.divider()
    st.markdown("### 📋 Guidelines")
    with eng.connect() as _c:
        gs = _c.execute(text(
            "SELECT guideline, created_at FROM classifier_guidelines "
            "WHERE NOT applied ORDER BY created_at DESC LIMIT 20"
        )).fetchall()
    if gs:
        for g in gs:
            st.caption(f"• {g[0][:80]}")
        if st.button("Mark all applied"):
            with eng.begin() as _c:
                _c.execute(text("UPDATE classifier_guidelines SET applied = true"))
            st.rerun()
    else:
        st.caption("No pending guidelines.")


# ── Top bar ───────────────────────────────────────────────────────────────────

stats = get_stats()
t1, t2, t3, t4, t5, _, btn_col = st.columns([2, 2, 2, 2, 2, 3, 2])
t1.metric("Annotated", f"{stats['done']:,}")
t2.metric("High conf", f"{stats['high']:,}")
t3.metric("Medium", f"{stats['medium']:,}")
t4.metric("Low", f"{stats['low']:,}")
t5.metric("Unclassified", f"{stats['unclassified']:,}")

if btn_col.button("🔀  New sample", use_container_width=True, type="secondary"):
    st.session_state.pop("poem", None)
    st.session_state.pop("verdict", None)
    st.session_state.pop("guideline", None)

st.divider()

# ── Load poem ─────────────────────────────────────────────────────────────────

if "poem" not in st.session_state:
    p = fetch_poem(mode, conf_filter if mode == "classified" else "all")
    if not p:
        st.warning("No poems available with current filters.")
        st.stop()
    st.session_state.poem = p

p = st.session_state.poem

# ── Two-column layout ─────────────────────────────────────────────────────────

left, right = st.columns([5, 4], gap="large")

# ── LEFT: Poem ────────────────────────────────────────────────────────────────

with left:
    title   = p["title"] or "بدون عنوان"
    poet    = p["poet_name"] or "شاعر مجهول"
    era     = p["era"] or ""
    meter   = p["meter"] or "?"
    nv      = p["verse_count"]

    st.markdown(f'<div class="poem-header">{title}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="poem-meta">{poet}  ·  {era}  ·  {meter}  ·  {nv} شطر</div>',
        unsafe_allow_html=True,
    )

    verses = p.get("verses") or []
    lines  = "".join(f"<div>{v}</div>" for v in verses[:40])
    more   = f"<div style='color:#aaa;font-size:0.8rem'>... +{len(verses)-40} شطر</div>" \
             if len(verses) > 40 else ""
    st.markdown(f'<div class="verse-list">{lines}{more}</div>', unsafe_allow_html=True)

# ── RIGHT: Classification + Verdict ───────────────────────────────────────────

with right:

    # ─ Classifier output ─
    conf = p.get("qafiya_confidence") or "none"
    st.markdown(
        f'<div class="cls-block">'
        f'<b style="font-size:0.85rem">Classifier Output</b> &nbsp; {conf_badge(conf)}'
        f'<div class="cls-row">'
        f'{field_html("Pattern", p.get("qafiya_pattern"), wide=True)}'
        f'{field_html("Rawiy", p.get("qafiya_rawiy"))}'
        f'{field_html("Radf", p.get("qafiya_radf"))}'
        f'{field_html("Wasl", p.get("qafiya_wasl"))}'
        f'{field_html("Harakah", p.get("qafiya_harakah"), wide=True)}'
        f'{field_html("Type", p.get("qafiya_type"), wide=True)}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if p.get("rhyme_letter"):
        st.caption(f"Ashaar rhyme_letter: **{p['rhyme_letter']}** (last-char convention)")

    st.markdown("**Verdict**")
    v1, v2, v3, v4 = st.columns(4)
    verdict = st.session_state.get("verdict", None)

    def set_verdict(v):
        st.session_state.verdict = v

    if v1.button("✓ Accept",  use_container_width=True,
                 type="primary" if verdict == "accept"  else "secondary"):
        set_verdict("accept")
    if v2.button("~ Partial", use_container_width=True,
                 type="primary" if verdict == "partial" else "secondary"):
        set_verdict("partial")
    if v3.button("✗ Reject",  use_container_width=True,
                 type="primary" if verdict == "reject"  else "secondary"):
        set_verdict("reject")
    if v4.button("⊘ Skip",    use_container_width=True,
                 type="primary" if verdict == "skip"    else "secondary"):
        set_verdict("skip")

    # ─ Corrections (shown for partial/reject) ─
    show_corrections = verdict in ("partial", "reject")
    if show_corrections:
        st.markdown("**Corrections**")
        cc1, cc2 = st.columns(2)
        cor_rawiy   = cc1.text_input("Rawiy",   placeholder="e.g. ن",  key="cor_rawiy",
                                     label_visibility="visible")
        cor_radf    = cc2.text_input("Radf",    placeholder="e.g. و",  key="cor_radf",
                                     label_visibility="visible")
        cc3, cc4 = st.columns(2)
        cor_harakah = cc3.selectbox("Harakah",
                                    ["", "maftouha", "maksura", "madmouma", "muqayyada"],
                                    key="cor_harakah")
        cor_type    = cc4.selectbox("Type",
                                    ["", "mutawatir", "mutadaarik", "mutarakib",
                                     "mutakaasis", "mutaradif", "indeterminate"],
                                    key="cor_type")
    else:
        cor_rawiy = cor_radf = cor_harakah = cor_type = None

    # ─ Notes + Guideline ─
    notes     = st.text_area("Notes", placeholder="Any observations…",
                             height=68, label_visibility="visible")
    guideline = st.text_area(
        "📌 Add classifier guideline",
        placeholder="e.g. When pattern ends in ـنا, rawiy=ن and ا is wasl (maftouha)",
        height=68, label_visibility="visible",
    )

    # ─ Save button ─
    st.markdown("")
    if st.button("💾  Save & Next", type="primary", use_container_width=True,
                 disabled=(verdict is None)):
        save_annotation(
            p["id"], verdict, notes, guideline,
            cor_rawiy, cor_radf, cor_harakah, cor_type,
        )
        # Clear and reload
        for k in ["poem", "verdict", "guideline", "cor_rawiy",
                  "cor_radf", "cor_harakah", "cor_type"]:
            st.session_state.pop(k, None)
        st.cache_data.clear()
        st.rerun()
    elif verdict is None:
        st.caption("Select a verdict to enable Save.")

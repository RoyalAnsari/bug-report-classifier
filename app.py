"""app.py - Streamlit user interface.

This file only SHOWS things. All the real work is done by the tested modules in src/.
Run it with:   python -m streamlit run app.py
"""

import html
import json
import math
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.pipeline import DEFAULT_MODEL_PATH, PipelineError, add_titles, run_pipeline
from src.reporting import (
    class_distribution,
    decision_distribution,
    duplicates_to_csv,
    filter_pairs,
    filter_reports,
    overview_metrics,
    reports_to_csv,
    search_reports,
    similarity_bands,
    sort_pairs,
    sort_reports,
)
from src.review import (
    ReviewError,
    decisions_to_store,
    load_review,
    merge_previous_decisions,
    save_review,
    set_decision,
)

DEMO_PATH = "data/demo_bug_reports.csv"
MODEL_INFO_PATH = "models/test_scores.json"
DECISIONS_PATH = "data/review_decisions.csv"
PAGE_SIZE = 8
BLUE, TEAL, AMBER, GREEN, RED = "#2563eb", "#14b8a6", "#f59e0b", "#22c55e", "#ef4444"
SEVERITY_COLORS = {"blocker": "#991b1b", "critical": RED, "major": "#f97316", "normal": "#64748b",
                   "minor": AMBER, "trivial": TEAL}

# Colours use transparency, so the design works in both light and dark themes.
CSS = """
<style>
@keyframes fadeUp { from {opacity: 0; transform: translateY(14px);} to {opacity: 1; transform: none;} }
@keyframes grow { from {width: 0;} to {width: var(--w);} }
.hero { animation: fadeUp .6s ease both; padding: 1.3rem 1.5rem; border-radius: 16px; margin-bottom: 1.1rem;
        background: linear-gradient(120deg, #2563eb 0%, #14b8a6 100%); color: #fff; }
.hero h1 { margin: 0; font-size: 1.9rem; color: #fff; }
.hero p { margin: .35rem 0 0; opacity: .93; }
.card { border: 1px solid rgba(127,127,127,.28); background: rgba(127,127,127,.07); border-radius: 14px;
        padding: .95rem 1.15rem; margin-bottom: .6rem; animation: fadeUp .55s ease both;
        transition: transform .2s ease, box-shadow .2s ease; }
.card:hover { transform: translateY(-3px); box-shadow: 0 10px 24px rgba(0,0,0,.15); }
.kpi-value { font-size: 2rem; font-weight: 700; line-height: 1.1; }
.kpi-label { font-size: .78rem; opacity: .7; text-transform: uppercase; letter-spacing: .05em; }
.badge { display: inline-block; padding: .1rem .65rem; border-radius: 999px; font-size: .75rem;
         font-weight: 600; margin-right: .4rem; }
.b-machine { background: rgba(37,99,235,.16); color: #3b82f6; }
.b-pending { background: rgba(245,158,11,.18); color: #d97706; }
.b-confirmed { background: rgba(34,197,94,.18); color: #16a34a; }
.b-rejected { background: rgba(239,68,68,.18); color: #dc2626; }
.simbar { height: 8px; border-radius: 99px; background: rgba(127,127,127,.22); overflow: hidden; margin: .55rem 0 .2rem; }
.simbar > span { display: block; height: 100%; width: var(--w); border-radius: 99px;
                 background: linear-gradient(90deg, #2563eb, #14b8a6); animation: grow .9s ease both; }
.pair { border-left: 5px solid #f59e0b; }
.pair.d-confirmed { border-left-color: #22c55e; }
.pair.d-rejected { border-left-color: #ef4444; }
.pair-head { display: flex; flex-wrap: wrap; align-items: center; gap: .3rem; }
.pill { margin-left: auto; padding: .1rem .7rem; border-radius: 999px; font-weight: 700; font-size: .85rem; }
.s-high { background: rgba(34,197,94,.18); color: #16a34a; }
.s-mid { background: rgba(37,99,235,.16); color: #3b82f6; }
.s-low { background: rgba(245,158,11,.18); color: #d97706; }
.pair-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: .4rem; }
@media (max-width: 700px) { .pair-grid { grid-template-columns: 1fr; } }
.side-label { font-size: .7rem; text-transform: uppercase; letter-spacing: .06em; opacity: .55; }
.note { border-left: 4px solid #2563eb; background: rgba(37,99,235,.08); padding: .6rem .9rem;
        border-radius: 8px; font-size: .9rem; margin-bottom: .8rem; }
#MainMenu, footer { visibility: hidden; }
.pair-title { font-weight: 600; margin: .15rem 0; }
.pair-id { opacity: .6; font-size: .8rem; }
.step-num { display: inline-block; width: 1.9rem; height: 1.9rem; line-height: 1.9rem; text-align: center;
            border-radius: 50%; background: linear-gradient(120deg, #2563eb, #14b8a6); color: #fff; font-weight: 700; }
</style>
"""


# ---------- small HTML helpers (all text from the CSV is escaped: it is untrusted) ----------
def esc(value):
    return html.escape(str(value))


def kpi_card(label, value, delay):
    return (f'<div class="card" style="animation-delay:{delay}s"><div class="kpi-value">{esc(value)}</div>'
            f'<div class="kpi-label">{esc(label)}</div></div>')


def pair_card(row, delay=0.0):
    decision = row["human_decision"]
    similarity = float(row["similarity"])
    percent = max(0, min(100, round(similarity * 100)))
    strength = "s-high" if similarity >= 0.9 else "s-mid" if similarity >= 0.8 else "s-low"
    return (
        f'<div class="card pair d-{esc(decision)}" style="animation-delay:{delay}s">'
        f'<div class="pair-head"><span class="badge b-machine">Machine: {esc(row["status"])}</span>'
        f'<span class="badge b-{esc(decision)}">Human: {esc(decision)}</span>'
        f'<span class="pill {strength}">{percent}% similar</span></div>'
        f'<div class="simbar" style="--w:{percent}%"><span></span></div>'
        f'<div class="pair-grid">'
        f'<div><div class="side-label">Report A</div><div class="pair-title">{esc(row.get("query_title", ""))}</div>'
        f'<div class="pair-id">{esc(row["query_id"])}</div></div>'
        f'<div><div class="side-label">Report B</div><div class="pair-title">{esc(row.get("match_title", ""))}</div>'
        f'<div class="pair-id">{esc(row["match_id"])}</div></div>'
        f'</div></div>'
    )


def load_model_info():
    """Read the saved training summary (or None if it is missing)."""
    try:
        with open(MODEL_INFO_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return None


# ---------- charts (Plotly draws; reporting.py supplies the numbers) ----------
def style_chart(fig, title):
    fig.update_layout(
        title=title, height=330, margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        transition={"duration": 600, "easing": "cubic-in-out"}, showlegend=False,
    )
    return fig


def bar_chart(table, x, y, title, color):
    fig = go.Figure(go.Bar(x=table[x], y=table[y], text=table[y], textposition="outside",
                           marker_color=color, hovertemplate="%{x}: %{y}<extra></extra>"))
    return style_chart(fig, title)


def donut_chart(table, title):
    colors = {"pending": AMBER, "confirmed": GREEN, "rejected": RED}
    fig = go.Figure(go.Pie(labels=table["decision"], values=table["count"], hole=0.55,
                           marker=dict(colors=[colors[d] for d in table["decision"]]),
                           hovertemplate="%{label}: %{value}<extra></extra>"))
    style_chart(fig, title)
    fig.update_layout(showlegend=True)
    return fig


# ---------- state and actions ----------
def merge_saved_decisions(pairs):
    """Bring back decisions saved earlier (for example from a previous session)."""
    if not os.path.exists(DECISIONS_PATH):
        return pairs
    try:
        return merge_previous_decisions(pairs, load_review(DECISIONS_PATH))
    except ReviewError:
        return pairs


def persist_decisions():
    """Save human decisions to disk so they survive a restart."""
    saved = None
    if os.path.exists(DECISIONS_PATH):
        try:
            saved = load_review(DECISIONS_PATH)
        except ReviewError:
            saved = None
    save_review(decisions_to_store(st.session_state["pairs"], saved), DECISIONS_PATH)


def record_decision(query_id, match_id, decision):
    """Called when a person presses Confirm, Reject or Undo."""
    st.session_state["pairs"] = set_decision(st.session_state["pairs"], query_id, match_id, decision)
    persist_decisions()
    messages = {"confirmed": "Marked as duplicate", "rejected": "Marked as NOT a duplicate", "pending": "Decision undone"}
    st.toast(messages[decision])


def analyse(source, name, threshold, top_k):
    """Run the backend workflow and keep the result in session_state."""
    with st.status("Analysing your bug reports...", expanded=True) as status:
        st.write("Loading, cleaning, classifying and searching for duplicates")
        try:
            result = run_pipeline(source, DEFAULT_MODEL_PATH, threshold=threshold, top_k=top_k)
        except PipelineError as error:
            status.update(label="Could not analyse the file", state="error")
            st.error(str(error))
            return
        pairs = merge_saved_decisions(add_titles(result["duplicates"], result["reports"]))
        st.session_state["result"] = result
        st.session_state["pairs"] = pairs
        st.session_state["file_name"] = name
        status.update(label="Analysis complete", state="complete", expanded=False)
    st.toast("Analysis complete")


# ---------- page parts ----------
def show_welcome(model_ready):
    """Draw the first page. Returns True if the person pressed the demo button."""
    st.markdown(
        '<div class="hero"><h1>Bug Report Classifier and Duplicate Detector</h1>'
        '<p>Upload bug reports, see their predicted severity, and review possible duplicates. '
        'The machine suggests. You decide.</p></div>', unsafe_allow_html=True)
    if not model_ready:
        st.warning("No trained model found. Run this in the terminal first:  python -m src.train_model")
    steps = [("1", "Upload", "Choose a CSV file in the sidebar."),
             ("2", "Analyse", "Set the similarity threshold and press Analyse reports."),
             ("3", "Review", "Confirm or reject each suggested duplicate, then export.")]
    for column, (number, title, text), delay in zip(st.columns(3), steps, (0.1, 0.25, 0.4)):
        with column:
            st.markdown(f'<div class="card" style="animation-delay:{delay}s"><span class="step-num">{number}</span> '
                        f'<b>{title}</b><div class="pair-id" style="margin-top:.4rem">{text}</div></div>',
                        unsafe_allow_html=True)
    st.write("Just want to look around? Try the small demo file (20 made-up reports).")
    return st.button("Try the demo file", type="primary", disabled=not model_ready)


def show_model_note():
    info = load_model_info()
    if info is None:
        return
    name = str(info.get("best_model", "model")).replace("_", " ")
    score = info.get("f1_macro")
    score_text = f" (test macro F1 {float(score):.2f})" if score is not None else ""
    st.markdown(
        f'<div class="note">Severity model: <b>{esc(name)}</b>{esc(score_text)}. '
        'Severity is hard to predict from a short title, so treat predictions as hints, not facts.</div>',
        unsafe_allow_html=True)


def show_overview(reports, pairs):
    metrics = overview_metrics(reports, pairs)
    cards = [("Reports", metrics["reports"]), ("Suggested pairs", metrics["pairs"]),
             ("Confirmed", metrics["confirmed"]), ("Rejected", metrics["rejected"]),
             ("Pending", metrics["pending"]), ("Top class", metrics["top_class"])]
    for column, (label, value), delay in zip(st.columns(len(cards)), cards, [0.05 * i for i in range(len(cards))]):
        with column:
            st.markdown(kpi_card(label, value, delay), unsafe_allow_html=True)

    left, middle, right = st.columns(3)
    with left:
        classes = class_distribution(reports)
        st.plotly_chart(bar_chart(classes, "class", "count", "Predicted classes",
                                  [SEVERITY_COLORS.get(c, BLUE) for c in classes["class"]]))
    with middle:
        st.plotly_chart(donut_chart(decision_distribution(pairs), "Human decisions"))
    with right:
        st.plotly_chart(bar_chart(similarity_bands(pairs), "band", "count", "Pairs by similarity", TEAL))
    if reports["confidence"].isna().all():
        st.caption("Confidence scores are not available for the Linear SVM model.")


def show_reports(reports):
    left, right = st.columns([2, 2])
    query = left.text_input("Search title or description", "")
    classes = sorted(reports["predicted_component"].unique())
    chosen = right.multiselect("Predicted class", classes, default=classes)
    components = None
    if "component" in reports.columns:
        options = sorted(reports["component"].unique())
        components = st.multiselect("True component (from your file)", options, default=options)
    sort_left, sort_right = st.columns(2)
    sort_by = sort_left.selectbox("Sort by", ["predicted_component", "title", "bug_id", "confidence"])
    order = sort_right.radio("Order", ["Ascending", "Descending"], horizontal=True)

    shown = filter_reports(search_reports(reports, query), chosen, components)
    shown = sort_reports(shown, sort_by, ascending=(order == "Ascending"))
    columns = [c for c in ["bug_id", "title", "component", "predicted_component", "confidence"] if c in shown.columns]
    table = shown[columns].copy()
    table["confidence"] = table["confidence"].map(lambda v: "n/a" if pd.isna(v) else f"{float(v):.0%}")
    st.caption(f"Showing {len(table)} of {len(reports)} reports")
    st.dataframe(table)


def show_review(pairs):
    if len(pairs) == 0:
        st.info("No possible duplicates were found at this threshold. Try a lower threshold and analyse again.")
        return
    st.caption("These are only suggestions. Press Confirm or Reject to record your own decision.")
    decided = int((pairs["human_decision"] != "pending").sum())
    st.progress(decided / len(pairs), text=f"{decided} of {len(pairs)} pairs reviewed")
    left, middle, right = st.columns([1, 2, 2])
    min_similarity = left.slider("Minimum similarity", 0.0, 1.0, 0.0, 0.05)
    decisions = middle.multiselect("Show", ["pending", "confirmed", "rejected"], default=["pending", "confirmed", "rejected"])
    query = right.text_input("Search pairs", "")

    shown = sort_pairs(filter_pairs(pairs, min_similarity, decisions, query))
    pages = max(1, math.ceil(len(shown) / PAGE_SIZE))
    page = st.number_input("Page", 1, pages, 1) if pages > 1 else 1
    st.caption(f"{len(shown)} pair(s) match your filters")

    start = (int(page) - 1) * PAGE_SIZE
    for offset, (_, row) in enumerate(shown.iloc[start:start + PAGE_SIZE].iterrows()):
        card_column, button_column = st.columns([5, 1])
        with card_column:
            st.markdown(pair_card(row, delay=0.05 * offset), unsafe_allow_html=True)
        with button_column:
            key = f'{row["query_id"]}|{row["match_id"]}'
            st.button("✅ Confirm", type="primary", key="c|" + key, on_click=record_decision, args=(row["query_id"], row["match_id"], "confirmed"))
            st.button("❌ Reject", key="r|" + key, on_click=record_decision, args=(row["query_id"], row["match_id"], "rejected"))
            if row["human_decision"] != "pending":
                st.button("↩️ Undo", key="u|" + key, on_click=record_decision, args=(row["query_id"], row["match_id"], "pending"))


def show_export(reports, pairs):
    st.write("Download your results as CSV files.")
    left, right = st.columns(2)
    with left:
        st.markdown(kpi_card("Classified reports", len(reports), 0.05), unsafe_allow_html=True)
        st.download_button("Download classification results", reports_to_csv(reports),
                           file_name="classification_results.csv", mime="text/csv")
    with right:
        st.markdown(kpi_card("Duplicate suggestions", len(pairs), 0.15), unsafe_allow_html=True)
        st.download_button("Download duplicate review", duplicates_to_csv(pairs),
                           file_name="duplicate_review.csv", mime="text/csv")
    st.caption("The duplicate file keeps the machine columns (status, similarity) separate from the human columns (human_decision, reviewed_at).")


def main():
    st.set_page_config(page_title="Bug Report Classifier", page_icon="🐞", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    st.sidebar.markdown("## 🐞 Bug Report Tool")
    uploaded = st.sidebar.file_uploader("Bug report CSV file", type=["csv"])
    use_demo = st.sidebar.checkbox("Use the demo file instead")
    threshold = st.sidebar.slider("Similarity threshold", 0.5, 1.0, 0.8, 0.05,
                                  help="Higher = fewer, safer suggestions. Lower = more suggestions.")
    top_k = st.sidebar.slider("Matches checked per report", 1, 10, 5)
    with st.sidebar.expander("Required file format"):
        st.write("Columns: bug_id, title, description. Optional: component.")
    analyse_clicked = st.sidebar.button("Analyse reports", type="primary")
    if "result" in st.session_state and st.sidebar.button("Clear results"):
        for key in ("result", "pairs", "file_name"):
            st.session_state.pop(key, None)

    model_ready = os.path.exists(DEFAULT_MODEL_PATH)
    if analyse_clicked:
        if use_demo:
            analyse(DEMO_PATH, "demo_bug_reports.csv", threshold, top_k)
        elif uploaded is not None:
            uploaded.seek(0)
            analyse(uploaded, uploaded.name, threshold, top_k)
        else:
            st.sidebar.warning("Please upload a CSV file or tick the demo file.")

    if "result" not in st.session_state:
        if show_welcome(model_ready):
            analyse(DEMO_PATH, "demo_bug_reports.csv", threshold, top_k)
            st.rerun()
        return

    reports, pairs = st.session_state["result"]["reports"], st.session_state["pairs"]
    st.markdown(f'<div class="hero"><h1>Results</h1><p>{esc(st.session_state["file_name"])}</p></div>',
                unsafe_allow_html=True)
    show_model_note()
    for message in st.session_state["result"]["warnings"]:
        st.info(message)

    overview_tab, reports_tab, review_tab, export_tab = st.tabs(["Overview", "Reports", "Duplicate review", "Export"])
    with overview_tab:
        show_overview(reports, pairs)
    with reports_tab:
        show_reports(reports)
    with review_tab:
        show_review(pairs)
    with export_tab:
        show_export(reports, pairs)


main()

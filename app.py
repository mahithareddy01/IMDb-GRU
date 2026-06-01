import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
import plotly.graph_objects as go
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="CINTEL-7 · Sentiment Terminal",
    page_icon="📟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────
# CSS — RETRO CRT TERMINAL
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=VT323&family=Orbitron:wght@400;700;900&display=swap');

:root {
    --phosphor:   #00ff88;
    --phosphor2:  #00cc66;
    --phosphor-dim: #004422;
    --amber:      #ffb000;
    --red-alert:  #ff3333;
    --bg:         #020c06;
    --bg2:        #030f07;
    --panel:      #040d07;
    --border:     #004422;
    --border-hi:  #00ff88;
    --text-body:  #a0e8b0;
    --text-dim:   #1a5c30;
}

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Share Tech Mono', monospace;
    background-color: var(--bg);
    color: var(--phosphor);
}
.stApp { background: var(--bg); }
.block-container {
    padding: 1.5rem 2.5rem 4rem;
    max-width: 1280px;
}

/* ── CRT scanline overlay ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.08) 2px,
        rgba(0,0,0,0.08) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ── Flicker animation ── */
@keyframes flicker {
    0%,100% { opacity: 1; }
    92%      { opacity: 1; }
    93%      { opacity: 0.85; }
    94%      { opacity: 1; }
    96%      { opacity: 0.9; }
    97%      { opacity: 1; }
}
@keyframes blink {
    0%,49%  { opacity: 1; }
    50%,100%{ opacity: 0; }
}
@keyframes scan {
    0%   { top: -10%; }
    100% { top: 110%; }
}
@keyframes glow-pulse {
    0%,100% { text-shadow: 0 0 8px var(--phosphor), 0 0 20px var(--phosphor2); }
    50%     { text-shadow: 0 0 15px var(--phosphor), 0 0 40px var(--phosphor2), 0 0 60px var(--phosphor-dim); }
}
@keyframes boot-in {
    from { opacity: 0; transform: scaleY(0.02); filter: blur(4px); }
    to   { opacity: 1; transform: scaleY(1); filter: blur(0); }
}

/* ── Terminal Header ── */
.terminal-header {
    border: 1px solid var(--border);
    border-bottom: 2px solid var(--phosphor-dim);
    background: var(--panel);
    padding: 1.2rem 2rem 1rem;
    margin-bottom: 0.3rem;
    position: relative;
    animation: boot-in 0.6s ease-out both, flicker 8s infinite;
    overflow: hidden;
}
.terminal-header::after {
    content: '';
    position: absolute;
    left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, var(--phosphor), transparent);
    animation: scan 4s linear infinite;
    opacity: 0.3;
}
.terminal-bar {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1rem;
}
.terminal-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    border: 1px solid currentColor;
}
.t-red   { color: #ff5f56; border-color: #ff5f56; background: rgba(255,95,86,0.2); }
.t-amber { color: #ffbd2e; border-color: #ffbd2e; background: rgba(255,189,46,0.2); }
.t-green { color: var(--phosphor); border-color: var(--phosphor); background: rgba(0,255,136,0.2); }
.t-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-dim);
    letter-spacing: 0.15em;
    margin-left: auto;
}

.system-id {
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(1.6rem, 3.5vw, 2.8rem);
    font-weight: 900;
    letter-spacing: 0.15em;
    color: var(--phosphor);
    animation: glow-pulse 3s ease-in-out infinite;
    line-height: 1.1;
}
.system-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-dim);
    letter-spacing: 0.2em;
    margin-top: 0.4rem;
    text-transform: uppercase;
}
.system-sub .cursor {
    display: inline-block;
    width: 8px; height: 1em;
    background: var(--phosphor);
    margin-left: 3px;
    vertical-align: text-bottom;
    animation: blink 1s step-end infinite;
}

.status-row {
    display: flex;
    gap: 2rem;
    margin-top: 1rem;
    font-size: 0.72rem;
    color: var(--text-dim);
    letter-spacing: 0.1em;
}
.status-ok   { color: var(--phosphor); }
.status-amber{ color: var(--amber); }

/* ── Panel / Card ── */
.panel {
    border: 1px solid var(--border);
    background: var(--panel);
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    animation: boot-in 0.5s ease-out both;
}
.panel::before {
    content: attr(data-label);
    position: absolute;
    top: -0.55rem;
    left: 1rem;
    background: var(--panel);
    padding: 0 0.5rem;
    font-size: 0.68rem;
    letter-spacing: 0.2em;
    color: var(--phosphor2);
    text-transform: uppercase;
}
.panel-highlight {
    border-color: var(--phosphor);
    box-shadow: 0 0 12px rgba(0,255,136,0.08), inset 0 0 30px rgba(0,255,136,0.02);
}

/* ── Divider ── */
.term-div {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.2rem 0;
    position: relative;
}
.term-div::after {
    content: '── ──';
    position: absolute;
    left: 50%;
    transform: translateX(-50%);
    top: -0.65rem;
    background: var(--bg);
    padding: 0 0.5rem;
    font-size: 0.6rem;
    color: var(--text-dim);
    letter-spacing: 0.3em;
}

/* ── Result display ── */
.result-block {
    border: 2px solid var(--phosphor);
    background: rgba(0,255,136,0.03);
    padding: 1.5rem 2rem;
    text-align: center;
    box-shadow: 0 0 30px rgba(0,255,136,0.08), inset 0 0 60px rgba(0,255,136,0.02);
    animation: boot-in 0.4s ease-out both;
    position: relative;
    overflow: hidden;
}
.result-block::before {
    content: '';
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(
        90deg,
        transparent, transparent 3px,
        rgba(0,255,136,0.01) 3px, rgba(0,255,136,0.01) 4px
    );
}
.result-block.negative {
    border-color: var(--red-alert);
    box-shadow: 0 0 30px rgba(255,51,51,0.08), inset 0 0 60px rgba(255,51,51,0.02);
}
.result-sentiment {
    font-family: 'VT323', monospace;
    font-size: 5rem;
    line-height: 1;
    letter-spacing: 0.1em;
    color: var(--phosphor);
    animation: glow-pulse 2s ease-in-out infinite;
}
.result-sentiment.negative {
    color: var(--red-alert);
    text-shadow: 0 0 10px var(--red-alert), 0 0 30px rgba(255,51,51,0.5);
    animation: none;
}
.result-conf {
    font-family: 'Orbitron', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: var(--amber);
    margin-top: 0.3rem;
    letter-spacing: 0.2em;
}
.result-label {
    font-size: 0.7rem;
    letter-spacing: 0.35em;
    color: var(--text-dim);
    text-transform: uppercase;
    margin-top: 0.3rem;
}

/* ── Meter bar ── */
.meter-wrap { margin: 0.8rem 0; }
.meter-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.7rem;
    color: var(--text-dim);
    letter-spacing: 0.12em;
    margin-bottom: 0.25rem;
    text-transform: uppercase;
}
.meter-outer {
    height: 12px;
    background: var(--phosphor-dim);
    border: 1px solid var(--border);
    position: relative;
    overflow: hidden;
}
.meter-inner {
    height: 100%;
    background: linear-gradient(90deg, var(--phosphor2), var(--phosphor));
    box-shadow: 0 0 8px var(--phosphor);
    transition: width 0.8s ease;
    position: relative;
}
.meter-inner::after {
    content: '';
    position: absolute;
    top: 0; right: 0; bottom: 0;
    width: 6px;
    background: rgba(255,255,255,0.4);
    filter: blur(2px);
}
.meter-inner.neg {
    background: linear-gradient(90deg, #990000, var(--red-alert));
    box-shadow: 0 0 8px var(--red-alert);
}

/* ── Comparison table ── */
.comp-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
    letter-spacing: 0.05em;
}
.comp-table th {
    border-bottom: 1px solid var(--border);
    padding: 0.5rem 0.8rem;
    text-align: left;
    color: var(--text-dim);
    font-size: 0.68rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    font-weight: normal;
}
.comp-table td {
    border-bottom: 1px solid rgba(0,68,34,0.5);
    padding: 0.55rem 0.8rem;
    color: var(--text-body);
}
.comp-table tr:hover td { background: rgba(0,255,136,0.03); }
.comp-table .pos { color: var(--phosphor); font-weight: bold; }
.comp-table .neg { color: var(--red-alert); font-weight: bold; }
.comp-table .active-row td { color: var(--phosphor); background: rgba(0,255,136,0.04); }

/* ── Streamlit widget overrides ── */
.stTextArea textarea {
    background: #000 !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
    color: var(--phosphor) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.9rem !important;
    caret-color: var(--phosphor) !important;
    line-height: 1.6 !important;
    box-shadow: inset 0 0 20px rgba(0,255,136,0.03) !important;
}
.stTextArea textarea:focus {
    border-color: var(--phosphor) !important;
    box-shadow: 0 0 8px rgba(0,255,136,0.15), inset 0 0 20px rgba(0,255,136,0.04) !important;
}
.stTextArea label { display: none !important; }

.stSelectbox > div > div {
    background: #000 !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
    color: var(--phosphor) !important;
    font-family: 'Share Tech Mono', monospace !important;
}
.stSelectbox label { display: none !important; }
[data-baseweb="select"] svg { color: var(--phosphor) !important; }

.stButton > button {
    background: transparent !important;
    border: 1px solid var(--phosphor) !important;
    border-radius: 0 !important;
    color: var(--phosphor) !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.3em !important;
    text-transform: uppercase !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    box-shadow: 0 0 10px rgba(0,255,136,0.1) !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: rgba(0,255,136,0.08) !important;
    box-shadow: 0 0 20px rgba(0,255,136,0.25) !important;
}

.stWarning {
    background: rgba(255,176,0,0.08) !important;
    border: 1px solid rgba(255,176,0,0.3) !important;
    border-radius: 0 !important;
    color: var(--amber) !important;
    font-family: 'Share Tech Mono', monospace !important;
}
.stSpinner > div { color: var(--phosphor) !important; }

h2, h3 {
    font-family: 'Orbitron', sans-serif !important;
    color: var(--phosphor2) !important;
    letter-spacing: 0.1em !important;
    font-size: 0.9rem !important;
}

/* ── Log line ── */
.log-line {
    font-size: 0.7rem;
    color: var(--text-dim);
    letter-spacing: 0.1em;
    padding: 0.15rem 0;
    border-bottom: 1px dotted rgba(0,68,34,0.3);
}
.log-line .ts { color: var(--phosphor-dim); margin-right: 0.8rem; }
.log-line .ok { color: var(--phosphor); }
.log-line .warn{ color: var(--amber); }

/* ── ASCII art stars ── */
.stars-display {
    font-family: 'VT323', monospace;
    font-size: 1.8rem;
    letter-spacing: 0.05em;
}

/* ── Footer ── */
.term-footer {
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    font-size: 0.65rem;
    color: var(--text-dim);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    rnn  = tf.keras.models.load_model("rnn_model.h5")
    lstm = tf.keras.models.load_model("lstm_model.h5")
    gru  = tf.keras.models.load_model("gru_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)
    return rnn, lstm, gru, tok

rnn_model, lstm_model, gru_model, tokenizer = load_models()
MAX_LEN = 200

def predict_sentiment(model, text):
    seq    = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    prob   = float(model.predict(padded, verbose=0)[0][0])
    label  = "POSITIVE" if prob >= 0.5 else "NEGATIVE"
    conf   = prob if prob >= 0.5 else (1 - prob)
    return label, conf, prob

MODELS = {"SimpleRNN": rnn_model, "LSTM": lstm_model, "GRU": gru_model}

import datetime
NOW = datetime.datetime.now().strftime("%H:%M:%S")
DATE = datetime.datetime.now().strftime("%Y-%m-%d")

# ─────────────────────────────────────────
# TERMINAL HEADER
# ─────────────────────────────────────────
st.markdown(f"""
<div class="terminal-header">
    <div class="terminal-bar">
        <div class="terminal-dot t-red"></div>
        <div class="terminal-dot t-amber"></div>
        <div class="terminal-dot t-green"></div>
        <div class="t-title">CINTEL-7 // SESSION {NOW} // {DATE}</div>
    </div>
    <div class="system-id">CINTEL<span style="color:#004422">-</span>7</div>
    <div class="system-sub">
        CINEMATIC INTELLIGENCE TERMINAL &nbsp;·&nbsp; SENTIMENT ANALYSIS UNIT
        <span class="cursor"></span>
    </div>
    <div class="status-row">
        <span><span class="status-ok">● </span>RNN ENGINE LOADED</span>
        <span><span class="status-ok">● </span>LSTM ENGINE LOADED</span>
        <span><span class="status-ok">● </span>GRU ENGINE LOADED</span>
        <span><span class="status-ok">● </span>TOKENIZER READY</span>
        <span><span class="status-amber">◆ </span>AWAITING INPUT</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# INPUT ROW
# ─────────────────────────────────────────
col_in, col_ctrl = st.columns([3, 1], gap="medium")

with col_in:
    st.markdown("""
    <div class="panel" data-label="INPUT // REVIEW TEXT" style="margin-bottom:0.4rem;">
        <div style="font-size:0.7rem; color:#004422; letter-spacing:0.15em; margin-bottom:0.5rem;">
            &gt; PASTE OR TYPE REVIEW BELOW. MAX SEQUENCE LEN: 200 TOKENS.
        </div>
    </div>
    """, unsafe_allow_html=True)
    review = st.text_area(
        "review",
        placeholder="> _",
        height=145,
        label_visibility="collapsed",
    )

with col_ctrl:
    st.markdown("""
    <div class="panel" data-label="CONFIG // MODEL SELECT">
        <div style="font-size:0.68rem; color:#004422; letter-spacing:0.12em; margin-bottom:0.6rem;">
            &gt; SELECT NEURAL ARCH:
        </div>
    </div>
    """, unsafe_allow_html=True)
    model_choice = st.selectbox("model", list(MODELS.keys()), label_visibility="collapsed")

    arch_info = {
        "SimpleRNN": ("⚡", "SIMPLE RNN", "FAST / LIGHTWEIGHT", "#00cc66"),
        "LSTM":      ("🧠", "LONG SHORT-TERM", "MEM / PRECISE", "#00ff88"),
        "GRU":       ("🔮", "GATED RECUR UNIT", "BALANCED / SMART", "#00ffaa"),
    }
    icon, a1, a2, col = arch_info[model_choice]
    st.markdown(f"""
    <div style="border:1px solid {col}; background:rgba(0,255,136,0.02);
                padding:0.8rem 1rem; margin-top:-0.5rem; margin-bottom:0.5rem;">
        <div style="font-size:1.5rem;">{icon}</div>
        <div style="font-family:'Orbitron',sans-serif; font-size:0.72rem;
                    color:{col}; letter-spacing:0.12em; margin-top:0.3rem;">{a1}</div>
        <div style="font-size:0.65rem; color:#1a5c30; letter-spacing:0.1em; margin-top:0.2rem;">{a2}</div>
    </div>
    """, unsafe_allow_html=True)

    run_btn = st.button("▶  EXECUTE ANALYSIS")

# ─────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────
if run_btn:
    if not review.strip():
        st.warning("> ERROR: NO INPUT DETECTED. REVIEW FIELD IS EMPTY.")
        st.stop()

    with st.spinner("> RUNNING INFERENCE..."):
        sentiment, conf, prob = predict_sentiment(MODELS[model_choice], review)

        all_results = []
        for name, m in MODELS.items():
            s, c, p = predict_sentiment(m, review)
            all_results.append({"Model": name, "Verdict": s, "Confidence": c, "PosPr": p, "NegPr": 1 - p})

    is_pos  = sentiment == "POSITIVE"
    stars_n = int(round(conf * 5))
    stars   = ("█" * stars_n) + ("░" * (5 - stars_n))

    st.markdown('<hr class="term-div">', unsafe_allow_html=True)

    # ── Log lines ──
    wc = len(review.split())
    st.markdown(f"""
    <div class="log-line"><span class="ts">[{NOW}]</span> INPUT RECEIVED · <span class="ok">{wc} TOKENS</span></div>
    <div class="log-line"><span class="ts">[SYS]</span> SELECTED MODEL: <span class="ok">{model_choice.upper()}</span></div>
    <div class="log-line"><span class="ts">[INF]</span> FORWARD PASS COMPLETE · PROB TENSOR EXTRACTED</div>
    <div class="log-line"><span class="ts">[OUT]</span> CLASSIFICATION: <span class="{'ok' if is_pos else 'warn'}">{sentiment}</span> · CONFIDENCE: <span class="{'ok' if is_pos else 'warn'}">{conf*100:.2f}%</span></div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Primary Result ──
    r1, r2 = st.columns([1.2, 1], gap="large")

    with r1:
        neg_cls = "" if is_pos else " negative"
        st.markdown(f"""
        <div class="result-block{neg_cls}">
            <div class="result-label">CLASSIFICATION OUTPUT</div>
            <div class="result-sentiment{'.' + 'negative' if not is_pos else ''}">{sentiment}</div>
            <div class="result-conf">{conf*100:.1f}%</div>
            <div class="result-label">CONFIDENCE SCORE</div>
            <div class="stars-display" style="margin-top:0.8rem; color:{'var(--phosphor)' if is_pos else 'var(--red-alert)'};">
                [{stars}]
            </div>
            <div style="font-size:0.68rem; color:#004422; margin-top:0.5rem; letter-spacing:0.12em;">
                SIGNAL STRENGTH: {stars_n}/5
            </div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        st.markdown(f"""
        <div class="panel" data-label="PROBABILITY METERS">
            <div class="meter-wrap">
                <div class="meter-label">
                    <span>▶ POSITIVE</span>
                    <span style="color:var(--phosphor)">{prob:.4f}</span>
                </div>
                <div class="meter-outer">
                    <div class="meter-inner" style="width:{prob*100}%"></div>
                </div>
            </div>
            <div class="meter-wrap">
                <div class="meter-label">
                    <span>▶ NEGATIVE</span>
                    <span style="color:var(--red-alert)">{1-prob:.4f}</span>
                </div>
                <div class="meter-outer">
                    <div class="meter-inner neg" style="width:{(1-prob)*100}%"></div>
                </div>
            </div>
            <div style="height:0.8rem"></div>
            <div class="meter-wrap">
                <div class="meter-label"><span>▶ CONFIDENCE</span><span style="color:var(--amber)">{conf:.4f}</span></div>
                <div class="meter-outer">
                    <div class="meter-inner" style="width:{conf*100}%; background: linear-gradient(90deg,#cc8800,var(--amber)); box-shadow:0 0 8px var(--amber);"></div>
                </div>
            </div>
        </div>
        <div class="panel" data-label="STATS" style="margin-top:0.8rem;">
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.5rem 1.5rem;">
                <div><div style="font-size:0.65rem;color:#004422;letter-spacing:0.15em;">TOKENS</div>
                     <div style="font-family:'Orbitron',sans-serif;font-size:1.1rem;color:var(--phosphor2);">{wc}</div></div>
                <div><div style="font-size:0.65rem;color:#004422;letter-spacing:0.15em;">MODEL</div>
                     <div style="font-family:'Orbitron',sans-serif;font-size:1.1rem;color:var(--phosphor2);">{model_choice[:3].upper()}</div></div>
                <div><div style="font-size:0.65rem;color:#004422;letter-spacing:0.15em;">POS LOGIT</div>
                     <div style="font-family:'Orbitron',sans-serif;font-size:1.1rem;color:var(--phosphor2);">{prob:.3f}</div></div>
                <div><div style="font-size:0.65rem;color:#004422;letter-spacing:0.15em;">NEG LOGIT</div>
                     <div style="font-family:'Orbitron',sans-serif;font-size:1.1rem;color:var(--phosphor2);">{1-prob:.3f}</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="term-div">', unsafe_allow_html=True)

    # ── Multi-model comparison ──
    st.markdown('<div class="panel" data-label="MULTI-MODEL COMPARISON // ALL ARCHITECTURES">', unsafe_allow_html=True)

    rows_html = ""
    for r in all_results:
        active = "active-row" if r["Model"] == model_choice else ""
        cls    = "pos" if r["Verdict"] == "POSITIVE" else "neg"
        bar_w  = int(r["Confidence"] * 100)
        bar_c  = "var(--phosphor)" if r["Verdict"] == "POSITIVE" else "var(--red-alert)"
        rows_html += f"""
        <tr class="{active}">
            <td style="font-family:'Orbitron',sans-serif;font-size:0.78rem;color:var(--phosphor2);">
                {'▶ ' if r['Model'] == model_choice else '  '}{r['Model'].upper()}
            </td>
            <td class="{cls}">{r['Verdict']}</td>
            <td>
                <div style="display:flex;align-items:center;gap:0.6rem;">
                    <div style="flex:1;height:8px;background:var(--phosphor-dim);border:1px solid var(--border);">
                        <div style="height:100%;width:{bar_w}%;background:{bar_c};box-shadow:0 0 5px {bar_c};"></div>
                    </div>
                    <span style="font-size:0.72rem;color:var(--text-body);min-width:3.5rem;">{r['Confidence']*100:.1f}%</span>
                </div>
            </td>
            <td style="color:var(--phosphor)">{r['PosPr']:.4f}</td>
            <td style="color:var(--red-alert)">{r['NegPr']:.4f}</td>
        </tr>"""

    st.markdown(f"""
    <table class="comp-table">
        <thead>
            <tr>
                <th>ARCHITECTURE</th>
                <th>VERDICT</th>
                <th>CONFIDENCE BAR</th>
                <th>POS PROB</th>
                <th>NEG PROB</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Plotly radar ──
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    labels = [r["Model"] for r in all_results]
    pos_v  = [r["PosPr"] for r in all_results]
    neg_v  = [r["NegPr"] for r in all_results]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=pos_v + [pos_v[0]], theta=labels + [labels[0]],
        name="POSITIVE", fill="toself",
        line=dict(color="#00ff88", width=2),
        fillcolor="rgba(0,255,136,0.07)",
        marker=dict(color="#00ff88", size=7),
    ))
    fig.add_trace(go.Scatterpolar(
        r=neg_v + [neg_v[0]], theta=labels + [labels[0]],
        name="NEGATIVE", fill="toself",
        line=dict(color="#ff3333", width=2),
        fillcolor="rgba(255,51,51,0.07)",
        marker=dict(color="#ff3333", size=7),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Share Tech Mono", color="#1a5c30", size=11),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            angularaxis=dict(
                linecolor="#004422", gridcolor="#011a09",
                tickfont=dict(family="Orbitron", size=10, color="#00cc66"),
            ),
            radialaxis=dict(
                linecolor="#004422", gridcolor="#011a09",
                tickfont=dict(size=9, color="#004422"),
                range=[0, 1],
            ),
        ),
        legend=dict(
            font=dict(family="Share Tech Mono", color="#1a5c30", size=10),
            bgcolor="rgba(0,0,0,0)", bordercolor="#004422", borderwidth=1,
        ),
        margin=dict(l=50, r=50, t=30, b=30),
        height=280,
        title=dict(text="// RADAR: POS vs NEG PROBABILITY", font=dict(family="Orbitron", size=10, color="#004422")),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Review echo ──
    preview = review[:300] + ("..." if len(review) > 300 else "")
    st.markdown(f"""
    <div class="panel" data-label="PROCESSED INPUT // REVIEW ECHO">
        <div style="font-size:0.82rem; color:#1a5c30; letter-spacing:0.05em; line-height:1.8;">
            &gt; {preview}
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Idle state ──
    st.markdown("""
    <div style="padding:3rem 1rem; text-align:center;">
        <div style="font-family:'VT323',monospace; font-size:5rem; color:#011a09; animation:none;">
            ░░░░░░░░░░░░░░░░░
        </div>
        <div style="font-family:'Orbitron',sans-serif; font-size:0.78rem;
                    color:#004422; letter-spacing:0.3em; margin-top:1rem;">
            SYSTEM IDLE // AWAITING REVIEW INPUT
        </div>
        <div style="font-size:0.65rem; color:#011a09; letter-spacing:0.2em; margin-top:0.5rem; text-transform:uppercase;">
            Insert film review to begin sentiment classification
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown(f"""
<div class="term-footer">
    <span>CINTEL-7 · SENTIMENT ANALYSIS TERMINAL</span>
    <span>MODELS: RNN · LSTM · GRU</span>
    <span>DATASET: IMDB 50K</span>
    <span>SESSION: {DATE} {NOW}</span>
</div>
""", unsafe_allow_html=True)

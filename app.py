import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time
import re
import joblib

st.set_page_config(
    page_title="FLASHLENS AI",
    page_icon="🔦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Session State ───────────────────────────────────────────────────────────────
if "logged_in"   not in st.session_state: st.session_state.logged_in   = False
if "history"     not in st.session_state: st.session_state.history     = []
if "username"    not in st.session_state: st.session_state.username    = ""
if "last_result" not in st.session_state: st.session_state.last_result = None

# ─── Load Models ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        sm = joblib.load("sms_model.pkl")
        sv = joblib.load("sms_vectorizer.pkl")
        em = joblib.load("email_model.pkl")
        ev = joblib.load("email_vectorizer.pkl")
        return sm, sv, em, ev, True
    except:
        return None, None, None, None, False

sms_model, sms_vec, email_model, email_vec, models_loaded = load_models()

# ─── Auto Detect ─────────────────────────────────────────────────────────────────
def auto_detect(text):
    t = text.lower().strip()
    strong = ["subject:", "from:", "to:", "cc:", "bcc:", "reply-to:", "unsubscribe",
              "dear sir", "dear madam", "dear customer", "dear user",
              "warm regards", "best regards", "yours faithfully", "sincerely yours",
              "view in browser", "this email was sent"]
    if any(s in t for s in strong): return "email"
    if re.search(r'\S+@\S+\.\S{2,}', t): return "email"
    if re.search(r'\b(from|to|subject)\s*:', t, re.I): return "email"
    weak = ["dear ", "regards", "sincerely", "greetings", "good morning",
            "please find", "attached", "hereby", "kindly", "enclosed"]
    wc = sum(1 for s in weak if s in t)
    if wc >= 2: return "email"
    if len(text) > 300 and wc >= 1: return "email"
    return "sms"

# ─── Keyword Fallback ────────────────────────────────────────────────────────────
SCAM_KW = ["lottery","winner","prize","claim","urgent","free money","bank account",
           "otp","password","verify","click here","guaranteed return","cryptocurrency",
           "bitcoin","send money","western union","gift card","inheritance",
           "congratulations","lucky","double your money","risk free","kyc update",
           "account blocked","suspended","refund","jackpot","limited time","act now",
           "expire","fraud","scam","cash prize","call now","free offer","win","won"]
SPATS = [r'\b\d{10,}\b',
         r'(?:https?://)?(?:bit\.ly|tinyurl|t\.co|goo\.gl|cutt\.ly)',
         r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
         r'(?i)(?:send|transfer|pay|deposit)\s+(?:rs\.?|inr|₹)?\s*\d+']

def kw_fallback(text):
    score, flags = 0, []
    hits = [k for k in SCAM_KW if k in text.lower()]
    if hits:
        score += min(len(hits)*14, 60)
        flags.append(f"🔴 Scam keywords: {', '.join(hits[:5])}")
    for p in SPATS:
        if re.search(p, text, re.I):
            score += 10; flags.append("🟠 Suspicious pattern detected"); break
    if len(text) < 80 and any(w in text.lower() for w in ["click","call","reply","send","act"]):
        score += 10; flags.append("🟡 Short action-forcing message")
    if sum(1 for c in text if c.isupper()) / max(len(text),1) > 0.3:
        score += 8; flags.append("🟡 Excessive capital letters")
    return min(score, 99), flags

# ─── Analyze ─────────────────────────────────────────────────────────────────────
def analyze(text, mtype):
    if not text.strip(): return None
    flags = []
    if models_loaded:
        try:
            vec   = email_vec   if mtype == "email" else sms_vec
            model = email_model if mtype == "email" else sms_model
            lbl   = "📧 EMAIL Model" if mtype == "email" else "💬 SMS Model"
            X = vec.transform([text])
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X)[0]
                score = int(p[1]*100) if len(p) > 1 else int(p[0]*100)
            else:
                score = 88 if model.predict(X)[0] == 1 else 12
            flags.append(f"🤖 {lbl} used")
        except Exception as e:
            score, flags = kw_fallback(text)
            flags.append(f"⚠️ Fallback used")
    else:
        score, flags = kw_fallback(text)
        flags.append("⚠️ Keyword fallback — models not loaded")
    score = min(int(score), 99)
    if score >= 60:   level, color = "SCAM",       "#ff3a5c"
    elif score >= 40: level, color = "SUSPICIOUS",  "#ffb020"
    else:             level, color = "SAFE",         "#00ffa3"
    if not flags: flags.append("✅ No suspicious patterns detected")
    return {
        "score": score, "level": level, "color": color, "flags": flags,
        "timestamp": datetime.now().strftime("%d %b · %H:%M"),
        "type": mtype.upper(),
        "preview": text[:65]+"…" if len(text) > 65 else text
    }

# ─── Charts ──────────────────────────────────────────────────────────────────────
def gauge_chart(score, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        domain={'x':[0,1],'y':[0,1]},
        number={'font':{'size':54,'color':color},'suffix':'%'},
        gauge={
            'axis':{'range':[0,100],'tickcolor':'#1f2d45','tickfont':{'size':9,'color':'#3d5478'},'tickvals':[0,20,40,60,70,80,100]},
            'bar':{'color':color,'thickness':0.2},
            'bgcolor':'#0b1120','borderwidth':0,
            'steps':[{'range':[0,40],'color':'rgba(0,255,163,0.04)'},
                     {'range':[40,70],'color':'rgba(255,176,32,0.04)'},
                     {'range':[70,100],'color':'rgba(255,58,92,0.06)'}],
            'threshold':{'line':{'color':color,'width':3},'thickness':0.85,'value':score}
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=20,r=20,t=15,b=5),height=255)
    return fig

def timeline_chart(history):
    if not history: return None
    df = pd.DataFrame(history)
    cm = {"SCAM":"#ff3a5c","SUSPICIOUS":"#ffb020","SAFE":"#00ffa3"}
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(df))), y=df['score'], mode='lines+markers',
        line=dict(color='rgba(0,229,255,0.5)',width=2,shape='spline'),
        marker=dict(color=[cm.get(l,'#00e5ff') for l in df['level']],size=10,line=dict(color='#060912',width=2)),
        fill='tozeroy', fillcolor='rgba(0,229,255,0.03)',
        hovertemplate='<b>%{y}% — %{text}</b><extra></extra>', text=df['level']
    ))
    fig.add_hline(y=70,line_dash="dash",line_color="rgba(255,58,92,0.4)",annotation_text="SCAM",annotation_font_color="#ff3a5c",annotation_font_size=9)
    fig.add_hline(y=40,line_dash="dash",line_color="rgba(255,176,32,0.35)",annotation_text="SUSPICIOUS",annotation_font_color="#ffb020",annotation_font_size=9)
    fig.add_hrect(y0=70,y1=100,fillcolor="rgba(255,58,92,0.03)",line_width=0)
    fig.add_hrect(y0=40,y1=70,fillcolor="rgba(255,176,32,0.025)",line_width=0)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=10,r=10,t=10,b=10),height=215,
                      xaxis=dict(showgrid=False,color='#3d5478'),
                      yaxis=dict(showgrid=True,gridcolor='rgba(31,45,69,0.2)',color='#3d5478',range=[0,108]),
                      showlegend=False)
    return fig

def donut_chart(history):
    if not history: return None
    df  = pd.DataFrame(history)
    cnt = df['level'].value_counts().reset_index()
    cnt.columns = ['level','count']
    cm  = {"SCAM":"#ff3a5c","SUSPICIOUS":"#ffb020","SAFE":"#00ffa3"}
    fig = go.Figure(go.Pie(
        labels=cnt['level'],values=cnt['count'],hole=0.65,
        marker=dict(colors=[cm.get(l,'#00e5ff') for l in cnt['level']],line=dict(color='#060912',width=3)),
        hovertemplate='<b>%{label}</b><br>%{value} · %{percent}<extra></extra>'
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=10,r=10,t=10,b=10),height=215,
                      legend=dict(font=dict(color='#b8cce4'),bgcolor='rgba(0,0,0,0)'))
    return fig

def bar_chart(history):
    if not history: return None
    sc   = [h['score'] for h in history]
    cols = ['#ff3a5c' if s>=70 else '#ffb020' if s>=40 else '#00ffa3' for s in sc]
    fig  = go.Figure(go.Bar(
        x=list(range(len(sc))),y=sc,
        marker=dict(color=cols,line=dict(color='#060912',width=1)),
        hovertemplate='Scan #%{x}<br>Score: <b>%{y}%</b><extra></extra>'
    ))
    fig.add_hline(y=70,line_dash="dot",line_color="rgba(255,58,92,0.4)")
    fig.add_hline(y=40,line_dash="dot",line_color="rgba(255,176,32,0.4)")
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=10,r=10,t=10,b=10),height=185,
                      xaxis=dict(showgrid=False,color='#3d5478'),
                      yaxis=dict(showgrid=True,gridcolor='rgba(31,45,69,0.2)',color='#3d5478',range=[0,108]),
                      bargap=0.15)
    return fig


# ════════════════════════════════════════════════════════════════════════════════
#  LOGIN PAGE  —  Simple centered card, fully working
# ════════════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, .stApp {
        background: #060912 !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    .block-container { padding-top: 0 !important; padding-bottom: 0 !important; max-width: 100% !important; }
    #MainMenu, footer, header { visibility: hidden; }

    /* Full screen dark bg with glow */
    .stApp {
        background:
            radial-gradient(ellipse 700px 500px at 20% 30%, rgba(0,229,255,0.07) 0%, transparent 60%),
            radial-gradient(ellipse 600px 400px at 80% 70%, rgba(162,89,255,0.06) 0%, transparent 60%),
            #060912 !important;
    }

    /* Animated grid */
    .stApp::before {
        content: '';
        position: fixed; inset: 0;
        background-image:
            linear-gradient(rgba(0,229,255,0.02) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,229,255,0.02) 1px, transparent 1px);
        background-size: 55px 55px;
        animation: gridmove 25s linear infinite;
        pointer-events: none; z-index: 0;
    }
    @keyframes gridmove { to { background-position: 55px 55px; } }

    /* Center the login form */
    .login-outer {
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        z-index: 10;
        padding: 20px;
    }

    /* The card */
    .login-card {
        width: 100%;
        max-width: 440px;
        background: linear-gradient(145deg, rgba(11,17,32,0.97), rgba(17,24,39,0.95));
        border: 1px solid rgba(0,229,255,0.14);
        border-radius: 24px;
        padding: 52px 44px 44px;
        box-shadow:
            0 0 0 1px rgba(0,229,255,0.04),
            0 0 80px rgba(0,229,255,0.08),
            0 50px 100px rgba(0,0,0,0.7),
            inset 0 1px 0 rgba(255,255,255,0.03);
        animation: cardin 0.8s cubic-bezier(0.34,1.56,0.64,1) both;
        position: relative;
        overflow: hidden;
    }
    .login-card::before {
        content: '';
        position: absolute; top: 0; left: 10%; right: 10%; height: 1px;
        background: linear-gradient(90deg, transparent, #00e5ff 30%, #a259ff 70%, transparent);
        animation: shimmer 4s ease-in-out infinite;
    }
    @keyframes shimmer { 0%,100%{opacity:0.5} 50%{opacity:1} }
    @keyframes cardin {
        from { opacity:0; transform: translateY(50px) scale(0.93); }
        to   { opacity:1; transform: none; }
    }

    /* Logo area */
    .lc-logo {
        text-align: center;
        margin-bottom: 38px;
    }
    .lc-icon {
        width: 72px; height: 72px; border-radius: 50%;
        background: linear-gradient(135deg, rgba(0,229,255,0.12), rgba(162,89,255,0.12));
        border: 1px solid rgba(0,229,255,0.22);
        display: flex; align-items: center; justify-content: center;
        margin: 0 auto 18px;
        font-size: 2rem;
        animation: iconpulse 3.5s ease-in-out infinite;
        box-shadow: 0 0 30px rgba(0,229,255,0.1);
    }
    @keyframes iconpulse {
        0%,100%{ box-shadow: 0 0 20px rgba(0,229,255,0.1); }
        50%{ box-shadow: 0 0 45px rgba(162,89,255,0.25); }
    }
    .lc-title {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 1.9rem;
        background: linear-gradient(135deg, #00e5ff, #a259ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .lc-sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        color: #3d5478;
        letter-spacing: 3px;
        text-transform: uppercase;
    }
    .lc-divider {
        width: 50px; height: 2px;
        background: linear-gradient(90deg, #00e5ff, #a259ff);
        border-radius: 2px;
        margin: 14px auto 0;
    }

    /* Pulse dot */
    .pulse-row {
        display: flex; align-items: center; justify-content: center;
        gap: 7px; margin-top: 10px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.62rem; color: #3d5478;
    }
    .pulse-dot {
        width: 7px; height: 7px; border-radius: 50%;
        background: #00ffa3;
        animation: blink 2s ease-in-out infinite;
        box-shadow: 0 0 8px #00ffa3;
    }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

    /* Input styling */
    .stTextInput > div > div > input {
        background: rgba(5, 10, 22, 0.85) !important;
        border: 1px solid #2a3f5f !important;
        border-radius: 12px !important;
        color: #e6f0ff !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 1rem !important;
        padding: 13px 18px !important;
        transition: all 0.25s !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #00e5ff !important;
        box-shadow: 0 0 0 2px rgba(0,229,255,0.12), 0 0 25px rgba(0,229,255,0.07) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #3d5478 !important;
    }
    .stTextInput label {
        color: #3d5478 !important;
        font-size: 0.68rem !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Login button */
    .stButton > button {
        background: linear-gradient(135deg, #00e5ff 0%, #a259ff 100%) !important;
        color: #060912 !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 800 !important;
        font-size: 1rem !important;
        letter-spacing: 2px !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px !important;
        width: 100% !important;
        transition: all 0.3s !important;
        text-transform: uppercase !important;
        box-shadow: 0 4px 25px rgba(0,229,255,0.25) !important;
        margin-top: 6px !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 40px rgba(0,229,255,0.35) !important;
    }

    /* Feature pills at bottom of card */
    .feat-pills {
        display: flex;
        gap: 8px;
        justify-content: center;
        flex-wrap: wrap;
        margin-top: 24px;
    }
    .pill {
        display: inline-flex; align-items: center; gap: 5px;
        background: rgba(0,229,255,0.06);
        border: 1px solid rgba(0,229,255,0.14);
        border-radius: 20px;
        padding: 4px 12px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.62rem;
        color: #3d5478;
    }
    </style>
    """, unsafe_allow_html=True)

    # Centered outer div
    st.markdown('<div class="login-outer">', unsafe_allow_html=True)

    # Use columns to center the card
    _, mid, _ = st.columns([1, 1.2, 1])

    with mid:
        # Card top HTML
        st.markdown("""
        <div class="login-card">
            <div class="lc-logo">
                <div class="lc-icon">🔦</div>
                <div class="lc-title">FLASHLENS AI</div>
                <div class="lc-sub">Scam Detection System</div>
                <div class="lc-divider"></div>
                <div class="pulse-row">
                    <div class="pulse-dot"></div>
                    <span>System Online · ML Models Ready</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Streamlit input — this is the KEY part that works
        uname = st.text_input(
            "Username",
            placeholder="Apna naam likhein...",
            key="login_user",
            label_visibility="visible"
        )

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        # Login button
        if st.button("⚡  Launch Dashboard", use_container_width=True, key="login_btn"):
            if uname.strip():
                st.session_state.logged_in = True
                st.session_state.username  = uname.strip().title()
                st.rerun()
            else:
                st.error("⚠️ Pehle apna naam likhein!")

        # Bottom pills
        st.markdown("""
        <div class="feat-pills">
            <span class="pill">🛡️ Smart Detection</span>
            <span class="pill">⚡ Real-time</span>
            <span class="pill">🔒 Secure</span>
        </div>
        <div style="text-align:center; margin-top:18px;
            font-family:'JetBrains Mono',monospace; font-size:0.6rem; color:#1f2d45;">
            SMS · EMAIL · AI POWERED
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()


# ════════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500;700&family=DM+Sans:wght@400;500;600&display=swap');

:root {
    --bg:     #060912;
    --surf:   #0b1120;
    --surf2:  #111827;
    --bord:   #1f2d45;
    --bord2:  #2a3f5f;
    --cyan:   #00e5ff;
    --violet: #a259ff;
    --green:  #00ffa3;
    --red:    #ff3a5c;
    --amber:  #ffb020;
    --text:   #b8cce4;
    --dim:    #3d5478;
    --bright: #e6f0ff;
}
*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.block-container { padding-top: 1rem !important; padding-bottom: 1rem !important; }
#MainMenu, footer, header { visibility: hidden; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--bord); border-radius: 3px; }

.stApp {
    background:
        radial-gradient(ellipse 900px 600px at 10% 20%, rgba(0,229,255,0.048) 0%, transparent 60%),
        radial-gradient(ellipse 700px 500px at 90% 80%, rgba(162,89,255,0.04) 0%, transparent 60%),
        var(--bg) !important;
}

/* Grid */
.grid-overlay {
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image:
        linear-gradient(rgba(0,229,255,0.016) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,229,255,0.016) 1px, transparent 1px);
    background-size: 60px 60px;
    animation: gm 30s linear infinite;
}
@keyframes gm { to { background-position: 60px 60px; } }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060e1c, #040b16) !important;
    border-right: 1px solid rgba(31,45,69,0.55) !important;
}
.sb-h { padding:26px 18px 16px; border-bottom:1px solid var(--bord); text-align:center; }
.sb-av {
    width:66px; height:66px; border-radius:50%;
    background:linear-gradient(135deg, var(--cyan), var(--violet));
    display:flex; align-items:center; justify-content:center;
    font-family:'Syne',sans-serif; font-weight:800; font-size:1.75rem; color:#fff;
    margin:0 auto 10px;
    box-shadow:0 0 0 3px rgba(0,229,255,0.14), 0 0 28px rgba(0,229,255,0.1);
    animation:ag 4s ease-in-out infinite;
}
@keyframes ag {
    0%,100%{box-shadow:0 0 0 3px rgba(0,229,255,0.14),0 0 28px rgba(0,229,255,0.09);}
    50%{box-shadow:0 0 0 5px rgba(0,229,255,0.26),0 0 46px rgba(0,229,255,0.2);}
}
.sb-nm { font-family:'Syne',sans-serif; font-weight:700; font-size:0.92rem; color:var(--bright); }
.sb-rl { font-family:'JetBrains Mono',monospace; font-size:0.56rem; color:var(--cyan); letter-spacing:3px; margin-top:3px; }
.sb-on { display:flex; align-items:center; justify-content:center; gap:6px; margin-top:7px; font-family:'JetBrains Mono',monospace; font-size:0.57rem; color:var(--dim); }
.sb-dt { width:6px; height:6px; border-radius:50%; background:var(--green); animation:blk 2s ease-in-out infinite; }
@keyframes blk { 0%,100%{opacity:1} 50%{opacity:0.2} }

.sb-st { padding:13px 18px; border-bottom:1px solid var(--bord); }
.sb-lb { font-family:'JetBrains Mono',monospace; font-size:0.55rem; color:var(--dim); letter-spacing:2px; text-transform:uppercase; margin-bottom:9px; }
.sb-rw { display:flex; justify-content:space-between; align-items:center; padding:6px 0; border-bottom:1px solid rgba(31,45,69,0.22); }
.sb-rw:last-child { border-bottom:none; }
.sb-k { font-size:0.66rem; color:var(--dim); font-family:'JetBrains Mono',monospace; }
.sb-v { font-family:'Syne',sans-serif; font-weight:700; font-size:0.88rem; }

.sb-tp { padding:13px 18px; }
.tp-it { display:flex; gap:9px; padding:8px 11px; margin-bottom:6px; background:rgba(31,45,69,0.1); border:1px solid rgba(31,45,69,0.32); border-radius:10px; font-size:0.74rem; color:var(--dim); line-height:1.5; transition:border-color 0.3s; }
.tp-it:hover { border-color:rgba(0,229,255,0.18); color:var(--text); }

/* Topbar */
.tbar { display:flex; align-items:center; justify-content:space-between; padding:10px 0 16px; border-bottom:1px solid var(--bord); margin-bottom:20px; position:relative; z-index:5; }
.tbar-logo { font-family:'Syne',sans-serif; font-weight:800; font-size:1.45rem; letter-spacing:2px; background:linear-gradient(135deg,var(--cyan),var(--violet)); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.tbar-meta { font-family:'JetBrains Mono',monospace; font-size:0.61rem; color:var(--dim); text-align:right; line-height:1.8; }

/* Metric cards */
.mc { background:linear-gradient(145deg,var(--surf),var(--surf2)); border:1px solid var(--bord); border-radius:16px; padding:18px 20px; text-align:center; position:relative; overflow:hidden; transition:transform 0.3s,box-shadow 0.3s; }
.mc:hover { transform:translateY(-4px); box-shadow:0 16px 50px rgba(0,229,255,0.08); }
.mc::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,transparent,var(--cyan),transparent); }
.mc-v { font-family:'Syne',sans-serif; font-weight:800; font-size:2.4rem; line-height:1; background:linear-gradient(135deg,var(--cyan),var(--violet)); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.mc-l { font-family:'JetBrains Mono',monospace; font-size:0.56rem; color:var(--dim); letter-spacing:2px; text-transform:uppercase; margin-top:4px; }
.mc-r::before { background:linear-gradient(90deg,transparent,var(--red),transparent); }
.mc-r .mc-v { background:linear-gradient(135deg,var(--red),#ff8a65); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.mc-a::before { background:linear-gradient(90deg,transparent,var(--amber),transparent); }
.mc-a .mc-v { background:linear-gradient(135deg,var(--amber),#ffe080); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.mc-g::before { background:linear-gradient(90deg,transparent,var(--green),transparent); }
.mc-g .mc-v { background:linear-gradient(135deg,var(--green),var(--cyan)); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }

.sec-t { font-family:'Syne',sans-serif; font-weight:700; font-size:0.74rem; color:var(--cyan); letter-spacing:3px; text-transform:uppercase; margin-bottom:14px; padding-bottom:8px; border-bottom:1px solid var(--bord); }

/* Alerts */
.a-sc { background:linear-gradient(135deg,rgba(255,58,92,0.1),rgba(255,58,92,0.02)); border:1px solid rgba(255,58,92,0.5); border-radius:16px; padding:22px 24px; animation:pr 2.5s ease-in-out infinite; }
@keyframes pr { 0%,100%{box-shadow:0 0 22px rgba(255,58,92,0.16);} 50%{box-shadow:0 0 50px rgba(255,58,92,0.44);} }
.a-su { background:linear-gradient(135deg,rgba(255,176,32,0.09),rgba(255,176,32,0.02)); border:1px solid rgba(255,176,32,0.5); border-radius:16px; padding:22px 24px; animation:pa 3s ease-in-out infinite; }
@keyframes pa { 0%,100%{box-shadow:0 0 18px rgba(255,176,32,0.14);} 50%{box-shadow:0 0 42px rgba(255,176,32,0.38);} }
.a-sf { background:linear-gradient(135deg,rgba(0,255,163,0.08),rgba(0,255,163,0.01)); border:1px solid rgba(0,255,163,0.4); border-radius:16px; padding:22px 24px; box-shadow:0 0 26px rgba(0,255,163,0.08); }

.bdg { display:inline-flex; align-items:center; gap:5px; border-radius:20px; padding:4px 14px; font-family:'JetBrains Mono',monospace; font-size:0.65rem; font-weight:700; letter-spacing:1.5px; }
.b-r { background:rgba(255,58,92,0.12); color:var(--red); border:1px solid rgba(255,58,92,0.4); }
.b-a { background:rgba(255,176,32,0.12); color:var(--amber); border:1px solid rgba(255,176,32,0.4); }
.b-g { background:rgba(0,255,163,0.1); color:var(--green); border:1px solid rgba(0,255,163,0.35); }
.b-c { background:rgba(0,229,255,0.1); color:var(--cyan); border:1px solid rgba(0,229,255,0.3); }

.dpill { display:inline-flex; align-items:center; gap:8px; background:rgba(0,229,255,0.06); border:1px solid rgba(0,229,255,0.17); border-radius:30px; padding:7px 18px; margin-bottom:12px; font-family:'JetBrains Mono',monospace; font-size:0.65rem; }
.ddot { width:8px; height:8px; border-radius:50%; animation:blk 2s ease-in-out infinite; }

.flg { background:rgba(11,17,32,0.7); border-left:3px solid var(--bord2); border-radius:0 10px 10px 0; padding:10px 16px; margin-bottom:8px; font-size:0.84rem; color:var(--text); transition:border-color 0.3s,background 0.3s; }
.flg:hover { border-left-color:var(--cyan); background:rgba(0,229,255,0.04); }

.hit { background:linear-gradient(135deg,rgba(11,17,32,0.85),rgba(17,24,39,0.6)); border:1px solid rgba(31,45,69,0.5); border-radius:12px; padding:14px 18px; margin-bottom:8px; display:flex; justify-content:space-between; align-items:center; transition:transform 0.2s; }
.hit:hover { transform:translateX(5px); }

.awt { text-align:center; padding:58px 20px; border:1px dashed rgba(31,45,69,0.48); border-radius:16px; background:rgba(7,14,28,0.4); }
.awt-i { font-size:3.4rem; margin-bottom:13px; animation:fl 3.5s ease-in-out infinite; display:block; }
@keyframes fl { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-12px)} }
.awt-t { font-family:'Syne',sans-serif; font-weight:700; font-size:0.92rem; letter-spacing:4px; color:var(--dim); }
.awt-s { font-family:'JetBrains Mono',monospace; font-size:0.66rem; color:rgba(61,84,120,0.7); margin-top:8px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background:var(--surf)!important; border-radius:12px; padding:5px; border:1px solid var(--bord)!important; gap:4px; }
.stTabs [data-baseweb="tab"] { background:transparent!important; color:var(--dim)!important; font-family:'Syne',sans-serif!important; font-weight:600!important; border-radius:8px!important; border:none!important; transition:all 0.25s!important; }
.stTabs [aria-selected="true"] { background:linear-gradient(135deg,rgba(0,229,255,0.1),rgba(162,89,255,0.1))!important; color:var(--cyan)!important; border:1px solid rgba(0,229,255,0.22)!important; }

/* Inputs */
.stTextInput>div>div>input { background:rgba(5,10,22,0.9)!important; border:1px solid var(--bord2)!important; border-radius:12px!important; color:var(--bright)!important; font-family:'DM Sans',sans-serif!important; font-size:1rem!important; padding:13px 18px!important; transition:all 0.3s!important; }
.stTextInput>div>div>input:focus { border-color:var(--cyan)!important; box-shadow:0 0 0 2px rgba(0,229,255,0.1)!important; }
.stTextArea>div>div>textarea { background:rgba(5,10,22,0.9)!important; border:1px solid var(--bord2)!important; border-radius:12px!important; color:var(--bright)!important; font-family:'DM Sans',sans-serif!important; font-size:0.95rem!important; }
.stTextArea>div>div>textarea:focus { border-color:var(--cyan)!important; box-shadow:0 0 0 2px rgba(0,229,255,0.1)!important; }
.stTextInput label, .stTextArea label { color:var(--dim)!important; font-size:0.65rem!important; letter-spacing:2px!important; text-transform:uppercase!important; font-family:'JetBrains Mono',monospace!important; }

/* Buttons (dashboard) */
.stButton>button { background:linear-gradient(135deg,var(--cyan) 0%,var(--violet) 100%)!important; color:#060912!important; font-family:'Syne',sans-serif!important; font-weight:800!important; letter-spacing:2px!important; border:none!important; border-radius:12px!important; padding:14px!important; transition:all 0.3s!important; text-transform:uppercase!important; width:100%!important; }
.stButton>button:hover { transform:translateY(-2px)!important; box-shadow:0 14px 45px rgba(0,229,255,0.26)!important; }
.stSelectbox>div>div { background:rgba(5,10,22,0.9)!important; border:1px solid var(--bord2)!important; border-radius:12px!important; color:var(--bright)!important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="grid-overlay"></div>', unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────────────────────────
total  = len(st.session_state.history)
scams  = sum(1 for h in st.session_state.history if h['level']=='SCAM')
suspic = sum(1 for h in st.session_state.history if h['level']=='SUSPICIOUS')
safe   = sum(1 for h in st.session_state.history if h['level']=='SAFE')
init   = st.session_state.username[0].upper() if st.session_state.username else "?"
etxt   = "ML ✓" if models_loaded else "KEYWORD"
ecol   = "#00ffa3" if models_loaded else "#ffb020"

with st.sidebar:
    st.markdown(f"""
    <div class="sb-h">
        <div class="sb-av">{init}</div>
        <div class="sb-nm">{st.session_state.username}</div>
        <div class="sb-rl">SECURITY ANALYST</div>
        <div class="sb-on"><div class="sb-dt"></div><span>Active Session</span></div>
    </div>
    <div class="sb-st">
        <div class="sb-lb">Session Stats</div>
        <div class="sb-rw"><span class="sb-k">TOTAL</span><span class="sb-v" style="color:#00e5ff;">{total}</span></div>
        <div class="sb-rw"><span class="sb-k">SCAMS</span><span class="sb-v" style="color:#ff3a5c;">{scams}</span></div>
        <div class="sb-rw"><span class="sb-k">SUSPICIOUS</span><span class="sb-v" style="color:#ffb020;">{suspic}</span></div>
        <div class="sb-rw"><span class="sb-k">SAFE</span><span class="sb-v" style="color:#00ffa3;">{safe}</span></div>
        <div class="sb-rw"><span class="sb-k">ENGINE</span><span class="sb-v" style="color:{ecol};font-size:0.7rem;">{etxt}</span></div>
    </div>
    <div class="sb-tp">
        <div class="sb-lb">How It Works</div>
        <div class="tp-it"><span>🤖</span><span>SMS &amp; Email ke alag ML models — auto-detect se sahi model use hota hai</span></div>
        <div class="tp-it"><span>🎯</span><span>0–40 Safe · 40–70 Suspicious · 70+ Scam</span></div>
        <div class="tp-it"><span>🔒</span><span>Messages sirf session tak store hote hain</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("🚪  Logout", use_container_width=True):
        st.session_state.logged_in   = False
        st.session_state.history     = []
        st.session_state.username    = ""
        st.session_state.last_result = None
        st.rerun()

# ── TOPBAR ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="tbar">
  <div class="tbar-logo">🔦 FLASHLENS AI</div>
  <div class="tbar-meta">
    {datetime.now().strftime('%d %b %Y')} &nbsp;|&nbsp;
    <span style="color:var(--cyan);">{datetime.now().strftime('%H:%M')}</span>
  </div>
</div>""", unsafe_allow_html=True)

# ── METRIC CARDS ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, val, lbl, cls in zip(
    [c1,c2,c3,c4], [total,scams,suspic,safe],
    ["Total Scanned","Scams Detected","Suspicious","Safe"],
    ["mc","mc mc-r","mc mc-a","mc mc-g"]
):
    with col:
        st.markdown(f'<div class="{cls}"><div class="mc-v">{val}</div><div class="mc-l">{lbl}</div></div>',
                    unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍  Analyze Message", "📊  Dashboard", "📋  History"])

# TAB 1
with tab1:
    cl, cr = st.columns([1,1], gap="large")
    with cl:
        st.markdown('<div class="sec-t">📩 Message Input</div>', unsafe_allow_html=True)
        msg = st.text_area("PASTE MESSAGE",
            placeholder="SMS ya Email paste karein...\n\nSMS: Congratulations! You won ₹10 Lakh! Click: bit.ly/xyz\n\nEmail: Subject: KYC Update\nDear Customer, your account will be suspended...",
            height=210, key="mi")

        if msg:
            mtype = auto_detect(msg)
            dc  = "#a259ff" if mtype=="email" else "#00e5ff"
            ico = "📧" if mtype=="email" else "💬"
            st.markdown(f"""<div class="dpill">
                <span class="ddot" style="background:{dc};box-shadow:0 0 8px {dc};"></span>
                {ico} AUTO-DETECTED:
                <span style="color:{dc};font-weight:700;letter-spacing:2px;">{mtype.upper()}</span>
                &nbsp;→&nbsp;<span style="color:#3d5478;">Using {mtype.upper()} Model</span>
            </div>""", unsafe_allow_html=True)
        else:
            mtype = "sms"

        ov = st.radio("🔧 Manual Override", ["Auto Detect","Force SMS","Force Email"],
                      horizontal=True, key="ov")
        if ov == "Force SMS":    mtype = "sms"
        elif ov == "Force Email": mtype = "email"

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        btn = st.button("⚡  ANALYZE MESSAGE", use_container_width=True, key="ab")

    with cr:
        st.markdown('<div class="sec-t">🎯 Detection Result</div>', unsafe_allow_html=True)
        if btn and msg:
            with st.spinner("FLASHLENS AI analyzing..."):
                time.sleep(0.5)
                r = analyze(msg, mtype)
                st.session_state.history.insert(0, r)
                st.session_state.last_result = r
                st.rerun()

        r = st.session_state.last_result
        if r:
            st.plotly_chart(gauge_chart(r['score'], r['color']), use_container_width=True, key="gm")
            if r['level'] == 'SCAM':
                st.markdown(f"""<div class="a-sc">
                    <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.4rem;color:#ff3a5c;">🚨 SCAM DETECTED</div>
                    <div style="font-size:0.87rem;color:#ffaaaa;margin-top:7px;line-height:1.6;"><b>DO NOT</b> click links or share personal info.</div>
                    <div style="margin-top:13px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
                        <span class="bdg b-r">HIGH RISK · {r['score']}%</span>
                        <span class="bdg b-c">{r['type']}</span>
                        <span style="color:#3d5478;font-size:0.62rem;font-family:'JetBrains Mono',monospace;">{r['timestamp']}</span>
                    </div></div>""", unsafe_allow_html=True)
            elif r['level'] == 'SUSPICIOUS':
                st.markdown(f"""<div class="a-su">
                    <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.4rem;color:#ffb020;">⚠️ SUSPICIOUS</div>
                    <div style="font-size:0.87rem;color:#ffd08a;margin-top:7px;line-height:1.6;">Official channels se verify karen before responding.</div>
                    <div style="margin-top:13px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
                        <span class="bdg b-a">MEDIUM RISK · {r['score']}%</span>
                        <span class="bdg b-c">{r['type']}</span>
                        <span style="color:#3d5478;font-size:0.62rem;font-family:'JetBrains Mono',monospace;">{r['timestamp']}</span>
                    </div></div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="a-sf">
                    <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.4rem;color:#00ffa3;">✅ SAFE</div>
                    <div style="font-size:0.87rem;color:#9effd8;margin-top:7px;line-height:1.6;">No significant scam indicators found.</div>
                    <div style="margin-top:13px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
                        <span class="bdg b-g">LOW RISK · {r['score']}%</span>
                        <span class="bdg b-c">{r['type']}</span>
                        <span style="color:#3d5478;font-size:0.62rem;font-family:'JetBrains Mono',monospace;">{r['timestamp']}</span>
                    </div></div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="sec-t" style="font-size:0.7rem;">🔎 Detection Flags</div>', unsafe_allow_html=True)
            for f in r['flags']:
                st.markdown(f'<div class="flg">{f}</div>', unsafe_allow_html=True)
        else:
            st.markdown("""<div class="awt"><span class="awt-i">🔍</span>
                <div class="awt-t">AWAITING INPUT</div>
                <div class="awt-s">Paste a message and click Analyze</div></div>""", unsafe_allow_html=True)

# TAB 2
with tab2:
    if not st.session_state.history:
        st.markdown("""<div class="awt" style="margin-top:30px;"><span class="awt-i">📊</span>
            <div class="awt-t">NO DATA YET</div>
            <div class="awt-s">Analyze messages to see charts</div></div>""", unsafe_allow_html=True)
    else:
        ca, cb = st.columns([3,2])
        with ca:
            st.markdown('<div class="sec-t">📈 Risk Score Timeline</div>', unsafe_allow_html=True)
            f = timeline_chart(st.session_state.history[-25:])
            if f: st.plotly_chart(f, use_container_width=True, key="tl")
        with cb:
            st.markdown('<div class="sec-t">🍩 Risk Breakdown</div>', unsafe_allow_html=True)
            f = donut_chart(st.session_state.history)
            if f: st.plotly_chart(f, use_container_width=True, key="dn")

        cb1, cb2 = st.columns([2,1])
        with cb1:
            st.markdown('<div class="sec-t">📊 Score Per Scan</div>', unsafe_allow_html=True)
            f = bar_chart(st.session_state.history)
            if f: st.plotly_chart(f, use_container_width=True, key="br")
        with cb2:
            st.markdown('<div class="sec-t">📌 Quick Stats</div>', unsafe_allow_html=True)
            sc   = [h['score'] for h in st.session_state.history]
            sm_c = sum(1 for h in st.session_state.history if h['type']=='SMS')
            em_c = sum(1 for h in st.session_state.history if h['type']=='EMAIL')
            for lb, vl, cl in [
                ("Avg Score", f"{int(np.mean(sc))}%", "#00e5ff"),
                ("Highest",   f"{max(sc)}%",           "#ff3a5c"),
                ("Lowest",    f"{min(sc)}%",            "#00ffa3"),
                ("SMS",       sm_c,                     "#a259ff"),
                ("Emails",    em_c,                     "#a259ff"),
            ]:
                st.markdown(f"""<div style='display:flex;justify-content:space-between;align-items:center;
                padding:9px 14px;margin-bottom:7px;background:rgba(11,17,32,0.6);
                border:1px solid rgba(31,45,69,0.4);border-radius:10px;'>
                <span style='font-size:0.67rem;color:#3d5478;font-family:JetBrains Mono,monospace;'>{lb}</span>
                <span style='font-family:Syne,sans-serif;font-weight:800;font-size:1.2rem;color:{cl};'>{vl}</span>
                </div>""", unsafe_allow_html=True)

# TAB 3
with tab3:
    st.markdown('<div class="sec-t">📋 Scan History</div>', unsafe_allow_html=True)
    if st.session_state.history:
        f1, f2, f3 = st.columns([2,2,1])
        with f1: fl = st.selectbox("Filter by Level", ["ALL","SCAM","SUSPICIOUS","SAFE"], key="hf")
        with f2: ft = st.selectbox("Filter by Type",  ["ALL","SMS","EMAIL"], key="tf")
        with f3:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.history     = []
                st.session_state.last_result = None
                st.rerun()

        filtered = [h for h in st.session_state.history
                    if (fl=="ALL" or h['level']==fl) and (ft=="ALL" or h['type']==ft)]

        for item in filtered[:50]:
            c  = {"SCAM":"#ff3a5c","SUSPICIOUS":"#ffb020","SAFE":"#00ffa3"}.get(item['level'],'#00e5ff')
            ic = {"SCAM":"🚨","SUSPICIOUS":"⚠️","SAFE":"✅"}.get(item['level'],'❓')
            tc = "#a259ff" if item['type']=="EMAIL" else "#00e5ff"
            st.markdown(f"""<div class="hit" style="border-left:3px solid {c};">
                <div style='flex:1;'>
                    <span style='color:{c};font-weight:700;font-size:0.87rem;font-family:Syne,sans-serif;'>{ic} {item['level']}</span>
                    <span style='color:{tc};font-size:0.62rem;margin-left:9px;font-family:JetBrains Mono,monospace;
                    background:rgba(0,0,0,0.3);padding:2px 8px;border-radius:10px;border:1px solid {tc}33;'>{item['type']}</span>
                    <div style='color:#4a6070;font-size:0.81rem;margin-top:5px;'>{item['preview']}</div>
                </div>
                <div style='text-align:right;min-width:72px;margin-left:14px;'>
                    <div style='font-family:Syne,sans-serif;font-weight:800;font-size:1.5rem;line-height:1;color:{c};'>{item['score']}%</div>
                    <div style='color:#3d5478;font-size:0.6rem;font-family:JetBrains Mono,monospace;margin-top:3px;'>{item['timestamp']}</div>
                </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="awt"><span class="awt-i">📭</span>
            <div class="awt-t">NO HISTORY YET</div>
            <div class="awt-s">Pehle kuch messages analyze karein</div></div>""", unsafe_allow_html=True)

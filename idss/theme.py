"""Visual theme: CSS, KPI cards and shared Plotly styling."""
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

NAVY, BLUE, TEAL, AMBER, RED, GREY = "#0B2545", "#1D6FD8", "#13A89E", "#E8A317", "#D64545", "#6B7A90"
PALETTE = [BLUE, TEAL, AMBER, RED, "#7A5AF8", "#5B8DB8", "#9AA5B8", "#2E4A7A"]
TIER_COLORS = {1: RED, 2: AMBER, 3: TEAL}

pio.templates["idss"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, Segoe UI, Roboto, sans-serif", size=13, color="#1F2A3C"),
    colorway=PALETTE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=44, b=10), title=dict(font=dict(size=15, color=NAVY), x=0.0),
    xaxis=dict(gridcolor="#E6EAF0", zerolinecolor="#E6EAF0"), yaxis=dict(gridcolor="#E6EAF0", zerolinecolor="#E6EAF0"),
    legend=dict(orientation="h", y=-0.18)))
pio.templates.default = "idss"

CSS = f"""
<style>
html, body, [class*="css"] {{ font-family: Inter, 'Segoe UI', Roboto, sans-serif; }}
#MainMenu, footer {{ visibility: hidden; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2rem; max-width: 1280px; }}
h1 {{ color: {NAVY}; font-weight: 700; letter-spacing: -0.02em; font-size: 1.9rem !important; }}
h2, h3 {{ color: {NAVY}; font-weight: 650; letter-spacing: -0.01em; }}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, {NAVY} 0%, #12335F 100%); }}
[data-testid="stSidebar"] * {{ color: #DCE6F5; }}
[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] {{ background: rgba(255,255,255,0.14); }}
[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.15); }}
[data-testid="stSidebar"] input, [data-testid="stSidebar"] [data-baseweb="select"] * {{ color: #1F2A3C !important; }}
.brand {{ font-size: 1.15rem; font-weight: 700; color: #fff; letter-spacing: .02em; padding: .2rem 0 .1rem; }}
.brand-sub {{ font-size: .78rem; color: #9FB3D1; margin-bottom: .6rem; }}
.page-sub {{ color: {GREY}; font-size: .95rem; margin: -0.6rem 0 1.2rem; }}
.kpi {{ background: #fff; border: 1px solid #E3E8F0; border-radius: 12px; padding: 16px 18px; box-shadow: 0 1px 2px rgba(11,37,69,.04); height: 100%; }}
.kpi .l {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .07em; color: {GREY}; font-weight: 600; }}
.kpi .v {{ font-size: 1.7rem; font-weight: 700; color: {NAVY}; line-height: 1.25; }}
.kpi .s {{ font-size: .8rem; color: {GREY}; }}
.card {{ background: #fff; border: 1px solid #E3E8F0; border-radius: 12px; padding: 18px 22px; margin-bottom: 14px; }}
.card h4 {{ margin: 0 0 .5rem; color: {NAVY}; font-size: 1rem; }}
.card p, .card li {{ color: #26334A; line-height: 1.6; font-size: .95rem; }}
.verdict {{ border-radius: 12px; padding: 20px 24px; color: #fff; margin-bottom: 16px; }}
.verdict .t {{ font-size: .72rem; letter-spacing: .09em; text-transform: uppercase; opacity: .85; }}
.verdict .h {{ font-size: 1.25rem; font-weight: 650; line-height: 1.4; }}
.chip {{ display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: .78rem; font-weight: 600; margin-right: 6px; }}
.chip.done {{ background: #DDF4EF; color: #0B6F66; }} .chip.todo {{ background: #EEF1F6; color: {GREY}; }}
.note {{ background: #F1F6FD; border-left: 3px solid {BLUE}; padding: 10px 14px; border-radius: 6px; font-size: .88rem; color: #26334A; margin: 8px 0 14px; }}
.warnbox {{ background: #FFF6E5; border-left: 3px solid {AMBER}; padding: 10px 14px; border-radius: 6px; font-size: .88rem; color: #4A3A12; margin: 8px 0 14px; }}
div[data-testid="stTabs"] button {{ font-weight: 600; }}
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def header(title: str, sub: str = ""):
    st.title(title)
    if sub:
        st.markdown(f'<div class="page-sub">{sub}</div>', unsafe_allow_html=True)


def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>', unsafe_allow_html=True)


def note(text, warn=False):
    st.markdown(f'<div class="{"warnbox" if warn else "note"}">{text}</div>', unsafe_allow_html=True)


def card(title, lines):
    body = "".join(f"<li>{l}</li>" for l in lines)
    st.markdown(f'<div class="card"><h4>{title}</h4><ul style="margin:0;padding-left:1.1rem">{body}</ul></div>', unsafe_allow_html=True)


def chips(items):
    st.markdown("".join(f'<span class="chip {"done" if ok else "todo"}">{n}</span>' for n, ok in items), unsafe_allow_html=True)

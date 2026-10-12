"""
EvidenceMeet — Centralized CSS Theme
Defines Dark and Light themes with design tokens, glassmorphism, responsive
typography, and component styles. Injected via inject_theme().
"""

def get_theme_css(theme: str = "dark") -> str:
    """Generate complete CSS for the requested theme ('dark' or 'light')."""
    is_light = (theme.lower() == "light")

    if is_light:
        tokens = """
  --bg-app:          #f8fafc;
  --bg-sidebar:      linear-gradient(180deg, #f1f5f9 0%, #ffffff 100%);
  --bg-card:         rgba(255, 255, 255, 0.94);
  --bg-card-hover:   #ffffff;
  --bg-input:        #ffffff;
  --bg-header:       linear-gradient(135deg, #ffffff 0%, rgba(124, 58, 237, 0.08) 100%);
  --border-glass:    rgba(124, 58, 237, 0.18);
  --border-focus:    #7c3aed;
  --border-subtle:   #e2e8f0;
  --text-primary:    #0f172a;
  --text-secondary:  #334155;
  --text-muted:      #64748b;
  --accent-violet:   #7c3aed;
  --accent-violet-light: #6d28d9;
  --accent-blue:     #2563eb;
  --accent-cyan:     #0891b2;
  --accent-emerald:  #059669;
  --accent-amber:    #d97706;
  --accent-rose:     #dc2626;
  --badge-ver-bg:    rgba(16, 185, 129, 0.12);
  --badge-ver-bd:    rgba(16, 185, 129, 0.45);
  --badge-ver-tx:    #047857;
  --badge-par-bg:    rgba(245, 158, 11, 0.12);
  --badge-par-bd:    rgba(245, 158, 11, 0.45);
  --badge-par-tx:    #b45309;
  --badge-unv-bg:    rgba(239, 68, 68, 0.12);
  --badge-unv-bd:    rgba(239, 68, 68, 0.45);
  --badge-unv-tx:    #b91c1c;
  --audio-filter:    none;
  --tbl-th-bg:       #ede9fe;
  --tbl-th-tx:       #5b21b6;
  --tbl-row-hover:   rgba(124, 58, 237, 0.05);
  --tbl-border:      #e2e8f0;
  --shadow-card:     0 4px 20px rgba(0, 0, 0, 0.06);
  --shadow-glow:     0 0 24px rgba(124, 58, 237, 0.14);
  --code-bg:         rgba(124, 58, 237, 0.08);
  --code-tx:         #6d28d9;
  --quote-bg:        rgba(124, 58, 237, 0.04);
  --title-gradient:  linear-gradient(135deg, #0f172a 0%, #6d28d9 60%, #2563eb 100%);
  --btn-proc-bg:     linear-gradient(135deg, #10b981 0%, #34d399 50%, #6ee7b7 100%);
  --btn-proc-hover:  linear-gradient(135deg, #059669 0%, #10b981 50%, #34d399 100%);
  --btn-proc-tx:     #064e3b;
  --btn-proc-bd:     rgba(16, 185, 129, 0.45);
  --btn-proc-shadow: 0 3px 12px rgba(16, 185, 129, 0.20);
  --btn-proc-shadow-hov: 0 5px 16px rgba(16, 185, 129, 0.30);
  --tooltip-bg:      rgba(15, 23, 42, 0.95);
  --tooltip-tx:      #f8fafc;
  --tooltip-bd:      rgba(16, 185, 129, 0.40);
"""
    else:
        tokens = """
  --bg-app:          #060b18;
  --bg-sidebar:      linear-gradient(180deg, #0a1128 0%, #060b18 100%);
  --bg-card:         rgba(15, 26, 58, 0.75);
  --bg-card-hover:   rgba(22, 36, 75, 0.85);
  --bg-input:        rgba(255, 255, 255, 0.05);
  --bg-header:       linear-gradient(135deg, #0f1a3a 0%, rgba(124, 58, 237, 0.12) 100%);
  --border-glass:    rgba(139, 92, 246, 0.20);
  --border-focus:    #8b5cf6;
  --border-subtle:   rgba(255, 255, 255, 0.05);
  --text-primary:    #f0f4ff;
  --text-secondary:  #94a3c0;
  --text-muted:      #64748b;
  --accent-violet:   #8b5cf6;
  --accent-violet-light: #a78bfa;
  --accent-blue:     #60a5fa;
  --accent-cyan:     #22d3ee;
  --accent-emerald:  #34d399;
  --accent-amber:    #fbbf24;
  --accent-rose:     #f87171;
  --badge-ver-bg:    rgba(52, 211, 153, 0.15);
  --badge-ver-bd:    rgba(52, 211, 153, 0.40);
  --badge-ver-tx:    #34d399;
  --badge-par-bg:    rgba(251, 191, 36, 0.15);
  --badge-par-bd:    rgba(251, 191, 36, 0.40);
  --badge-par-tx:    #fbbf24;
  --badge-unv-bg:    rgba(248, 113, 113, 0.15);
  --badge-unv-bd:    rgba(248, 113, 113, 0.40);
  --badge-unv-tx:    #f87171;
  --audio-filter:    invert(0.85) hue-rotate(200deg);
  --tbl-th-bg:       rgba(124, 58, 237, 0.25);
  --tbl-th-tx:       #a78bfa;
  --tbl-row-hover:   rgba(139, 92, 246, 0.08);
  --tbl-border:      rgba(255, 255, 255, 0.05);
  --shadow-card:     0 4px 24px rgba(0, 0, 0, 0.45);
  --shadow-glow:     0 0 32px rgba(124, 58, 237, 0.18);
  --code-bg:         rgba(139, 92, 246, 0.18);
  --code-tx:         #a78bfa;
  --quote-bg:        rgba(124, 58, 237, 0.08);
  --title-gradient:  linear-gradient(135deg, #f0f4ff 0%, #a78bfa 60%, #93c5fd 100%);
  --btn-proc-bg:     linear-gradient(135deg, #059669 0%, #10b981 40%, #34d399 100%);
  --btn-proc-hover:  linear-gradient(135deg, #10b981 0%, #34d399 50%, #6ee7b7 100%);
  --btn-proc-tx:     #022c22;
  --btn-proc-bd:     rgba(52, 211, 153, 0.45);
  --btn-proc-shadow: 0 3px 14px rgba(16, 185, 129, 0.25);
  --btn-proc-shadow-hov: 0 6px 20px rgba(16, 185, 129, 0.35);
  --tooltip-bg:      rgba(10, 16, 34, 0.96);
  --tooltip-tx:      #e2e8f0;
  --tooltip-bd:      rgba(52, 211, 153, 0.35);
"""

    return f"""
<style>
/* ── Google Fonts ─────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Design Tokens ────────────────────────────────────── */
:root {{
{tokens}
  --radius-sm: 8px;
  --radius-md: 14px;
  --radius-lg: 20px;
  --radius-xl: 28px;
  --transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1);
}}

/* ── Global Reset & Base ──────────────────────────────── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"] {{
  background: var(--bg-app) !important;
  color: var(--text-primary) !important;
  font-family: 'Inter', sans-serif !important;
}}

[data-testid="stMain"] {{
  background: transparent !important;
}}

/* Hide default streamlit header / footer branding */
#MainMenu, [data-testid="stHeader"], footer {{ display: none !important; }}

/* ── Scrollbar ────────────────────────────────────────── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: var(--bg-app); }}
::-webkit-scrollbar-thumb {{ background: var(--accent-violet); border-radius: 999px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--accent-violet-light); }}

/* ── Sidebar ──────────────────────────────────────────── */
[data-testid="stSidebar"] {{
  background: var(--bg-sidebar) !important;
  border-right: 1px solid var(--border-glass) !important;
}}
[data-testid="stSidebar"] * {{
  color: var(--text-primary) !important;
}}
[data-testid="stSidebar"] .stTextInput input {{
  background: var(--bg-input) !important;
  border: 1px solid var(--border-glass) !important;
  color: var(--text-primary) !important;
  border-radius: var(--radius-sm) !important;
}}
[data-testid="stSidebar"] .stTextInput input:focus {{
  border-color: var(--border-focus) !important;
  box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.20) !important;
}}
[data-testid="stSidebar"] hr {{ border-color: var(--border-glass) !important; }}

/* ── File Uploader ────────────────────────────────────── */
[data-testid="stFileUploader"] {{
  background: var(--bg-card) !important;
  border: 2px dashed var(--border-glass) !important;
  border-radius: var(--radius-lg) !important;
  padding: 14px !important;
  transition: var(--transition) !important;
  backdrop-filter: blur(12px) !important;
  box-shadow: var(--shadow-card) !important;
}}
[data-testid="stFileUploader"]:hover {{
  border-color: var(--border-focus) !important;
  box-shadow: var(--shadow-glow) !important;
}}
[data-testid="stFileUploader"] * {{ color: var(--text-secondary) !important; }}

/* ── Primary Process Button (Soft Mint / Light-Green Gradient) ── */
button[data-testid="stBaseButton-primary"],
div[data-testid="stButton"] button[kind="primary"],
.stButton button[kind="primary"],
div[data-testid="stTooltipHoverTarget"] button[kind="primary"],
div[data-testid="stTooltipHoverTarget"] button,
button[data-testid="baseButton-primary"] {{
  background: var(--btn-proc-bg) !important;
  color: var(--btn-proc-tx) !important;
  border: 1px solid var(--btn-proc-bd) !important;
  border-radius: 12px !important;
  font-family: 'Inter', -apple-system, sans-serif !important;
  font-weight: 700 !important;
  font-size: 0.95rem !important;
  padding: 0.65rem 1.4rem !important;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
  box-shadow: var(--btn-proc-shadow) !important;
  letter-spacing: 0.015em !important;
  cursor: pointer !important;
}}

/* Ensure inner label text and icons have readable high-contrast dark green color */
button[data-testid="stBaseButton-primary"] *,
div[data-testid="stButton"] button[kind="primary"] *,
.stButton button[kind="primary"] *,
div[data-testid="stTooltipHoverTarget"] button[kind="primary"] *,
div[data-testid="stTooltipHoverTarget"] button *,
button[data-testid="baseButton-primary"] * {{
  color: var(--btn-proc-tx) !important;
  font-weight: 700 !important;
}}

/* Restrained Hover transition */
button[data-testid="stBaseButton-primary"]:hover,
div[data-testid="stButton"] button[kind="primary"]:hover,
.stButton button[kind="primary"]:hover,
div[data-testid="stTooltipHoverTarget"] button[kind="primary"]:hover,
div[data-testid="stTooltipHoverTarget"] button:hover,
button[data-testid="baseButton-primary"]:hover {{
  background: var(--btn-proc-hover) !important;
  transform: translateY(-1px) !important;
  box-shadow: var(--btn-proc-shadow-hov) !important;
  border-color: rgba(52, 211, 153, 0.65) !important;
}}

button[data-testid="stBaseButton-primary"]:hover *,
div[data-testid="stButton"] button[kind="primary"]:hover *,
div[data-testid="stTooltipHoverTarget"] button:hover * {{
  color: var(--btn-proc-tx) !important;
}}

/* Active / Click */
button[data-testid="stBaseButton-primary"]:active,
div[data-testid="stButton"] button[kind="primary"]:active,
.stButton button[kind="primary"]:active,
div[data-testid="stTooltipHoverTarget"] button:active,
button[data-testid="baseButton-primary"]:active {{
  transform: translateY(0) !important;
  box-shadow: 0 2px 8px rgba(16, 185, 129, 0.18) !important;
}}

/* Restrained Focus / Focus-Visible */
button[data-testid="stBaseButton-primary"]:focus-visible,
div[data-testid="stButton"] button[kind="primary"]:focus-visible,
.stButton button[kind="primary"]:focus-visible,
div[data-testid="stTooltipHoverTarget"] button:focus-visible,
button[data-testid="baseButton-primary"]:focus-visible {{
  outline: 2px solid #34d399 !important;
  outline-offset: 2px !important;
  box-shadow: 0 0 0 3px rgba(52, 211, 153, 0.25) !important;
}}

/* ── Tooltip Component (Subtle, Compact, Non-intrusive) ─ */
div[data-testid="stTooltipContent"],
div[data-testid="stTooltipContent"] > div,
div[data-baseweb="tooltip"],
div[data-baseweb="tooltip"] > div,
[data-baseweb="popover"] div[role="tooltip"] {{
  background: var(--tooltip-bg) !important;
  color: var(--tooltip-tx) !important;
  border: 1px solid var(--tooltip-bd) !important;
  border-radius: 8px !important;
  font-size: 0.74rem !important;
  line-height: 1.35 !important;
  padding: 5px 10px !important;
  max-width: 290px !important;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.40) !important;
  backdrop-filter: blur(12px) !important;
  -webkit-backdrop-filter: blur(12px) !important;
  pointer-events: none !important;
  letter-spacing: 0.01em !important;
  z-index: 99999 !important;
}}

div[data-testid="stTooltipContent"] p,
div[data-baseweb="tooltip"] p,
[data-baseweb="popover"] div[role="tooltip"] p {{
  font-size: 0.74rem !important;
  line-height: 1.35 !important;
  color: var(--tooltip-tx) !important;
  margin: 0 !important;
  padding: 0 !important;
}}

/* Immediately hide tooltips when button is clicked / active */
body:has(button[data-testid="stBaseButton-primary"]:active) div[data-testid="stTooltipContent"],
body:has(button[data-testid="stBaseButton-primary"]:active) div[data-baseweb="tooltip"],
body:has(button[data-testid="stBaseButton-primary"]:active) [data-baseweb="popover"],
body:has(button[kind="primary"]:active) div[data-testid="stTooltipContent"],
body:has(button[kind="primary"]:active) div[data-baseweb="tooltip"],
body:has(button[kind="primary"]:active) [data-baseweb="popover"] {{
  display: none !important;
  opacity: 0 !important;
  visibility: hidden !important;
}}

/* ── Standard / Secondary Buttons ─────────────────────── */
.stButton > button {{
  background: linear-gradient(135deg, var(--accent-violet) 0%, #4f46e5 100%) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: var(--radius-md) !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
  padding: 0.55rem 1.2rem !important;
  transition: var(--transition) !important;
  box-shadow: 0 4px 14px rgba(124, 58, 237, 0.30) !important;
}}
.stButton > button:hover {{
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 20px rgba(124, 58, 237, 0.45) !important;
  filter: brightness(1.08) !important;
}}
.stButton > button:active {{
  transform: translateY(0) !important;
  filter: brightness(0.95) !important;
}}

/* Jump / Clear / Return action buttons */
.stButton > button[kind="secondary"],
button[data-testid*="clear"],
button[data-testid*="jump"],
button[data-testid*="return"] {{
  background: var(--code-bg) !important;
  border: 1px solid var(--border-glass) !important;
  color: var(--code-tx) !important;
  box-shadow: none !important;
  font-weight: 600 !important;
}}
.stButton > button[kind="secondary"]:hover,
button[data-testid*="clear"]:hover,
button[data-testid*="jump"]:hover,
button[data-testid*="return"]:hover {{
  background: rgba(124, 58, 237, 0.20) !important;
  border-color: var(--border-focus) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 12px rgba(124, 58, 237, 0.20) !important;
}}

/* Download buttons */
[data-testid="stDownloadButton"] > button {{
  background: rgba(34, 211, 238, 0.12) !important;
  border: 1px solid rgba(34, 211, 238, 0.35) !important;
  color: var(--accent-cyan) !important;
  font-weight: 600 !important;
  box-shadow: none !important;
}}
[data-testid="stDownloadButton"] > button:hover {{
  background: rgba(34, 211, 238, 0.22) !important;
  border-color: var(--accent-cyan) !important;
  box-shadow: 0 4px 14px rgba(34, 211, 238, 0.25) !important;
  transform: translateY(-1px) !important;
}}

/* ── Tabs ─────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {{
  background: var(--bg-card) !important;
  border-radius: var(--radius-md) !important;
  border: 1px solid var(--border-glass) !important;
  padding: 4px !important;
  gap: 3px !important;
  backdrop-filter: blur(12px) !important;
  box-shadow: var(--shadow-card) !important;
}}
[data-testid="stTabs"] [role="tab"] {{
  background: transparent !important;
  color: var(--text-secondary) !important;
  border-radius: var(--radius-sm) !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 500 !important;
  font-size: 0.84rem !important;
  padding: 0.5rem 0.95rem !important;
  border: none !important;
  transition: var(--transition) !important;
}}
[data-testid="stTabs"] [role="tab"]:hover {{
  color: var(--accent-violet) !important;
  background: rgba(124, 58, 237, 0.08) !important;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
  background: linear-gradient(135deg, var(--accent-violet), #4f46e5) !important;
  color: #ffffff !important;
  font-weight: 700 !important;
  box-shadow: 0 2px 10px rgba(124, 58, 237, 0.40) !important;
}}

/* ── Metrics ──────────────────────────────────────────── */
[data-testid="stMetric"] {{
  background: var(--bg-card) !important;
  border: 1px solid var(--border-glass) !important;
  border-radius: var(--radius-md) !important;
  padding: 16px 20px !important;
  backdrop-filter: blur(12px) !important;
  transition: var(--transition) !important;
  box-shadow: var(--shadow-card) !important;
}}
[data-testid="stMetric"]:hover {{
  border-color: var(--border-focus) !important;
  box-shadow: var(--shadow-glow) !important;
  transform: translateY(-2px) !important;
}}
[data-testid="stMetricLabel"] {{
  color: var(--text-secondary) !important;
  font-size: 0.78rem !important;
  font-weight: 600 !important;
  text-transform: uppercase;
  letter-spacing: 0.06em !important;
}}
[data-testid="stMetricValue"] {{
  color: var(--text-primary) !important;
  font-size: 1.55rem !important;
  font-weight: 700 !important;
  font-family: 'JetBrains Mono', monospace !important;
}}

/* ── Expanders (Collapsible sections) ─────────────────── */
[data-testid="stExpander"] {{
  background: var(--bg-card) !important;
  border: 1px solid var(--border-glass) !important;
  border-radius: var(--radius-md) !important;
  backdrop-filter: blur(12px) !important;
  margin-bottom: 12px !important;
  overflow: hidden !important;
  transition: var(--transition) !important;
  box-shadow: var(--shadow-card) !important;
}}
[data-testid="stExpander"]:hover {{
  border-color: var(--border-focus) !important;
}}
[data-testid="stExpander"] summary {{
  color: var(--text-primary) !important;
  font-weight: 600 !important;
  padding: 14px 18px !important;
  font-size: 0.95rem !important;
}}
[data-testid="stExpander"] summary:hover {{
  color: var(--accent-violet) !important;
}}
[data-testid="stExpander"] summary svg {{
  fill: var(--accent-violet) !important;
}}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
  padding: 14px 18px !important;
  border-top: 1px solid var(--border-glass) !important;
}}

/* ── Alerts ───────────────────────────────────────────── */
[data-testid="stAlert"] {{
  border-radius: var(--radius-md) !important;
  border-left-width: 4px !important;
  backdrop-filter: blur(8px) !important;
  font-size: 0.88rem !important;
}}

/* ── Text Inputs & Textareas ──────────────────────────── */
.stTextInput input, .stTextArea textarea {{
  background: var(--bg-input) !important;
  border: 1px solid var(--border-glass) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.88rem !important;
  transition: var(--transition) !important;
}}
.stTextInput input:focus, .stTextArea textarea:focus {{
  border-color: var(--border-focus) !important;
  box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.18) !important;
  outline: none !important;
}}
.stTextInput label, .stTextArea label {{
  color: var(--text-secondary) !important;
  font-size: 0.82rem !important;
}}

/* ── Radio & Toggles (Theme Switcher) ─────────────────── */
[data-testid="stRadio"] [role="radiogroup"] {{
  gap: 8px !important;
}}
[data-testid="stRadio"] label {{
  color: var(--text-primary) !important;
  font-size: 0.85rem !important;
  font-weight: 600 !important;
}}

/* ── Audio Player ─────────────────────────────────────── */
audio {{
  width: 100% !important;
  border-radius: var(--radius-sm) !important;
  outline: none !important;
  filter: var(--audio-filter) !important;
}}

/* ── Status & Progress ────────────────────────────────── */
[data-testid="stStatus"] {{
  background: var(--bg-card) !important;
  border: 1px solid var(--border-glass) !important;
  border-radius: var(--radius-md) !important;
  backdrop-filter: blur(12px) !important;
}}
[data-testid="stStatus"] * {{ color: var(--text-primary) !important; }}

/* ── Chat Messages (Smart Transcript) ────────────────── */
[data-testid="stChatMessage"] {{
  background: var(--bg-card) !important;
  border: 1px solid var(--border-glass) !important;
  border-radius: var(--radius-md) !important;
  padding: 12px 16px !important;
  margin-bottom: 8px !important;
  backdrop-filter: blur(8px) !important;
  transition: var(--transition) !important;
  box-shadow: var(--shadow-card) !important;
}}
[data-testid="stChatMessage"]:hover {{
  border-color: var(--border-focus) !important;
  transform: translateX(2px) !important;
}}

/* ── Tables ───────────────────────────────────────────── */
[data-testid="stTable"] table, .stDataFrame table {{
  background: var(--bg-card) !important;
  border-collapse: separate !important;
  border-spacing: 0 !important;
  border-radius: var(--radius-md) !important;
  overflow: hidden !important;
  width: 100% !important;
}}
[data-testid="stTable"] th, .stDataFrame th {{
  background: var(--tbl-th-bg) !important;
  color: var(--tbl-th-tx) !important;
  font-size: 0.78rem !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.06em !important;
  padding: 12px 16px !important;
  border-bottom: 1px solid var(--border-glass) !important;
}}
[data-testid="stTable"] td, .stDataFrame td {{
  color: var(--text-primary) !important;
  padding: 11px 16px !important;
  font-size: 0.86rem !important;
  border-bottom: 1px solid var(--tbl-border) !important;
  font-family: 'JetBrains Mono', monospace !important;
}}
[data-testid="stTable"] tr:last-child td {{ border-bottom: none !important; }}
[data-testid="stTable"] tr:hover td {{ background: var(--tbl-row-hover) !important; }}

/* ── Dividers ─────────────────────────────────────────── */
hr {{ border-color: var(--border-glass) !important; }}

/* ── Markdown Headings & Text ─────────────────────────── */
h1 {{ font-size: 1.95rem !important; font-weight: 800 !important; color: var(--text-primary) !important; }}
h2 {{ font-size: 1.4rem !important; font-weight: 700 !important; color: var(--text-primary) !important; }}
h3 {{ font-size: 1.15rem !important; font-weight: 700 !important; color: var(--accent-violet) !important; }}
h4 {{ font-size: 0.98rem !important; font-weight: 600 !important; color: var(--text-secondary) !important; }}
p, li {{ color: var(--text-secondary) !important; font-size: 0.9rem !important; }}
strong {{ color: var(--text-primary) !important; }}
code {{
  background: var(--code-bg) !important;
  color: var(--code-tx) !important;
  border-radius: 4px !important;
  padding: 2px 6px !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.82em !important;
}}
blockquote {{
  border-left: 3px solid var(--accent-violet) !important;
  background: var(--quote-bg) !important;
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0 !important;
  padding: 10px 16px !important;
  margin: 8px 0 !important;
  color: var(--text-secondary) !important;
}}
blockquote em {{ color: var(--accent-blue) !important; font-style: italic !important; }}

/* ── Caption ──────────────────────────────────────────── */
[data-testid="stCaptionContainer"] {{
  color: var(--text-muted) !important;
  font-size: 0.78rem !important;
}}

/* ── Spinner ──────────────────────────────────────────── */
[data-testid="stSpinner"] > div {{ border-top-color: var(--accent-violet) !important; }}

/* ── Custom Component Classes ─────────────────────────── */

/* Intelligence section header */
.em-section-header {{
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 12px 18px;
  background: var(--code-bg);
  border: 1px solid var(--border-glass);
  border-radius: var(--radius-md);
  margin: 14px 0 10px 0;
  backdrop-filter: blur(8px);
}}
.em-section-icon {{ font-size: 1.25rem; }}
.em-section-title {{
  font-family: 'Inter', sans-serif;
  font-size: 0.98rem;
  font-weight: 700;
  color: var(--accent-violet);
  margin: 0;
}}
.em-section-count {{
  margin-left: auto;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  font-weight: 600;
  color: var(--text-muted);
  background: var(--bg-card);
  border: 1px solid var(--border-glass);
  border-radius: 999px;
  padding: 2px 10px;
}}

/* Intelligence item card */
.em-item-card {{
  background: var(--bg-card);
  border: 1px solid var(--border-glass);
  border-radius: var(--radius-md);
  padding: 16px 20px;
  margin-bottom: 12px;
  backdrop-filter: blur(12px);
  transition: var(--transition);
  position: relative;
  overflow: hidden;
  box-shadow: var(--shadow-card);
}}
.em-item-card::before {{
  content: '';
  position: absolute;
  top: 0; left: 0;
  width: 3px; height: 100%;
  background: linear-gradient(180deg, var(--accent-violet), var(--accent-blue));
  border-radius: 3px 0 0 3px;
}}
.em-item-card:hover {{
  border-color: var(--border-focus);
  box-shadow: var(--shadow-glow);
  transform: translateX(3px);
}}
.em-item-card.em-verified::before   {{ background: linear-gradient(180deg, var(--accent-emerald), #047857); }}
.em-item-card.em-partial::before    {{ background: linear-gradient(180deg, var(--accent-amber), #b45309); }}
.em-item-card.em-unverified::before {{ background: linear-gradient(180deg, var(--accent-rose), #b91c1c); }}

/* Verification badge pill */
.em-badge {{
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-family: 'Inter', sans-serif;
  font-size: 0.72rem;
  font-weight: 700;
  padding: 3px 10px;
  border-radius: 999px;
  letter-spacing: 0.03em;
}}
.em-badge-verified   {{ background: var(--badge-ver-bg); border: 1px solid var(--badge-ver-bd); color: var(--badge-ver-tx); }}
.em-badge-partial    {{ background: var(--badge-par-bg); border: 1px solid var(--badge-par-bd); color: var(--badge-par-tx); }}
.em-badge-unverified {{ background: var(--badge-unv-bg); border: 1px solid var(--badge-unv-bd); color: var(--badge-unv-tx); }}

/* Header branding */
.em-header-brand {{
  background: var(--bg-header);
  border: 1px solid var(--border-glass);
  border-radius: var(--radius-xl);
  padding: 24px 32px;
  margin-bottom: 20px;
  position: relative;
  overflow: hidden;
  backdrop-filter: blur(16px);
  box-shadow: var(--shadow-card);
}}
.em-header-brand::after {{
  content: '';
  position: absolute;
  top: -60px; right: -60px;
  width: 180px; height: 180px;
  background: radial-gradient(circle, rgba(124,58,237,0.15) 0%, transparent 70%);
  pointer-events: none;
}}
.em-header-title {{
  font-family: 'Inter', sans-serif;
  font-size: 1.9rem;
  font-weight: 800;
  background: var(--title-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0 0 4px 0;
  line-height: 1.2;
}}
.em-header-sub {{
  font-family: 'Inter', sans-serif;
  font-size: 0.88rem;
  color: var(--text-muted);
  margin: 0;
  font-weight: 400;
}}

/* Sidebar brand pill */
.em-sidebar-brand {{
  background: var(--code-bg);
  border: 1px solid var(--border-glass);
  border-radius: var(--radius-md);
  padding: 14px 16px;
  margin-bottom: 14px;
  text-align: center;
}}
.em-sidebar-brand-name {{
  font-family: 'Inter', sans-serif;
  font-size: 1.05rem;
  font-weight: 800;
  background: var(--title-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}}
.em-sidebar-brand-tagline {{ font-size: 0.70rem; color: var(--text-muted); }}

/* Module chip */
.em-module-chip {{
  display: inline-block;
  background: var(--code-bg);
  border: 1px solid var(--border-glass);
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 600;
  color: var(--accent-violet);
  padding: 2px 10px;
  margin: 3px 3px 0 0;
  font-family: 'Inter', sans-serif;
}}

/* Team member row */
.em-team-row {{
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: var(--code-bg);
  border-radius: var(--radius-sm);
  margin: 4px 0;
  border: 1px solid var(--border-glass);
  font-size: 0.82rem;
}}

/* Stat pill */
.em-stat-pill {{
  display: inline-flex;
  align-items: center;
  gap: 5px;
  background: var(--code-bg);
  border: 1px solid var(--border-glass);
  border-radius: 999px;
  padding: 5px 14px;
  font-size: 0.78rem;
  font-weight: 600;
  color: var(--text-secondary);
  font-family: 'JetBrains Mono', monospace;
  margin: 3px 3px 3px 0;
}}
.em-stat-pill span {{ color: var(--accent-violet); }}

/* Gradient divider */
.em-gradient-divider {{
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--accent-violet), var(--accent-blue), transparent);
  border: none;
  margin: 18px 0;
  opacity: 0.6;
}}

/* Highlighted transcript segment (pulsing animation) */
.em-highlight-segment {{
  padding: 10px 16px;
  background: linear-gradient(90deg, rgba(251, 191, 36, 0.18), rgba(251, 191, 36, 0.06));
  border-left: 4px solid #fbbf24;
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  margin: 8px 0;
  animation: em-pulse-border 2s ease-in-out infinite;
}}
@keyframes em-pulse-border {{
  0%, 100% {{ border-left-color: #fbbf24; box-shadow: none; }}
  50% {{ border-left-color: #f59e0b; box-shadow: 0 0 14px rgba(251, 191, 36, 0.35); }}
}}

/* Anchor jump link in transcript */
.em-anchor-jump {{
  display: inline-block;
  padding: 4px 10px;
  background: rgba(251, 191, 36, 0.15);
  border: 1px solid rgba(251, 191, 36, 0.35);
  border-radius: var(--radius-sm);
  color: #fbbf24 !important;
  text-decoration: none !important;
  font-size: 0.78rem;
  font-weight: 600;
  margin-top: 4px;
  transition: var(--transition);
}}
.em-anchor-jump:hover {{
  background: rgba(251, 191, 36, 0.25);
  border-color: #fbbf24;
}}

/* Smooth fade-in */
@keyframes em-fadein {{
  from {{ opacity: 0; transform: translateY(6px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}
.em-fadein {{ animation: em-fadein 0.4s ease-out both; }}
</style>
"""


def inject_theme(theme: str = "dark"):
    """Inject the requested theme into the Streamlit app ('dark' or 'light')."""
    import streamlit as st
    css = get_theme_css(theme)
    st.markdown(css, unsafe_allow_html=True)

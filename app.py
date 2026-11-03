from src.data_manager import save_data, load_data, normalize_date_series
import streamlit as st
import re
import math
import pandas as pd
import os
from ui.hero import show_hero
from src.analytics import get_summary
from datetime import datetime
from src.wellness_score import (
    calculate_wellness_score,
    get_score_breakdown,
    _valid_number
)
from ui.about import show_about
from ui.reports import show_reports
from ui.monthly_highlight import show_monthly_highlight
from ui.discovery import show_biggest_discovery
from ui.navigation import show_navigation
from ui.action_plan import show_action_plan
from ui.ai_insight import show_ai_insight, show_trend_analysis
from ui.stat_card import show_stat_card
from ui.forecast import show_forecast
from ui.ai_insight import show_ai_insight
from ui.best_action import show_best_action
from ui.weekly import show_weekly_snapshot
from ui.correlation_card import show_correlation_card
from ui.monthly_score import show_monthly_score
from src.insights import (
    generate_insights,
    generate_trend_analysis,
    generate_correlation_insights,
    generate_relationship_explanation
)
from src.charts import (
    plot_sleep_trend,
    plot_mood_trend,
    plot_stress_trend,
    plot_screen_time_trend,
    create_wellness_breakdown_chart,
    create_wellness_radar_chart,
    plot_correlation_heatmap
)
from src.recommendations import generate_recommendations
from src.pattern_analysis import analyze_patterns
from src.weekly_report import (
    generate_weekly_report,
    calculate_weekly_facts
)
from src.correlation import get_correlation_matrix
from src.monthly_report import generate_monthly_report
from src.forecast import (
    predict_next_day,
    generate_forecast_summary,
    generate_forecast_explanation,
    generate_ai_forecast_insight
)
from ml.predictor import predict_next_day_ml, ml_available
from src.intelligence import (
    get_biggest_discovery,
    get_todays_focus
)
from ml.nlp.reporter import (
    generate_wellness_report,
    explain_weekly_facts,
    explain_tomorrow_forecast
)
from src.prediction_tracker import log_prediction
from src.data_manager import check_data_file

@st.dialog("Check-in already exists")
def confirm_overwrite(new_entry, date):
    st.write(
        f"A check-in for **{date}** already exists."
    )

    st.write(
        "Do you want to replace the existing check-in "
        "with this new data?"
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Cancel", use_container_width=True):
            st.session_state["pending_overwrite"] = None
            st.rerun()

    with col2:
        if st.button("Yes, Overwrite", use_container_width=True):
            try:
                save_data(new_entry)
                st.session_state["pending_overwrite"] = None
                st.session_state["checkin_success_message"] = "Check-in overwritten successfully."
                st.rerun()
            except Exception as e:
                st.error(f"Failed to overwrite check-in: {e}")

check_data_file()
st.set_page_config(
    page_title="LifeLens",
    page_icon="🌱",
    layout="wide"
)
import os

if "lifelens_theme" not in st.session_state:
    st.session_state["lifelens_theme"] = "dark"

active_theme = st.session_state.get("lifelens_theme", "dark")


def load_css(theme="dark"):
    with open("assets/style.css", "r", encoding="utf-8") as f:
        css = f.read()

    if theme == "light":
        theme_override = """
        :root, .stApp {
            --ll-bg: #F7F5FC !important;
            --ll-surface: #FFFFFF !important;
            --ll-surface-2: #F0ECF8 !important;
            --ll-surface-glass: rgba(255, 255, 255, 0.90) !important;
            --ll-border: #E8E1F4 !important;
            --ll-border-strong: #D5C8EA !important;
            --ll-text: #29243A !important;
            --ll-muted: #77718C !important;
            --ll-faint: #A59FB8 !important;
            --ll-purple: #7950C8 !important;
            --ll-violet: #7950C8 !important;
            --ll-pink: #DB4C9A !important;
            --ll-good: #168B66 !important;
            --ll-warn: #B7791F !important;
            --ll-bad: #D94C68 !important;
            --ll-info: #2563EB !important;
            --ll-shadow: 0 12px 35px rgba(121, 80, 200, 0.08) !important;
            --ll-shadow-sm: 0 4px 14px rgba(121, 80, 200, 0.05) !important;
            --ll-bg-gradient: radial-gradient(900px 520px at 12% -8%, rgba(121, 80, 200, 0.09), transparent 60%),
                               radial-gradient(800px 480px at 95% 0%, rgba(219, 76, 154, 0.07), transparent 60%),
                               #F7F5FC !important;
            --ll-card-bg: linear-gradient(180deg, #FFFFFF, #FAF8FE) !important;
            --ll-card-glass: rgba(255, 255, 255, 0.88) !important;
            --ll-nav-bg: rgba(247, 245, 252, 0.85) !important;
            --ll-menu-bg: #FFFFFF !important;
            --ll-score-grad: linear-gradient(180deg, #29243A 0%, #7950C8 100%) !important;
            --ll-ring-inner: radial-gradient(circle at 50% 30%, #FFFFFF 0%, #F0ECF8 70%) !important;
            --ll-ring-track: rgba(0, 0, 0, 0.06) !important;
            --ll-tile-text: #3B3550 !important;
            --ll-input-bg: #FFFFFF !important;
            --ll-input-color: #29243A !important;
            --ll-slider-track: #E2D9F3 !important;
        }
        """
    else:
        theme_override = """
        :root, .stApp {
            --ll-bg: #0D0B19 !important;
            --ll-surface: #19162A !important;
            --ll-surface-2: #211D3B !important;
            --ll-surface-glass: rgba(25, 22, 42, 0.78) !important;
            --ll-border: #302A45 !important;
            --ll-border-strong: #423A5E !important;
            --ll-text: #F5F2FF !important;
            --ll-muted: #AAA5C9 !important;
            --ll-faint: #726D92 !important;
            --ll-purple: #8B5CF6 !important;
            --ll-violet: #A78BFA !important;
            --ll-pink: #F472B6 !important;
            --ll-good: #34D399 !important;
            --ll-warn: #FBBF24 !important;
            --ll-bad: #FB7185 !important;
            --ll-info: #60A5FA !important;
            --ll-shadow: 0 16px 40px rgba(0, 0, 0, 0.40) !important;
            --ll-shadow-sm: 0 4px 16px rgba(0, 0, 0, 0.25) !important;
            --ll-bg-gradient: radial-gradient(900px 520px at 12% -8%, rgba(139, 92, 246, 0.22), transparent 60%),
                               radial-gradient(800px 480px at 95% 0%, rgba(244, 114, 182, 0.14), transparent 60%),
                               #0D0B19 !important;
            --ll-card-bg: linear-gradient(180deg, rgba(33, 29, 59, 0.75), rgba(25, 22, 42, 0.88)) !important;
            --ll-card-glass: rgba(25, 22, 42, 0.70) !important;
            --ll-nav-bg: rgba(13, 11, 25, 0.82) !important;
            --ll-menu-bg: #19162A !important;
            --ll-score-grad: linear-gradient(180deg, #FFFFFF 0%, #CFC6FF 100%) !important;
            --ll-ring-inner: radial-gradient(circle at 50% 30%, #211D3B 0%, #110F20 70%) !important;
            --ll-ring-track: rgba(255, 255, 255, 0.08) !important;
            --ll-tile-text: #E0DDF5 !important;
            --ll-input-bg: rgba(255, 255, 255, 0.05) !important;
            --ll-input-color: #F5F2FF !important;
            --ll-slider-track: rgba(255, 255, 255, 0.12) !important;
        }
        """

    st.markdown(
        f"<style>{css}\n{theme_override}</style>",
        unsafe_allow_html=True,
    )


load_css(active_theme)

# ==========================================================
# LIFELENS UI HELPERS  (presentation only — no calculations)
# ==========================================================
import html as _html


def _esc(value):
    """HTML-escape any value before placing it inside markup."""
    return _html.escape(str(value))


def _fmt_num(value, decimals=1):
    """Format an existing value for display: 7.0 -> '7', 7.5 -> '7.5'."""
    number = _valid_number(value)
    if number is None:
        return None
    text = f"{number:.{decimals}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def ll_section(icon, title, subtitle=""):
    sub = f'<div class="ll-section-sub">{_esc(subtitle)}</div>' if subtitle else ""
    st.html(
        f'<div class="ll-section-head"><div class="ll-section-title">'
        f'<span>{icon}</span>{_esc(title)}</div>{sub}</div>'
    )


def ll_style_fig(fig, accent=None, height=None, theme=None):
    """Re-skin an EXISTING Plotly figure dynamically for light or dark theme."""
    if theme is None:
        theme = st.session_state.get("lifelens_theme", "dark")

    if theme == "light":
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, system-ui, sans-serif", color="#29243A"),
            title_font=dict(color="#29243A", size=15),
            hoverlabel=dict(bgcolor="#FFFFFF", font_color="#29243A", bordercolor="#E8E1F4"),
        )
        fig.update_xaxes(
            gridcolor="rgba(0,0,0,0.06)",
            linecolor="rgba(0,0,0,0.12)",
            zerolinecolor="rgba(0,0,0,0.12)"
        )
        fig.update_yaxes(
            gridcolor="rgba(0,0,0,0.06)",
            linecolor="rgba(0,0,0,0.12)",
            zerolinecolor="rgba(0,0,0,0.12)"
        )
    else:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, system-ui, sans-serif", color="#AAA5C9"),
            title_font=dict(color="#F5F2FF", size=15),
            hoverlabel=dict(bgcolor="#19162A", font_color="#F5F2FF", bordercolor="#302A45"),
        )
        fig.update_xaxes(
            gridcolor="rgba(255,255,255,0.06)",
            linecolor="rgba(255,255,255,0.10)",
            zerolinecolor="rgba(255,255,255,0.10)"
        )
        fig.update_yaxes(
            gridcolor="rgba(255,255,255,0.06)",
            linecolor="rgba(255,255,255,0.10)",
            zerolinecolor="rgba(255,255,255,0.10)"
        )

    if accent:
        fig.update_traces(line_color=accent, marker_color=accent)
    if height:
        fig.update_layout(height=height)
    return fig


def ll_style_radar(fig, theme=None):
    if theme is None:
        theme = st.session_state.get("lifelens_theme", "dark")
    ll_style_fig(fig, theme=theme)
    if theme == "light":
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    gridcolor="rgba(0,0,0,0.08)",
                    linecolor="rgba(0,0,0,0.12)",
                    tickfont=dict(color="#77718C")
                ),
                angularaxis=dict(
                    gridcolor="rgba(0,0,0,0.08)",
                    linecolor="rgba(0,0,0,0.12)",
                    tickfont=dict(color="#29243A")
                ),
            )
        )
        fig.update_traces(
            line_color="#7950C8",
            fillcolor="rgba(121, 80, 200, 0.20)",
            marker_color="#DB4C9A",
            selector=dict(type="scatterpolar")
        )
    else:
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    gridcolor="rgba(255,255,255,0.10)",
                    linecolor="rgba(255,255,255,0.10)",
                    tickfont=dict(color="#8F8BB8")
                ),
                angularaxis=dict(
                    gridcolor="rgba(255,255,255,0.10)",
                    linecolor="rgba(255,255,255,0.10)",
                    tickfont=dict(color="#F4F2FF")
                ),
            )
        )
        fig.update_traces(
            line_color="#A78BFA",
            fillcolor="rgba(139, 92, 246, 0.28)",
            marker_color="#F472B6",
            selector=dict(type="scatterpolar")
        )
    return fig


_CATEGORY_CLASS = {
    "Excellent": "ll-cat-excellent",
    "Healthy": "ll-cat-healthy",
    "Needs Improvement": "ll-cat-needs",
    "At Risk": "ll-cat-risk",
}

_STATUS_TONE = {
    "Excellent": "good",
    "Good": "good",
    "Fair": "fair",
    "Needs Improvement": "bad",
}


def ll_parse_report(text):
    """
    Split the V5 NLP report into its own headings for display.
    Supports V5 FLAN-T5 generated headings:
    OVERALL, SUPPORTING SIGNALS, KEY FACTOR, MAIN CONCERN, RECENT CHANGE, PATTERN, PRIORITY
    Returns [(heading, body), ...]; empty list if headings are not found.
    """
    cleaned = re.sub(r"\*\*", "", str(text))
    cleaned = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F]+", "", cleaned)
    pattern = re.compile(
        r"(?:^|\s)(?:[0-9]+\.\s*)?(OVERALL|SUPPORTING SIGNALS|KEY FACTOR|MAIN CONCERN|RECENT CHANGE|PATTERN|PRIORITY)\s*:?\s*",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(cleaned))
    sections = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
        body = re.sub(r"\s*\d+\.\s*$", "", cleaned[m.end():end].strip())
        heading = m.group(1).upper()
        if heading == "OVERALL":
            head_title = "Overall Wellness"
        elif heading in ("SUPPORTING SIGNALS", "KEY FACTOR"):
            head_title = "Key Strengths"
        elif heading in ("MAIN CONCERN", "PATTERN"):
            head_title = "Key Observations"
        elif heading == "RECENT CHANGE":
            head_title = "Recent Change"
        elif heading == "PRIORITY":
            head_title = "Actionable Priority"
        else:
            head_title = heading.title()
        if body:
            sections.append((head_title, body))
    return sections


try:
    # Existing Python-side forecast direction logic (same facts the NLP
    # forecast explanation uses). Imported, never modified.
    from ml.nlp.reporter import _build_forecast_facts as _forecast_facts
except Exception:  # pragma: no cover - arrows simply won't render
    _forecast_facts = None

_FORECAST_UNITS = {
    "Mood": "/10", "Energy": "/10", "Stress": "/10", "Productivity": "/10",
    "SleepHours": " h", "ExerciseMinutes": " min", "ScreenTime": " h",
}




st.markdown('<div id="top"></div>', unsafe_allow_html=True)

CSV_FILE = "data/daily_checkins.csv"

today = datetime.now()
today_date = today.date()
today_str = today.strftime("%Y-%m-%d")

greeting = "🌅 Good Morning"

if 12 <= today.hour < 17:
    greeting = "☀️ Good Afternoon"

elif today.hour >= 17:
    greeting = "🌙 Good Evening"

st.html(
    f"""
    <div class="ll-header">
        <div class="ll-brand">
            <div class="ll-brand-mark">🌱</div>
            <div>
                <div class="ll-brand-name">LifeLens</div>
                <div class="ll-brand-tag">Your personal wellness command center</div>
            </div>
        </div>
        <div class="ll-date-pill"><span class="ll-dot"></span>Today • {today.strftime("%b")} {today.day}</div>
    </div>
    """
)
data = load_data()

# Centralized check-in state for today
today_recorded = False
today_row = None

if not data.empty:
    data = data.copy()
    if "Date" in data.columns:
        normalized_dates = normalize_date_series(data["Date"])
        data["Date"] = pd.to_datetime(normalized_dates, errors="coerce")
        data = data.dropna(subset=["Date"]).sort_values("Date")
        data = data.drop_duplicates(subset=["Date"], keep="last")
        today_mask = (data["Date"].dt.strftime("%Y-%m-%d") == today_str)
        if today_mask.any():
            today_recorded = True
            today_row = data[today_mask].iloc[-1]

    numeric_columns = [
        "SleepHours",
        "ScreenTime",
        "ExerciseMinutes",
        "StudyHours",
        "Mood",
        "Energy",
        "Stress",
        "Productivity"
    ]
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

# Set safe defaults so the dashboard handles missing check-in or empty dataset safely.
wellness_score = None
wellness_category = "Not recorded"
score_breakdown = []
available_factors = 0
radar_chart = create_wellness_radar_chart(score_breakdown)

summary = None
insights = []
trend_analysis = None
ml_prediction = None
nlp_wellness_report = None
weekly_facts = {}
weekly_ai_analysis = None
forecast_explanation = None
prediction = {}

# Compute Today's Wellness and Today's Report ONLY if today's check-in exists
if today_recorded and today_row is not None:
    wellness_score, wellness_category = calculate_wellness_score(
        today_row["SleepHours"],
        today_row["Mood"],
        today_row["Stress"],
        today_row["Energy"],
        today_row["ExerciseMinutes"],
        today_row["ScreenTime"],
        today_row["Productivity"],
    )
    
    score_factors = [
        today_row["SleepHours"],
        today_row["Mood"],
        today_row["Stress"],
        today_row["Energy"],
        today_row["ExerciseMinutes"],
        today_row["ScreenTime"],
        today_row["Productivity"],
    ]

    available_factors = sum(
        1 for value in score_factors
        if _valid_number(value) is not None
    )

    score_breakdown = get_score_breakdown(
        today_row["SleepHours"],
        today_row["Mood"],
        today_row["Stress"],
        today_row["Energy"],
        today_row["ExerciseMinutes"],
        today_row["ScreenTime"],
        today_row["Productivity"],
    )

    radar_chart = create_wellness_radar_chart(
        score_breakdown,
        recorded_values={
            "SleepHours": today_row["SleepHours"],
            "Mood": today_row["Mood"],
            "Stress": today_row["Stress"],
            "Energy": today_row["Energy"],
            "ExerciseMinutes": today_row["ExerciseMinutes"],
            "ScreenTime": today_row["ScreenTime"],
            "StudyHours": today_row.get("StudyHours"),
            "Productivity": today_row["Productivity"],
        },
    )

    nlp_wellness_report = generate_wellness_report(
        wellness_score,
        wellness_category,
        today_row,
        history=data,
        daily_note=today_row.get("DailyNote", "") if pd.notna(today_row.get("DailyNote")) else "",
    )

# Historical analysis, trends, correlations, and reports remain available from history
if not data.empty:
    summary = get_summary(data)
    insights = generate_insights(data)
    trend_analysis = generate_trend_analysis(data)
    recommendations = generate_recommendations(data)
    patterns = analyze_patterns(data)

    # Existing weekly report remains unchanged
    weekly_report = generate_weekly_report(data)

    # Deterministic weekly facts calculation using authoritative facts engine
    from src.nlp_facts import build_weekly_facts
    weekly_facts = build_weekly_facts(data)

    # Local V5 NLP model explains those facts with validation and fallback
    weekly_ai_analysis = explain_weekly_facts(
        weekly_facts
    )

    monthly_report = generate_monthly_report(data)
    correlation_matrix = get_correlation_matrix(data)

    biggest_discovery = get_biggest_discovery(data)
    next_step = get_todays_focus(data)
    today_focus = get_todays_focus(data)

# =========================================================
# TOMORROW'S FORECAST (BASE + ML)
# ONLY run if today's check-in has been recorded
# =========================================================
if today_recorded and today_row is not None:
    try:
        prediction = predict_next_day(data) or {}
    except Exception as error:
        print("Base forecast unavailable:", repr(error))
        prediction = {}

    if ml_available():
        try:
            ml_pred = predict_next_day_ml(data) or {}
            ml_metrics = ["Mood", "Energy", "Stress", "Productivity"]
            for metric in ml_metrics:
                if metric in ml_pred and ml_pred[metric] is not None:
                    prediction[metric] = ml_pred[metric]
        except Exception as error:
            print("ML forecast unavailable:", repr(error))

    if prediction:
        try:
            tomorrow_str = (today + timedelta(days=1)).strftime("%Y-%m-%d")
            log_prediction(prediction, today_str, tomorrow_str)
        except Exception as pred_err:
            print("Forecast logging error:", repr(pred_err))
        try:
            forecast_explanation = explain_tomorrow_forecast(
                prediction,
                latest=today_row,
                history=data,
                daily_note=today_row.get("DailyNote", "") if pd.notna(today_row.get("DailyNote")) else ""
            )
        except Exception as error:
            print("Forecast explanation unavailable:", repr(error))
            forecast_explanation = None

selected_tab = show_navigation()

if selected_tab == "📝 Daily Check-in":
    st.html(
        """
        <div class="ll-page-head">
            <span class="ll-eyebrow">⏱ Daily Wellness Journal · Guided Check-in</span>
            <h1>Daily Check-in</h1>
            <p>Take a minute to reflect and keep your wellness picture up to date.</p>
        </div>
        """
    )

    # Display celebration card if a check-in was just submitted
    if st.session_state.get("checkin_success_message"):
        sc = st.session_state.pop("last_submitted_score", None)
        cat = st.session_state.pop("last_submitted_category", None)
        d_str = st.session_state.pop("last_submitted_date", None)
        msg = st.session_state.pop("checkin_success_message")

        st.success(f"✓ {msg}")
        if sc is not None:
            cat_cls = _CATEGORY_CLASS.get(cat, "ll-cat-healthy")
            st.html(
                f"""
                <div class="ll-card {cat_cls}" style="margin-bottom: 1.5rem; text-align: center; padding: 22px 24px;">
                    <span class="ll-chip" style="margin-bottom: 8px;">🎉 Check-in Recorded for {d_str}</span>
                    <div style="font-size: 2.8rem; font-weight: 800; letter-spacing: -0.03em; color: var(--ll-text); line-height: 1.1;">
                        {sc:.0f}<small style="font-size: 1.1rem; color: var(--ll-faint);">/100</small>
                    </div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: var(--ll-cat); margin-top: 6px;">
                        {cat} Wellness Status
                    </div>
                </div>
                """
            )
            col_dash, _ = st.columns([1.5, 3])
            with col_dash:
                if st.button("👉 View in Dashboard", type="primary", use_container_width=True, key="btn_goto_dash"):
                    st.session_state["lifelens_page_selector"] = "🏠 Dashboard"
                    st.rerun()

    # Load existing records to detect whether selected date already exists
    existing_data = load_data()
    if not existing_data.empty and "Date" in existing_data.columns:
        existing_data["Date"] = normalize_date_series(existing_data["Date"])
    else:
        existing_data = pd.DataFrame()

    # Date selector (outside form for instant reactivity and pre-population)
    sel_col1, sel_col2 = st.columns([1.5, 2.5], gap="medium", vertical_alignment="center")
    with sel_col1:
        date = st.date_input(
            "📅 Check-in Date",
            value=today_date,
            max_value=today_date,
            help="Select today or a past date to record or update your wellness entry.",
            key="checkin_date_picker"
        )

    date_string = pd.to_datetime(date).strftime("%Y-%m-%d")
    is_today_date = (date_string == today_str)

    matched_entry = pd.DataFrame()
    if not existing_data.empty:
        matched_entry = existing_data[existing_data["Date"] == date_string]

    already_exists = not matched_entry.empty
    prev_row = matched_entry.iloc[-1] if already_exists else None

    with sel_col2:
        if already_exists:
            st.html(
                f"""
                <div style="background: rgba(251, 191, 36, 0.09); border: 1px solid rgba(251, 191, 36, 0.35); border-radius: 12px; padding: 10px 16px;">
                    <span style="color: var(--ll-warn); font-weight: 700; font-size: 0.88rem;">✏️ Existing Record Detected ({date_string})</span><br>
                    <span style="color: var(--ll-muted); font-size: 0.82rem;">A check-in already exists. Submitting will update your existing entry with new values.</span>
                </div>
                """
            )
        else:
            st.html(
                f"""
                <div style="background: rgba(52, 211, 153, 0.09); border: 1px solid rgba(52, 211, 153, 0.35); border-radius: 12px; padding: 10px 16px;">
                    <span style="color: var(--ll-good); font-weight: 700; font-size: 0.88rem;">🌱 New Record Entry ({date_string})</span><br>
                    <span style="color: var(--ll-muted); font-size: 0.82rem;">No entry recorded yet for this date. Complete the journal below.</span>
                </div>
                """
            )

    # Pre-populate defaults from existing record if available
    def_sleep = float(prev_row["SleepHours"]) if (prev_row is not None and pd.notna(prev_row.get("SleepHours"))) else 7.5
    def_screen = float(prev_row["ScreenTime"]) if (prev_row is not None and pd.notna(prev_row.get("ScreenTime"))) else 4.0
    def_exercise = int(prev_row["ExerciseMinutes"]) if (prev_row is not None and pd.notna(prev_row.get("ExerciseMinutes"))) else 30
    def_study = float(prev_row["StudyHours"]) if (prev_row is not None and pd.notna(prev_row.get("StudyHours"))) else 4.0
    def_mood = int(prev_row["Mood"]) if (prev_row is not None and pd.notna(prev_row.get("Mood"))) else 7
    def_energy = int(prev_row["Energy"]) if (prev_row is not None and pd.notna(prev_row.get("Energy"))) else 7
    def_stress = int(prev_row["Stress"]) if (prev_row is not None and pd.notna(prev_row.get("Stress"))) else 3
    def_prod = int(prev_row["Productivity"]) if (prev_row is not None and pd.notna(prev_row.get("Productivity"))) else 7
    def_note = str(prev_row["DailyNote"]) if (prev_row is not None and pd.notna(prev_row.get("DailyNote"))) else ""

    # Completion progress indicator
    st.html(
        """
        <div style="display: flex; align-items: center; justify-content: space-between; margin: 18px 0 10px;">
            <span style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: var(--ll-violet);">
                Journal Completion Meter
            </span>
            <span style="font-size: 0.8rem; font-weight: 700; color: var(--ll-good);">
                8 of 8 Core Dimensions Ready (100%)
            </span>
        </div>
        <div style="height: 6px; width: 100%; background: var(--ll-slider-track); border-radius: 999px; overflow: hidden; margin-bottom: 22px;">
            <div style="height: 100%; width: 100%; background: linear-gradient(90deg, var(--ll-purple), var(--ll-pink)); border-radius: 999px;"></div>
        </div>
        """
    )

    with st.form("daily_checkin_journal"):
        # Section 1: Sleep & Recovery & Digital Habits
        with st.container(border=True, key="panel_sleep_digital"):
            st.html(
                """
                <div class="ll-panel-head">
                    <h3>😴 Sleep, Recovery & Digital Habits</h3>
                    <span>Capture the restorative balance and screen exposure that framed your day.</span>
                </div>
                """
            )
            col_s, col_d = st.columns(2, gap="medium")
            with col_s:
                sleep_hours = st.slider(
                    "Hours of Sleep (hrs)",
                    min_value=0.0,
                    max_value=14.0,
                    value=def_sleep,
                    step=0.5,
                    format="%.1f hrs",
                    help="Total time slept overnight and including naps. Optimal range: 7.0 – 9.0 hrs."
                )
            with col_d:
                screen_time = st.slider(
                    "Screen Time (hrs)",
                    min_value=0.0,
                    max_value=24.0,
                    value=def_screen,
                    step=0.5,
                    format="%.1f hrs",
                    help="Recreational, mobile, and non-essential display time. Target under 4.0 hrs."
                )

        # Section 2: Physical Activity & Learning & Work
        with st.container(border=True, key="panel_activity_work"):
            st.html(
                """
                <div class="ll-panel-head">
                    <h3>🏃 Physical Activity & Intellectual Engagement</h3>
                    <span>Active movement sessions and focused professional/study hours.</span>
                </div>
                """
            )
            col_a, col_w = st.columns(2, gap="medium")
            with col_a:
                exercise_minutes = st.slider(
                    "Physical Movement / Exercise (min)",
                    min_value=0,
                    max_value=1440,
                    value=def_exercise,
                    step=5,
                    format="%d min",
                    help="Intentional walking, gym workouts, yoga, cycling, or active sports."
                )
            with col_w:
                study_hours = st.slider(
                    "Learning & Deep Work (hrs)",
                    min_value=0.0,
                    max_value=24.0,
                    value=def_study,
                    step=0.5,
                    format="%.1f hrs",
                    help="Dedicated time spent studying, writing, programming, or focused occupational output."
                )

        # Section 3: Emotional Wellbeing & Productivity
        with st.container(border=True, key="panel_emotional_prod"):
            st.html(
                """
                <div class="ll-panel-head">
                    <h3>😊 Emotional Wellbeing & Energy</h3>
                    <span>Reflect on your mood, cognitive pressure, and stamina levels (1 to 10 scale).</span>
                </div>
                """
            )
            em_col1, em_col2 = st.columns(2, gap="medium")
            with em_col1:
                mood = st.slider(
                    "😊 Subjective Mood (1 = Very Low / Depleted, 5 = Neutral, 10 = Peak / Joyful)",
                    min_value=1,
                    max_value=10,
                    value=def_mood,
                    step=1,
                    help="How was your overarching emotional state today?"
                )
                stress = st.slider(
                    "🧘 Stress Level (1 = Completely Calm, 5 = Manageable, 10 = Overwhelmed)",
                    min_value=1,
                    max_value=10,
                    value=def_stress,
                    step=1,
                    help="Level of mental tension, anxiety, or cognitive pressure experienced today."
                )
            with em_col2:
                energy = st.slider(
                    "⚡ Vital Energy (1 = Exhausted, 5 = Moderate, 10 = High Stamina)",
                    min_value=1,
                    max_value=10,
                    value=def_energy,
                    step=1,
                    help="Physical and mental vigor throughout the waking hours."
                )
                productivity = st.slider(
                    "🎯 Productivity Output (1 = Minimal Progress, 5 = Steady, 10 = Exceptional)",
                    min_value=1,
                    max_value=10,
                    value=def_prod,
                    step=1,
                    help="How effective and accomplished did you feel with your intended goals?"
                )



        submit_btn_label = f"🔄 Update Today's Check-in ({date_string})" if (already_exists and is_today_date) else (
            f"🔄 Update Check-in for {date_string}" if already_exists else (
                f"✓ Complete Today's Check-in ({date_string})" if is_today_date else f"✓ Complete Check-in for {date_string}"
            )
        )

        submitted = st.form_submit_button(
            submit_btn_label,
            use_container_width=True,
            type="primary"
        )

        if submitted:
            if date > today_date:
                st.error("Future dates cannot be recorded.")
                st.stop()

            new_entry = pd.DataFrame([{
                "Date": date,
                "SleepHours": sleep_hours,
                "ScreenTime": screen_time,
                "ExerciseMinutes": exercise_minutes,
                "StudyHours": study_hours,
                "Mood": mood,
                "Energy": energy,
                "Stress": stress,
                "Productivity": productivity,
                "DailyNote": daily_note.strip() if daily_note and daily_note.strip() else None
            }])

            try:
                save_data(new_entry)
                cal_score, cal_cat = calculate_wellness_score(
                    sleep_hours, mood, stress, energy, exercise_minutes, screen_time, productivity
                )
                st.session_state["last_submitted_score"] = cal_score
                st.session_state["last_submitted_category"] = cal_cat
                st.session_state["last_submitted_date"] = date_string
                action_word = "updated" if already_exists else "recorded"
                st.session_state["checkin_success_message"] = f"Your check-in for {date_string} has been successfully {action_word}."
                st.rerun()
            except Exception as e:
                st.error(f"Unable to save check-in: {e}")

    with st.expander("🔍 View Raw Historical Check-in Data"):
        st.dataframe(data, use_container_width=True, hide_index=False)

if selected_tab == "🏠 Dashboard":

    if data.empty:
        st.html(
            """
            <div class="ll-empty">
                <span class="ll-empty-icon">🌱</span>
                <b style="color:#F4F2FF;font-size:1.1rem;">Welcome to LifeLens</b><br>
                Your personal wellness journey starts with your first check-in.<br>
                Go to <b>📝 Daily Check-in</b> and complete your first entry
                to unlock your personalised dashboard.
            </div>
            """
        )
        st.stop()

    if not today_recorded:
        st.html(
            f"""
            <div class="ll-empty" style="border: 1px solid rgba(245, 158, 11, 0.4); background: rgba(245, 158, 11, 0.08); margin-bottom: 2rem;">
                <span class="ll-empty-icon">📅</span>
                <b style="color:#FBBF24;font-size:1.15rem;">Today's Check-in Pending</b><br>
                Today's check-in has not been recorded yet ({today.strftime('%B %d, %Y')}).<br>
                Please complete <b>📝 Daily Check-in</b> for today to unlock Today's Wellness Score, Today's Observations, and Tomorrow's Forecast.<br>
                <span style="color:#8F8BB8;font-size:0.85rem;">Historical trends, observations, correlations, and reports remain available below.</span>
            </div>
            """
        )
    else:
        # ==========================================
        # 1. HERO — TODAY'S WELLNESS SCORE
        # (value + category come straight from calculate_wellness_score)
        # ==========================================
        score_display = f"{round(wellness_score)}" if wellness_score is not None else "--"
        cat_display = wellness_category if wellness_score is not None else "Insufficient Data"
        cat_class = (
            _CATEGORY_CLASS.get(wellness_category, "ll-cat-none")
            if wellness_score is not None else "ll-cat-none"
        )
        ring_pct = max(0.0, min(100.0, float(wellness_score))) if wellness_score is not None else 0.0

        st.html(
            f"""
            <div class="ll-hero {cat_class}">
                <div class="ll-hero-label">Today's Wellness</div>
                <div class="ll-ring" style="--p:{ring_pct:.1f}">
                    <div class="ll-ring-inner">
                        <div class="ll-ring-score">{score_display}</div>
                        <span class="ll-ring-max">/ 100</span>
                    </div>
                </div>
                <div class="ll-hero-cat">{_esc(cat_display)}</div>
                <div class="ll-hero-foot">Based on {available_factors} of 7 recorded factors</div>
            </div>
            """
        )

        # ==========================================
        # 2. METRIC CARDS (real values from today's check-in;
        #    status + bar come from the existing get_score_breakdown)
        # ==========================================
        ll_section("📊", "Today's Signals", "From today's check-in")

        breakdown_lookup = {name: (score, status) for name, score, status in score_breakdown}
        prior_data = data[data["Date"].dt.strftime("%Y-%m-%d") < today_str] if "Date" in data.columns else pd.DataFrame()
        previous = prior_data.iloc[-1] if not prior_data.empty else None

        metric_specs = [
            # label, column, icon, unit, decimals, key in score_breakdown
            ("Sleep", "SleepHours", "😴", "hrs", 1, "Sleep"),
            ("Mood", "Mood", "😊", "/ 10", 0, "Mood"),
            ("Energy", "Energy", "⚡", "/ 10", 0, "Energy"),
            ("Productivity", "Productivity", "🎯", "/ 10", 0, "Productivity"),
            ("Stress", "Stress", "😌", "/ 10", 0, "Stress"),
            ("Screen Time", "ScreenTime", "💻", "hrs", 1, "Screen Time"),
            ("Exercise", "ExerciseMinutes", "🏃", "min", 0, "Exercise"),
            ("Study / Work", "StudyHours", "📚", "hrs", 1, None),
        ]

        metric_cards = []
        for label, column, icon, unit, decimals, bd_key in metric_specs:
            current_value = today_row.get(column)
            shown = _fmt_num(current_value, decimals)
            value_html = (
                f'{_esc(shown)}<small>{_esc(unit)}</small>' if shown is not None else "--"
            )

            score_pct, status_text = breakdown_lookup.get(bd_key, (None, None)) if bd_key else (None, None)
            tone = _STATUS_TONE.get(status_text, "none")

            delta_html = ""
            if previous is not None:
                cur_n = _valid_number(current_value)
                prev_n = _valid_number(previous.get(column))
                if cur_n is not None and prev_n is not None:
                    diff = cur_n - prev_n
                    if abs(diff) < 0.05:
                        delta_html = "→ same as last"
                    else:
                        arrow = "↑" if diff > 0 else "↓"
                        delta_html = f"{arrow} {_esc(_fmt_num(abs(diff), decimals) or '0')} vs last"

            status_html = f'<span class="ll-status">{_esc(status_text)}</span>' if status_text else "<span></span>"
            bar_html = ""
            if score_pct is not None:
                bar_html = f'<div class="ll-bar"><span style="width:{max(4, min(100, float(score_pct))):.0f}%"></span></div>'

            metric_cards.append(
                f"""
                <div class="ll-metric ll-tone-{tone}">
                    <div class="ll-metric-top">
                        <div class="ll-metric-name"><span class="ll-metric-icon">{icon}</span>{_esc(label)}</div>
                    </div>
                    <div class="ll-metric-value">{value_html}</div>
                    <div class="ll-metric-meta">{status_html}<span class="ll-delta">{delta_html}</span></div>
                    {bar_html}
                </div>
                """
            )

        st.html(f'<div class="ll-metrics">{"".join(metric_cards)}</div>')

        # ==========================================
        # 3. AI INSIGHT  +  TOMORROW'S FORECAST
        # ==========================================
        st.html('<div style="height: 1.6rem;"></div>')
        col_insight, col_forecast = st.columns([1.15, 1], gap="medium")

        with col_insight:
            if nlp_wellness_report:
                if isinstance(nlp_wellness_report, dict) and nlp_wellness_report.get("status") in ("not_found", "load_failed", "generation_failed"):
                    err_msg = nlp_wellness_report.get("message", "AI Wellness Insight is currently unavailable.")
                    report_html = f'<div class="ll-report-raw" style="color:#FCA5A5;line-height:1.5;">{_esc(err_msg)}</div>'
                else:
                    raw_text = nlp_wellness_report.get("text", "") if isinstance(nlp_wellness_report, dict) else str(nlp_wellness_report)
                    report_sections = ll_parse_report(raw_text)
                    if report_sections:
                        report_items = "".join(
                            f'<div class="ll-report-item"><div class="ll-report-head">{_esc(head)}</div>'
                            f'<div class="ll-report-text">{_esc(body)}</div></div>'
                            for head, body in report_sections
                        )
                        report_html = f'<div class="ll-report">{report_items}</div>'
                    else:
                        report_html = f'<div class="ll-report-raw">{_esc(raw_text.strip())}</div>'
            else:
                report_html = (
                    '<div class="ll-report-raw" style="color:#A9A5CE;">'
                    'The AI Wellness Report is temporarily unavailable.</div>'
                )

            st.html(
                f"""
                <div class="ll-card">
                    <div class="ll-card-label"><span>🧠 AI Wellness Insight</span><span class="ll-chip">AI generated</span></div>
                    {report_html}
                </div>
                """
            )

        with col_forecast:
            if prediction:
                forecast_summary = generate_forecast_summary(prediction)
                t_score, t_cat = calculate_wellness_score(
                    prediction.get("SleepHours"),
                    prediction.get("Mood"),
                    prediction.get("Stress"),
                    prediction.get("Energy"),
                    prediction.get("ExerciseMinutes"),
                    prediction.get("ScreenTime"),
                    prediction.get("Productivity"),
                )
                t_score_str = f"{round(t_score)}" if t_score is not None else "--"

                forecast_facts = []
                if _forecast_facts is not None:
                    try:
                        forecast_facts = _forecast_facts(prediction, today_row)
                    except Exception as error:
                        print("Forecast direction facts unavailable:", repr(error))

                forecast_rows = []
                for fact in forecast_facts:
                    predicted_txt = _fmt_num(fact["predicted"], 1)
                    if predicted_txt is None:
                        continue
                    unit = _FORECAST_UNITS.get(fact["metric"], "")
                    change_type = fact.get("change_type")
                    status = fact.get("status")
                    arrow = {"increase": "↑", "decrease": "↓", "same": "→"}.get(change_type, "–")
                    arrow_cls = (
                        "good" if status == "improving"
                        else "bad" if status in ("worsening", "declining")
                        else "flat"
                    )
                    forecast_rows.append(
                        f'<div class="ll-row"><span class="ll-row-name">{_esc(fact["name"])}</span>'
                        f'<span class="ll-row-val">{_esc(predicted_txt)}{_esc(unit)}</span>'
                        f'<span class="ll-arrow {arrow_cls}">{arrow}</span></div>'
                    )

                st.html(
                    f"""
                    <div class="ll-card">
                        <div class="ll-card-label"><span>🔮 Tomorrow's Forecast</span>
                            <span class="ll-chip">{_esc(forecast_summary)}</span></div>
                        <div class="ll-forecast-score"><b>{t_score_str}</b><span>/ 100 predicted</span></div>
                        <div class="ll-forecast-cat">{_esc(t_cat)}</div>
                        <div class="ll-rows">{"".join(forecast_rows)}</div>
                    </div>
                    """
                )
            else:
                st.html(
                    """
                    <div class="ll-empty">
                        <span class="ll-empty-icon">🔮</span>
                        <b style="color:#F4F2FF;">Tomorrow's Forecast</b><br>
                        Log check-ins for a few consecutive days to unlock your AI forecast.
                    </div>
                    """
                )

        # Forecast explanation (generated by V5 FLAN-T5 LLM)
        if prediction and forecast_explanation:
            if isinstance(forecast_explanation, dict) and forecast_explanation.get("status") in ("not_found", "load_failed", "generation_failed"):
                err_msg = forecast_explanation.get("message", "Tomorrow Forecast explanation is currently unavailable.")
                st.html(
                    f"""
                    <div class="ll-card" style="margin-top: 1.1rem;">
                        <div class="ll-card-label"><span>🔮 Tomorrow Forecast Analysis</span><span class="ll-chip">AI explained</span></div>
                        <div class="ll-report-raw" style="color:#FCA5A5;line-height:1.5;">{_esc(err_msg)}</div>
                    </div>
                    """
                )
            else:
                outlook_text = (
                    forecast_explanation.get("Tomorrow Outlook")
                    or forecast_explanation.get("outlook")
                    or forecast_explanation.get("Tomorrow's Outlook")
                    or "No insight available."
                )
                factor_text = (
                    forecast_explanation.get("Main Factor")
                    or forecast_explanation.get("main_factor")
                    or forecast_explanation.get("Key Factor")
                    or "No insight available."
                )
                pattern_text = (
                    forecast_explanation.get("Pattern")
                    or forecast_explanation.get("pattern")
                    or forecast_explanation.get("Pattern Detected")
                    or "No insight available."
                )
                priority_text = (
                    forecast_explanation.get("Priority")
                    or forecast_explanation.get("priority")
                    or forecast_explanation.get("Tomorrow's Focus")
                    or "No insight available."
                )

                st.html(
                    f"""
                    <div class="ll-card" style="margin-top: 1.1rem;">
                        <div class="ll-card-label"><span>🔮 Tomorrow Forecast Analysis</span><span class="ll-chip">AI explained</span></div>
                        <div class="ll-grid-4">
                            <div class="ll-tile"><div class="ll-tile-head">Tomorrow Outlook</div><div class="ll-tile-text">{_esc(outlook_text)}</div></div>
                            <div class="ll-tile"><div class="ll-tile-head">Main Factor</div><div class="ll-tile-text">{_esc(factor_text)}</div></div>
                            <div class="ll-tile"><div class="ll-tile-head">Pattern</div><div class="ll-tile-text">{_esc(pattern_text)}</div></div>
                            <div class="ll-tile"><div class="ll-tile-head">Priority</div><div class="ll-tile-text">{_esc(priority_text)}</div></div>
                        </div>
                    </div>
                    """
                )

    # ==========================================
    # 4. WELLNESS TRENDS (existing charts + existing trend analysis)
    # ==========================================
    ll_section("📈", "Wellness Trends", "Your recent check-in history")

    with st.container(key="ll_trends"):
        trend_sum_col, trend_charts_col = st.columns([1, 1.65], gap="large")

        with trend_sum_col:
            trend_pos = trend_analysis.get("positive", "Keep logging to unlock trend analysis.") if trend_analysis else "Keep logging."
            trend_watch = trend_analysis.get("watch", "No major changes to monitor.") if trend_analysis else "No data."
            trend_rec = trend_analysis.get("recommendation", "Maintain routine.") if trend_analysis else "Maintain routine."

            st.html(
                f"""
                <div class="ll-card-label" style="margin-top:6px;"><span>🧠 AI Trend Assessment</span></div>
                <div class="ll-trend-list">
                    <div class="ll-trend-item pos"><div class="ll-tile-head" style="color:#34D399;">Positive shift</div><div class="ll-tile-text">{_esc(trend_pos)}</div></div>
                    <div class="ll-trend-item watch"><div class="ll-tile-head" style="color:#FBBF24;">Area to watch</div><div class="ll-tile-text">{_esc(trend_watch)}</div></div>
                    <div class="ll-trend-item info"><div class="ll-tile-head" style="color:#60A5FA;">Guidance</div><div class="ll-tile-text">{_esc(trend_rec)}</div></div>
                </div>
                """
            )

        with trend_charts_col:
            with st.container(key="trends_tabs"):
                sleep_tab, mood_tab, stress_tab, screen_tab = st.tabs([
                    "😴 Sleep",
                    "😊 Mood",
                    "😌 Stress",
                    "📱 Screen",
                ])
                trend_margin = dict(l=10, r=10, t=40, b=20)
                with sleep_tab:
                    fig_sleep = ll_style_fig(plot_sleep_trend(data), accent="#A78BFA")
                    fig_sleep.update_layout(height=260, margin=trend_margin)
                    st.plotly_chart(fig_sleep, use_container_width=True)
                with mood_tab:
                    fig_mood = ll_style_fig(plot_mood_trend(data), accent="#F472B6")
                    fig_mood.update_layout(height=260, margin=trend_margin)
                    st.plotly_chart(fig_mood, use_container_width=True)
                with stress_tab:
                    fig_stress = ll_style_fig(plot_stress_trend(data), accent="#FB7185")
                    fig_stress.update_layout(height=260, margin=trend_margin)
                    st.plotly_chart(fig_stress, use_container_width=True)
                with screen_tab:
                    fig_screen = ll_style_fig(plot_screen_time_trend(data), accent="#60A5FA")
                    fig_screen.update_layout(height=260, margin=trend_margin)
                    st.plotly_chart(fig_screen, use_container_width=True)

    # ==========================================
    # 5. TODAY'S FOCUS  +  BIGGEST DISCOVERY
    # (text comes from get_todays_focus / generate_recommendations /
    #  generate_ai_forecast_insight / get_biggest_discovery — unchanged)
    # ==========================================
    ll_section("🎯", "Focus & Discovery")

    top_recs = [r for r in recommendations if r != today_focus][:3]
    col_focus, col_discovery = st.columns([1.15, 1], gap="medium")

    with col_focus:
        checks_html = "".join(
            f'<div class="ll-check"><i>✓</i><span>{_esc(rec)}</span></div>' for rec in top_recs
        )
        supporting_html = (
            f'<div class="ll-sub-label">Supporting habits</div><div class="ll-checks">{checks_html}</div>'
            if top_recs else ""
        )
        why_html = ""
        if prediction:
            why_html = (
                f'<div class="ll-why"><b>Why this matters</b><br>'
                f'{_esc(generate_ai_forecast_insight(prediction))}</div>'
            )
        st.html(
            f"""
            <div class="ll-card">
                <div class="ll-card-label"><span>🎯 Today's Focus</span></div>
                <div class="ll-focus-main">{_esc(today_focus)}</div>
                {supporting_html}
                {why_html}
            </div>
            """
        )

    with col_discovery:
        st.html(
            f"""
            <div class="ll-card ll-discovery">
                <div class="ll-card-label"><span>💡 Biggest Discovery</span></div>
                <div class="ll-discovery-text">{_esc(biggest_discovery)}</div>
            </div>
            """
        )

    # ==========================================
    # 6. WELLNESS BREAKDOWN (existing radar + strongest / focus areas)
    # ==========================================
    ll_section("🧭", "Wellness Breakdown", "Factor scores from today's check-in" if today_recorded else "Factor scores breakdown")

    with st.expander("🔍 View Wellness Radar & Area Breakdown", expanded=False):
        if not today_recorded:
            st.info("Today's check-in is required to view today's wellness radar and factor breakdown.")
        else:
            radar_col, areas_col = st.columns([1.3, 1], gap="medium")
            with radar_col:
                st.plotly_chart(
                    ll_style_radar(radar_chart),
                    use_container_width=True,
                    config={"displayModeBar": False}
                )
            with areas_col:
                st.markdown("##### 🌟 Strongest Areas (≥80)")
                strongest_areas = [
                    (factor, score)
                    for factor, score, status in score_breakdown
                    if score is not None and score >= 80
                ]
                if strongest_areas:
                    for factor, score in strongest_areas:
                        st.html(
                            f"""
                            <div class="wellness-insight-card positive" style="padding: 8px 12px; margin-bottom: 6px;">
                                <span>{factor}</span>
                                <strong>{score:.0f}/100</strong>
                            </div>
                            """
                        )
                else:
                    st.caption("No factors currently at or above 80.")

                st.markdown("##### ⚠️ Focus Areas (<60)")
                attention_areas = [
                    (factor, score)
                    for factor, score, status in score_breakdown
                    if score is not None and score < 60
                ]
                if attention_areas:
                    for factor, score in attention_areas:
                        st.html(
                            f"""
                            <div class="wellness-insight-card attention" style="padding: 8px 12px; margin-bottom: 6px;">
                                <span>{factor}</span>
                                <strong>{score:.0f}/100</strong>
                            </div>
                            """
                        )
                else:
                    st.caption("No factors currently below 60.")

    # ==========================================
    # 7. HABIT RELATIONSHIPS (unchanged logic)
    # ==========================================
    ll_section("🧩", "Habit Relationships", "Key connections across your check-ins")

    insights = generate_correlation_insights(correlation_matrix)
    positive = insights["positive"] if insights else None
    negative = insights["negative"] if insights else None

    positive_explanation = None
    negative_explanation = None
    if positive is not None:
        positive_explanation = generate_relationship_explanation(
            positive["feature1"], positive["feature2"], positive["value"]
        )
    if negative is not None:
        negative_explanation = generate_relationship_explanation(
            negative["feature1"], negative["feature2"], negative["value"]
        )

    col_pos, col_neg = st.columns(2, gap="medium")
    with col_pos:
        if positive is not None:
            show_correlation_card(
                title="Strongest Positive Connection",
                title_color="#34D399",
                progress_color="#34D399",
                icon="🟢",
                correlation=positive,
                explanation=positive_explanation,
            )
        else:
            st.info("A positive relationship cannot be identified from the available data yet.")

    with col_neg:
        if negative is not None:
            show_correlation_card(
                title="Strongest Negative Connection",
                title_color="#FB7185",
                progress_color="#FB7185",
                icon="🔴",
                correlation=negative,
                explanation=negative_explanation,
            )
        else:
            st.info("A negative relationship cannot be identified from the available data yet.")

    with st.expander("📊 Explore Detailed Correlation Heatmap", expanded=False):
        if not correlation_matrix.empty and correlation_matrix.notna().to_numpy().any():
            st.caption(
                "This chart shows how strongly your recorded habits are related. "
                "Positive values indicate habits that tend to move together; "
                "negative values indicate habits that tend to move in opposite directions. "
                "Correlation does not necessarily mean one habit causes another."
            )
            st.plotly_chart(
                ll_style_fig(plot_correlation_heatmap(correlation_matrix)),
                use_container_width=True,
            )
        else:
            st.info(
                "Not enough valid variation in your check-in data to display "
                "the correlation heatmap yet. Record more check-ins with "
                "varied values to reveal relationships."
            )

    # ==========================================
    # 8. WEEKLY AI ANALYSIS (same parsing rules as before)
    # ==========================================
    if weekly_ai_analysis:
        ll_section("🗓️", "Weekly AI Analysis")

        weekly_sections = []  # [[title, [lines]], ...]
        if isinstance(weekly_ai_analysis, dict):
            key_map = [
                (["Weekly Overview", "WEEKLY OVERVIEW", "WEEKLY SUMMARY", "overview", "summary"], "📊 Weekly Overview"),
                (["Key Trend", "KEY TREND", "METRIC COMPARISONS", "trend", "metrics"], "📈 Key Trend"),
                (["Main Factor", "MAIN FACTOR", "ROUTINE CHANGES", "factor", "routine"], "🎯 Main Factor"),
                (["Next Week Priority", "NEXT WEEK PRIORITY", "NEXT WEEK", "priority"], "🚀 Next Week Priority"),
            ]
            for candidate_keys, title in key_map:
                for k in candidate_keys:
                    if k in weekly_ai_analysis and weekly_ai_analysis[k]:
                        val = weekly_ai_analysis[k]
                        lines = val if isinstance(val, list) else [str(val).strip()]
                        weekly_sections.append([title, lines])
                        break
        elif isinstance(weekly_ai_analysis, str):
            headings_map = [
                ("📊 Weekly Overview", r"(?:^|\s+)(?:WEEKLY\s+(?:SUMMARY|OVERVIEW)|OVERVIEW):?"),
                ("📈 Key Trend", r"(?:\s+)(?:METRIC\s+COMPARISONS|KEY\s+TREND|TRENDS?):?"),
                ("🎯 Main Factor", r"(?:\s+)(?:ROUTINE\s+CHANGES|MAIN\s+FACTOR|KEY\s+FACTOR):?"),
                ("🚀 Next Week Priority", r"(?:\s+)(?:NEXT\s+WEEK(?:\s+PRIORITY)?|PRIORITY|FOCUS):?"),
            ]
            positions = []
            for title, regex in headings_map:
                m = re.search(regex, weekly_ai_analysis, re.IGNORECASE)
                if m:
                    positions.append((m.start(), m.end(), title))
            positions.sort(key=lambda x: x[0])
            if positions:
                for idx, (_, end, title) in enumerate(positions):
                    nxt = positions[idx + 1][0] if idx + 1 < len(positions) else len(weekly_ai_analysis)
                    content = weekly_ai_analysis[end:nxt].strip(" \t\n:-")
                    if content:
                        weekly_sections.append([title, [content]])
            else:
                for line in weekly_ai_analysis.split("\n"):
                    line = line.strip()
                    if line:
                        if not weekly_sections:
                            weekly_sections.append(["📊 Weekly Overview", []])
                        weekly_sections[-1][1].append(line)

        weekly_tiles = "".join(
            f'<div class="ll-tile"><div class="ll-tile-head">{_esc(title)}</div>'
            f'<div class="ll-tile-text">{"<br>".join(_esc(l) for l in lines)}</div></div>'
            for title, lines in weekly_sections
        )
        st.html(f'<div class="ll-card"><div class="ll-grid-4">{weekly_tiles}</div></div>')

if selected_tab == "📊 Reports":
    show_reports(monthly_report)
else:
    st.session_state.pop("show_weekly_pdf_download", None)
    st.session_state.pop("show_monthly_pdf_download", None)

if selected_tab == "ℹ️ About":
    show_about()
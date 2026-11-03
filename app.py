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

def load_css():
    with open("assets/style.css", "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True,
        )
load_css()

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


def ll_style_fig(fig, accent=None, height=None):
    """Re-skin an EXISTING Plotly figure for the dark theme (colours only)."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color="#B9B5DC"),
        title_font=dict(color="#F4F2FF", size=15),
        hoverlabel=dict(bgcolor="#17163A", font_color="#F4F2FF"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", linecolor="rgba(255,255,255,0.10)",
                     zerolinecolor="rgba(255,255,255,0.10)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", linecolor="rgba(255,255,255,0.10)",
                     zerolinecolor="rgba(255,255,255,0.10)")
    if accent:
        fig.update_traces(line_color=accent, marker_color=accent)
    if height:
        fig.update_layout(height=height)
    return fig


def ll_style_radar(fig):
    ll_style_fig(fig)
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(gridcolor="rgba(255,255,255,0.10)", linecolor="rgba(255,255,255,0.10)",
                            tickfont=dict(color="#8F8BB8")),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.10)", linecolor="rgba(255,255,255,0.10)",
                             tickfont=dict(color="#F4F2FF")),
        )
    )
    fig.update_traces(line_color="#A78BFA", fillcolor="rgba(139, 92, 246, 0.28)",
                      marker_color="#F472B6", selector=dict(type="scatterpolar"))
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
    Split the existing NLP report into its own headings for display.
    Returns [(heading, body), ...]; empty list if headings are not found
    (the caller then shows the untouched raw text).
    """
    cleaned = re.sub(r"\*\*", "", str(text))
    pattern = re.compile(
        r"(?:^|\s)(?:\d\.\s*)?(OVERALL|KEY FACTOR|PATTERN|PRIORITY)\s*:\s*",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(cleaned))
    sections = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
        body = re.sub(r"\s*\d\.\s*$", "", cleaned[m.end():end].strip())
        if body:
            sections.append((m.group(1).title(), body))
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


st.markdown(
    """
    <a href="#top" class="lifelens-back-to-top" title="Back to top">
        ↑
    </a>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div id="top"></div>', unsafe_allow_html=True)

CSV_FILE = "data/daily_checkins.csv"

today = datetime.now()

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

if not data.empty:
    data = data.copy()
    if "Date" in data.columns:
        normalized_dates = normalize_date_series(data["Date"])
        data["Date"] = pd.to_datetime(normalized_dates, errors="coerce")
        data = data.dropna(subset=["Date"]).sort_values("Date")
        data = data.drop_duplicates(subset=["Date"], keep="last")

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




# Set safe defaults so the dashboard can handle an empty dataset.
wellness_score = None
wellness_category = "Insufficient data"
score_breakdown = []
radar_chart = create_wellness_radar_chart(score_breakdown)

summary = None
insights = []
trend_analysis = None
ml_prediction = None
nlp_wellness_report = None
weekly_facts = {}
weekly_ai_analysis = None
forecast_explanation = None


if not data.empty:
    latest = data.iloc[-1]

    wellness_score, wellness_category = calculate_wellness_score(
        latest["SleepHours"],
        latest["Mood"],
        latest["Stress"],
        latest["Energy"],
        latest["ExerciseMinutes"],
        latest["ScreenTime"],
        latest["Productivity"],
    )
    
    score_factors = [
        latest["SleepHours"],
        latest["Mood"],
        latest["Stress"],
        latest["Energy"],
        latest["ExerciseMinutes"],
        latest["ScreenTime"],
        latest["Productivity"],
    ]

    available_factors = sum(
        1 for value in score_factors
        if _valid_number(value) is not None
    )


    score_breakdown = get_score_breakdown(
        latest["SleepHours"],
        latest["Mood"],
        latest["Stress"],
        latest["Energy"],
        latest["ExerciseMinutes"],
        latest["ScreenTime"],
        latest["Productivity"],
    )

    radar_chart = create_wellness_radar_chart(
    score_breakdown,
    recorded_values={
        "SleepHours": latest["SleepHours"],
        "Mood": latest["Mood"],
        "Stress": latest["Stress"],
        "Energy": latest["Energy"],
        "ExerciseMinutes": latest["ExerciseMinutes"],
        "ScreenTime": latest["ScreenTime"],
        "StudyHours": latest["StudyHours"],
        "Productivity": latest["Productivity"],
    },
)

    summary = get_summary(data)
    insights = generate_insights(data)
    trend_analysis = generate_trend_analysis(data)
    nlp_wellness_report = generate_wellness_report(
        wellness_score,
        wellness_category,
        latest,
        history=data,
        daily_note=latest.get("DailyNote", ""),
        
    )


if not data.empty:
    recommendations = generate_recommendations(data)
    patterns = analyze_patterns(data)

    # Existing weekly report remains unchanged
    weekly_report = generate_weekly_report(data)

    # Python calculates factual weekly statistics
    weekly_facts = calculate_weekly_facts(data)

    # Local NLP model explains those facts
    weekly_ai_analysis = explain_weekly_facts(
        weekly_facts
    )

    monthly_report = generate_monthly_report(data)
    correlation_matrix = get_correlation_matrix(data)

    # =========================================================
    # TOMORROW'S FORECAST (BASE + ML)
    # =========================================================
    prediction = {}

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

    forecast_explanation = None
    if prediction:
        try:
            forecast_explanation = explain_tomorrow_forecast(
                prediction,
                latest=latest,
                history=data,
                daily_note=latest.get("DailyNote", "")
            )
        except Exception as error:
            print("Forecast explanation unavailable:", repr(error))
            forecast_explanation = None

    biggest_discovery = get_biggest_discovery(data)
    next_step = get_todays_focus(data)
    today_focus = get_todays_focus(data)
today = datetime.now()

selected_tab = show_navigation()

if selected_tab == "📝 Daily Check-in":
    st.html(
        """
        <div class="ll-page-head">
            <span class="ll-eyebrow">⏱ Daily Check-in · ~1 min</span>
            <h1>Daily Check-in</h1>
            <p>Take a minute to reflect and keep your wellness picture up to date.</p>
        </div>
        """
    )

    if st.session_state.get("checkin_success_message"):
        st.success(st.session_state.pop("checkin_success_message"))

    with st.form("daily_checkin"):

        # Date: automatic current date (read-only display)
        checkin_date = today.date()
        date = checkin_date
        formatted_date_display = checkin_date.strftime("%B %d, %Y").replace(" 0", " ")

        st.html(f"""
            <div class="ll-date-chip">
                <span class="muted">📅 Check-in date</span>
                <b>{formatted_date_display}</b>
                <span class="tag">Today</span>
            </div>
        """)

        # CARD 1 — 🌱 DAILY HABITS (Compact 2x2 slider grid)
        with st.container(border=True, key="daily_habits_panel"):
            st.html(
                """
                <div class="ll-panel-head">
                    <h3>🌱 Daily Habits</h3>
                    <span>Track the behaviours that shaped your day.</span>
                </div>
                """
            )

            h_row1_col1, h_row1_col2 = st.columns(2, gap="medium")
            with h_row1_col1:
                sleep_hours = st.slider(
                    "😴 Hours of sleep",
                    min_value=0.0,
                    max_value=14.0,
                    value=7.0,
                    step=0.5,
                    format="%.1f hrs",
                    help="How long did you sleep in total?"
                )
            with h_row1_col2:
                screen_time = st.slider(
                    "💻 Screen time",
                    min_value=0.0,
                    max_value=24.0,
                    value=4.0,
                    step=0.5,
                    format="%.1f hrs",
                    help="Total recreational or non-work screen time"
                )

            h_row2_col1, h_row2_col2 = st.columns(2, gap="medium")
            with h_row2_col1:
                exercise_minutes = st.slider(
                    "🏃 Exercise",
                    min_value=0,
                    max_value=1440,
                    value=30,
                    step=5,
                    format="%d min",
                    help="Total active movement or workout time"
                )
            with h_row2_col2:
                study_hours = st.slider(
                    "📚 Study / work",
                    min_value=0.0,
                    max_value=24.0,
                    value=4.0,
                    step=0.5,
                    format="%.1f hrs",
                    help="Time spent studying or focused work"
                )

        # CARD 2 — 💭 PERSONAL REFLECTIONS (Compact 2x2 dropdown grid)
        with st.container(border=True, key="personal_reflections_panel"):
            st.html(
                """
                <div class="ll-panel-head">
                    <h3>💭 Personal Reflections</h3>
                    <span>How did you feel today? (Optional descriptions)</span>
                </div>
                """
            )

            r_row1_col1, r_row1_col2 = st.columns(2, gap="medium")
            with r_row1_col1:
                mood_choice = st.selectbox(
                    "😊 Mood",
                    options=[
                        "Prefer not to answer",
                        "Very low",
                        "A little low",
                        "Mixed or okay",
                        "Mostly positive",
                        "Very positive"
                    ],
                    index=0
                )
            with r_row1_col2:
                energy_choice = st.selectbox(
                    "⚡ Energy",
                    options=[
                        "Prefer not to answer",
                        "Very low",
                        "Low",
                        "Moderate",
                        "High",
                        "Very high"
                    ],
                    index=0
                )

            r_row2_col1, r_row2_col2 = st.columns(2, gap="medium")
            with r_row2_col1:
                stress_choice = st.selectbox(
                    "😌 Stress",
                    options=[
                        "Prefer not to answer",
                        "Very low",
                        "Mild",
                        "Moderate",
                        "High",
                        "Very high"
                    ],
                    index=0
                )
            with r_row2_col2:
                productivity_choice = st.selectbox(
                    "🎯 Productivity",
                    options=[
                        "Prefer not to answer",
                        "Very little progress",
                        "A little progress",
                        "Reasonable progress",
                        "Good progress",
                        "Excellent progress"
                    ],
                    index=0
                )

        # CARD 3 — OPTIONAL REFLECTION
        with st.container(key="optional_reflection_panel"):
            st.html(
                """
                <div class="ll-panel-head">
                    <h3>📝 What influenced your day?</h3>
                    <span>Optional · add context that numbers and ratings cannot capture.</span>
                </div>
                """
            )

            daily_note = st.text_area(
                "What influenced your day?",
                label_visibility="collapsed",
                height=70,
                placeholder=(
                    "For example: an exam, a busy day, "
                    "a good night's sleep, or a walk outdoors."
                ),
                help="This note is optional and is not used in calculations yet."
            )

        submitted = st.form_submit_button(
            "✓ Save Check-in",
            use_container_width=True,
            type="primary"
        )


        if submitted:

            # Convert descriptive choices to the existing numeric scale.
            # Missing answers remain None rather than being treated as zero.
            mood_map = {
                "Very low": 2,
                "A little low": 4,
                "Mixed or okay": 6,
                "Mostly positive": 8,
                "Very positive": 10
            }

            energy_map = {
                "Very low": 2,
                "Low": 4,
                "Moderate": 6,
                "High": 8,
                "Very high": 10
            }

            stress_map = {
                "Very low": 1,
                "Mild": 3,
                "Moderate": 5,
                "High": 8,
                "Very high": 10
            }

            productivity_map = {
                "Very little progress": 2,
                "A little progress": 4,
                "Reasonable progress": 6,
                "Good progress": 8,
                "Excellent progress": 10
            }

            mood = mood_map.get(mood_choice)
            energy = energy_map.get(energy_choice)
            stress = stress_map.get(stress_choice)
            productivity = productivity_map.get(productivity_choice)

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
                "DailyNote": daily_note.strip() or None
            }])

            existing_data = load_data()

            date_string = pd.to_datetime(date).strftime("%Y-%m-%d")

            if not existing_data.empty:
                existing_dates = normalize_date_series(existing_data["Date"])
            else:
                existing_dates = pd.Series(dtype="object")

            if date_string in existing_dates.values:
                st.session_state["pending_overwrite"] = new_entry
                confirm_overwrite(new_entry, date_string)
            else:
                try:
                    save_data(new_entry)

                    data = load_data()

                    st.success("Your daily check-in has been saved.")
                    st.subheader("Today's Entry")
                    st.dataframe(new_entry, use_container_width=True)
                except Exception as e:
                    st.error(f"Unable to save check-in: {e}")

    
    with st.expander("View Raw Data"):
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
    # 2. METRIC CARDS (real values from the latest check-in;
    #    status + bar come from the existing get_score_breakdown)
    # ==========================================
    ll_section("📊", "Today's Signals", "From your latest check-in")

    breakdown_lookup = {name: (score, status) for name, score, status in score_breakdown}
    previous = data.iloc[-2] if len(data) >= 2 else None

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
        current_value = latest.get(column)
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
            report_sections = ll_parse_report(nlp_wellness_report)
            if report_sections:
                report_items = "".join(
                    f'<div class="ll-report-item"><div class="ll-report-head">{_esc(head)}</div>'
                    f'<div class="ll-report-text">{_esc(body)}</div></div>'
                    for head, body in report_sections
                )
                report_html = f'<div class="ll-report">{report_items}</div>'
            else:
                report_html = f'<div class="ll-report-raw">{_esc(str(nlp_wellness_report).strip())}</div>'
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
                    forecast_facts = _forecast_facts(prediction, latest)
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

    # Forecast explanation (existing explain_tomorrow_forecast output)
    if prediction and forecast_explanation:
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
    ll_section("🧭", "Wellness Breakdown", "Factor scores from your latest check-in")

    with st.expander("🔍 View Wellness Radar & Area Breakdown", expanded=False):
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
        for line in weekly_ai_analysis.split("\n"):
            line = line.strip()

            if not line:
                continue

            if line.startswith("WEEKLY OVERVIEW") or "WEEKLY SUMMARY" in line:
                weekly_sections.append(["📊 Weekly Overview", []])
            elif line.startswith("KEY TREND") or "METRIC COMPARISONS" in line:
                weekly_sections.append(["📈 Key Trend", []])
            elif line.startswith("MAIN FACTOR") or "ROUTINE CHANGES" in line:
                weekly_sections.append(["🎯 Main Factor", []])
            elif line.startswith("NEXT WEEK PRIORITY") or "NEXT WEEK" in line:
                weekly_sections.append(["🚀 Next Week Priority", []])
            else:
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

if selected_tab == "ℹ️ About":
    show_about()
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

show_hero(greeting, today)
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
    st.markdown('''
        <div style="margin-bottom: 12px;">
            <div style="display: inline-flex; align-items: center; gap: 6px; padding: 2px 9px; border-radius: 999px; background: rgba(37, 99, 235, 0.08); border: 1px solid rgba(37, 99, 235, 0.2); color: #2563EB; font-size: 11px; font-weight: 700; letter-spacing: 0.03em; text-transform: uppercase; margin-bottom: 4px;">
                <span>⏱</span> Daily Check-in · ~1 min
            </div>
            <h1 style="margin: 0 0 2px 0; font-size: 24px; font-weight: 800; color: #0F172A; letter-spacing: -0.02em;">Daily Check-in</h1>
            <p style="margin: 0; font-size: 13px; color: #475569; line-height: 1.4;">Take a minute to reflect on your day and keep your wellness picture up to date.</p>
        </div>
    ''', unsafe_allow_html=True)

    if st.session_state.get("checkin_success_message"):
        st.success(st.session_state.pop("checkin_success_message"))

    with st.form("daily_checkin"):

        # Date: automatic current date (read-only display)
        checkin_date = today.date()
        date = checkin_date
        formatted_date_display = checkin_date.strftime("%B %d, %Y").replace(" 0", " ")

        st.markdown(f'''
            <div style="margin-bottom: 12px; display: inline-flex; align-items: center; gap: 8px; background: #F8FAFC; border: 1px solid #E2E8F0; padding: 6px 14px; border-radius: 8px;">
                <span style="font-size: 13px; font-weight: 600; color: #64748B;">📅 Check-in date:</span>
                <span style="font-size: 13.5px; font-weight: 700; color: #1E293B;">{formatted_date_display}</span>
                <span style="font-size: 11px; font-weight: 600; color: #2563EB; background: rgba(37, 99, 235, 0.08); padding: 1px 7px; border-radius: 999px;">Today</span>
            </div>
        ''', unsafe_allow_html=True)

        # CARD 1 — 🌱 DAILY HABITS (Compact 2x2 slider grid)
        with st.container(border=True, key="daily_habits_panel"):
            st.markdown('''
                <div style="margin-bottom: 8px;">
                    <h3 style="margin: 0 0 2px 0; font-size: 16px; font-weight: 700; color: #1E293B;">🌱 Daily Habits</h3>
                    <div style="font-size: 12px; color: #64748B;">Track the behaviours that shaped your day.</div>
                </div>
            ''', unsafe_allow_html=True)

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
            st.markdown('''
                <div style="margin-bottom: 8px;">
                    <h3 style="margin: 0 0 2px 0; font-size: 16px; font-weight: 700; color: #1E293B;">💭 Personal Reflections</h3>
                    <div style="font-size: 12px; color: #64748B;">How did you feel today? (Optional descriptions)</div>
                </div>
            ''', unsafe_allow_html=True)

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
            st.markdown('''
                <div style="margin-top: 4px; margin-bottom: 4px;">
                    <div style="font-size: 13px; font-weight: 600; color: #334155;">💭 What influenced your day?</div>
                    <div style="font-size: 11px; color: #64748B;">Optional reflection · Add context that numbers and ratings cannot capture.</div>
                </div>
            ''', unsafe_allow_html=True)

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
        st.markdown("## 🌱 Welcome to LifeLens")
        st.markdown(
            "Your personal wellness journey starts with your first check-in."
        )
        st.info(
            "📋 No check-in data has been recorded yet. "
            "Go to **📝 Daily Check-in** and complete your first entry "
            "to unlock your personalised dashboard."
        )
        st.stop()

    # ==========================================
    # 1. COMPACT DASHBOARD HEADER
    # ==========================================
    st.markdown(
        """
        <div style="margin-bottom: 16px;">
            <div style="font-size: 24px; font-weight: 800; color: #0F172A; letter-spacing: -0.025em; line-height: 1.2;">
                Your Wellness Snapshot
            </div>
            <div style="font-size: 13.5px; color: #64748B; margin-top: 2px;">
                Here's how you're doing today.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ==========================================
    # 2 & 3. COMMAND CENTRE HERO CARDS
    # Today's Wellness Score (Strongest) + Tomorrow's Forecast (Second)
    # ==========================================
    col_today, col_tomorrow = st.columns([1.18, 1], gap="medium")

    with col_today:
        score_display = f"{round(wellness_score)}" if wellness_score is not None else "--"
        cat_display = wellness_category if wellness_score is not None else "Insufficient Data"
        
        if wellness_score is not None:
            if wellness_score >= 80:
                score_msg = "Outstanding balance across sleep, habits, and mindset."
                badge_bg = "#DCFCE7"
                badge_text = "#15803D"
            elif wellness_score >= 65:
                score_msg = "Solid overall wellbeing with steady daily rhythm."
                badge_bg = "#DBEAFE"
                badge_text = "#1D4ED8"
            elif wellness_score >= 50:
                score_msg = "Moderate balance with room for targeted improvements."
                badge_bg = "#FEF3C7"
                badge_text = "#B45309"
            else:
                score_msg = "Key areas need attention to help you recover energy."
                badge_bg = "#FEE2E2"
                badge_text = "#B91C1C"
        else:
            score_msg = "Add your daily check-in to generate your wellness score."
            badge_bg = "#F1F5F9"
            badge_text = "#475569"

        st.html(
            f"""
            <div style="
                background: linear-gradient(135deg, #0F172A 0%, #1E293B 55%, #1E3A8A 100%);
                border: 1px solid #334155;
                border-radius: 14px;
                padding: 22px 24px;
                color: #FFFFFF;
                min-height: 220px;
                box-shadow: 0 4px 18px rgba(15, 23, 42, 0.12);
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                box-sizing: border-box;
            ">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <span style="font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #94A3B8;">
                            🌱 Today's Wellness
                        </span>
                        <span style="background: {badge_bg}; color: {badge_text}; padding: 3px 10px; border-radius: 9999px; font-size: 12px; font-weight: 700; letter-spacing: 0.01em;">
                            {cat_display}
                        </span>
                    </div>
                    <div style="display: flex; align-items: baseline; gap: 4px; margin-bottom: 6px;">
                        <span style="font-size: 48px; font-weight: 800; line-height: 1; color: #F8FAFC; letter-spacing: -0.03em;">
                            {score_display}
                        </span>
                        <span style="font-size: 20px; font-weight: 600; color: #64748B;">
                            /100
                        </span>
                    </div>
                </div>
                <div>
                    <div style="font-size: 13.5px; color: #CBD5E1; line-height: 1.45; font-weight: 450;">
                        {score_msg}
                    </div>
                    <div style="font-size: 12px; color: #94A3B8; margin-top: 5px;">
                        Based on {available_factors} of 7 recorded factors
                    </div>
                </div>
            </div>
            """
        )

    with col_tomorrow:
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
            
            p_sleep = (
                f"{prediction['SleepHours']}h"
                if prediction.get("SleepHours") is not None
                else "--"
            )

            p_mood = (
                f"{prediction['Mood']}/10"
                if prediction.get("Mood") is not None
                else "--"
            )

            p_stress = (
                f"{prediction['Stress']}/10"
                if prediction.get("Stress") is not None
                else "--"
            )

            p_screen = (
                f"{prediction['ScreenTime']}h"
                if prediction.get("ScreenTime") is not None
                else "--"
            )

            p_energy = (
                f"{prediction['Energy']}/10"
                if prediction.get("Energy") is not None
                else "--"
            )

            p_exercise = (
                f"{prediction['ExerciseMinutes']} min"
                if prediction.get("ExerciseMinutes") is not None
                else "--"
            )

            p_productivity = (
                f"{prediction['Productivity']}/10"
                if prediction.get("Productivity") is not None
                else "--"
            )

            st.html(
                f"""
                <div style="
                    background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
                    border: 1px solid #4338CA;
                    border-radius: 14px;
                    padding: 22px 24px;
                    color: #FFFFFF;
                    min-height: 220px;
                    box-shadow: 0 4px 18px rgba(67, 56, 202, 0.12);
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                    box-sizing: border-box;
                ">
                    <div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <span style="font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #C7D2FE;">
                                🔮 Tomorrow's Forecast
                            </span>
                            <span style="background: rgba(255, 255, 255, 0.14); color: #EEF2FF; padding: 3px 10px; border-radius: 9999px; font-size: 12px; font-weight: 600;">
                                {forecast_summary}
                            </span>
                        </div>
                        <div style="display: flex; align-items: baseline; gap: 5px; margin-bottom: 2px;">
                            <span style="font-size: 38px; font-weight: 800; line-height: 1; color: #FFFFFF; letter-spacing: -0.02em;">
                                {t_score_str}
                            </span>
                            <span style="font-size: 16px; font-weight: 600; color: #A5B4FC;">
                                /100 projected
                            </span>
                        </div>
                        <div style="font-size: 12.5px; font-weight: 700; color: #C7D2FE; margin-bottom: 6px;">
                            {t_cat}
                        </div>
                        <div style="font-size: 13px; color: #E0E7FF; line-height: 1.4; font-weight: 450;">
                            AI projection based on your recent behavioural momentum.
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(7, 1fr); gap: 6px; margin-top: 10px; background: rgba(0, 0, 0, 0.22); padding: 7px 8px; border-radius: 9px;">
                        <div style="text-align: center;">
                            <div style="font-size: 11px; color: #C7D2FE;">Sleep</div>
                            <div style="font-size: 13px; font-weight: 700; color: #FFFFFF;">{p_sleep}</div>
                        </div>

                        <div style="text-align: center;">
                            <div style="font-size: 11px; color: #C7D2FE;">Mood</div>
                            <div style="font-size: 13px; font-weight: 700; color: #FFFFFF;">{p_mood}</div>
                        </div>

                        <div style="text-align: center;">
                            <div style="font-size: 11px; color: #C7D2FE;">Stress</div>
                            <div style="font-size: 13px; font-weight: 700; color: #FFFFFF;">{p_stress}</div>
                        </div>

                        <div style="text-align: center;">
                            <div style="font-size: 11px; color: #C7D2FE;">Energy</div>
                            <div style="font-size: 13px; font-weight: 700; color: #FFFFFF;">{p_energy}</div>
                        </div>

                        <div style="text-align: center;">
                            <div style="font-size: 11px; color: #C7D2FE;">Exercise</div>
                            <div style="font-size: 13px; font-weight: 700; color: #FFFFFF;">{p_exercise}</div>
                        </div>

                        <div style="text-align: center;">
                            <div style="font-size: 11px; color: #C7D2FE;">Screen</div>
                            <div style="font-size: 13px; font-weight: 700; color: #FFFFFF;">{p_screen}</div>
                        </div>

                        <div style="text-align: center;">
                            <div style="font-size: 11px; color: #C7D2FE;">Productivity</div>
                            <div style="font-size: 13px; font-weight: 700; color: #FFFFFF;">{p_productivity}</div>
                        </div>
                    </div>
                </div>
                """
            )
        else:
            st.html(
                """
                <div style="
                    background: #F8FAFC;
                    border: 1px dashed #CBD5E1;
                    border-radius: 14px;
                    padding: 22px 24px;
                    min-height: 220px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    text-align: center;
                    box-sizing: border-box;
                ">
                    <div style="font-size: 28px; margin-bottom: 6px;">🔮</div>
                    <div style="font-size: 15px; font-weight: 700; color: #1E293B; margin-bottom: 3px;">
                        Tomorrow's Forecast
                    </div>
                    <div style="font-size: 13px; color: #64748B; max-width: 270px; line-height: 1.45;">
                        Log check-ins for a few consecutive days to unlock your AI predictive forecast.
                    </div>
                </div>
                """
            )

    # ==========================================
    # 🔮 TOMORROW FORECAST AI EXPLANATION
    # ==========================================
    if prediction and forecast_explanation:
        outlook_text = forecast_explanation.get("Tomorrow Outlook", "")
        factor_text = forecast_explanation.get("Main Factor", "")
        pattern_text = forecast_explanation.get("Pattern", "")
        priority_text = forecast_explanation.get("Priority", "")

        st.html(
            f"""
            <div style="
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-left: 4px solid #6366F1;
                border-radius: 12px;
                padding: 16px 20px;
                margin-top: 14px;
                margin-bottom: 14px;
                box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
            ">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                    <span style="font-size: 16px;">🔮</span>
                    <span style="font-size: 14px; font-weight: 700; color: #1E293B; letter-spacing: -0.01em;">
                        Tomorrow Forecast Analysis
                    </span>
                    <span style="font-size: 11px; background: #EEF2FF; color: #4F46E5; padding: 2px 8px; border-radius: 999px; font-weight: 600;">
                        AI Explained
                    </span>
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px;">
                    <div style="background: #F8FAFC; padding: 12px 14px; border-radius: 8px; border: 1px solid #F1F5F9;">
                        <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #4F46E5; margin-bottom: 4px;">
                            Tomorrow Outlook
                        </div>
                        <div style="font-size: 13px; color: #334155; line-height: 1.45;">
                            {outlook_text}
                        </div>
                    </div>
                    <div style="background: #F8FAFC; padding: 12px 14px; border-radius: 8px; border: 1px solid #F1F5F9;">
                        <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #0284C7; margin-bottom: 4px;">
                            Main Factor
                        </div>
                        <div style="font-size: 13px; color: #334155; line-height: 1.45;">
                            {factor_text}
                        </div>
                    </div>
                    <div style="background: #F8FAFC; padding: 12px 14px; border-radius: 8px; border: 1px solid #F1F5F9;">
                        <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #D97706; margin-bottom: 4px;">
                            Pattern
                        </div>
                        <div style="font-size: 13px; color: #334155; line-height: 1.45;">
                            {pattern_text}
                        </div>
                    </div>
                    <div style="background: #F8FAFC; padding: 12px 14px; border-radius: 8px; border: 1px solid #F1F5F9;">
                        <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #059669; margin-bottom: 4px;">
                            Priority
                        </div>
                        <div style="font-size: 13px; color: #334155; line-height: 1.45;">
                            {priority_text}
                        </div>
                    </div>
                </div>
            </div>
            """
        )
    else:
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # ==========================================
    # 🧠 AI WELLNESS REPORT (NLP Model)
    # ==========================================
    st.markdown(
        """
        <div style="font-size: 17px; font-weight: 700; color: #0F172A; letter-spacing: -0.01em; margin-bottom: 2px;">
            🧠 Your Wellness Report
        </div>
        """,
        unsafe_allow_html=True
    )
    st.caption("AI-generated narrative summary based on your latest wellness metrics.")

    if nlp_wellness_report:
        formatted_report = str(nlp_wellness_report).strip()
        for marker in ["2.", "3.", "4."]:
            formatted_report = re.sub(rf"\s+(?={re.escape(marker)})", "\n\n", formatted_report)
        st.html(
            f"""
            <div style="
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-left: 4px solid #6366F1;
                border-radius: 12px;
                padding: 16px 20px;
                color: #1E293B;
                box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
                line-height: 1.6;
                font-size: 14px;
                white-space: pre-line;
            ">
                {formatted_report}
            </div>
            """
        )
    else:
        st.info("The AI Wellness Report is temporarily unavailable.")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # ==========================================
    # 4. WELLNESS BREAKDOWN SECTION
    # ==========================================
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;">
            <div style="font-size: 17px; font-weight: 700; color: #0F172A; letter-spacing: -0.01em;">
                📊 Wellness Breakdown
            </div>
            <div style="font-size: 12.5px; color: #64748B;">
                Current factor status from your latest check-in
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    factor_items = [
        ("Sleep", f"{latest['SleepHours']} hrs" if _valid_number(latest.get("SleepHours")) is not None else "--", "😴", "#6366F1", "#EEF2FF"),
        ("Mood", f"{latest['Mood']}/10" if _valid_number(latest.get("Mood")) is not None else "--", "😊", "#F59E0B", "#FEF3C7"),
        ("Stress", f"{latest['Stress']}/10" if _valid_number(latest.get("Stress")) is not None else "--", "😌", "#EF4444", "#FEE2E2"),
        ("Energy", f"{latest['Energy']}/10" if _valid_number(latest.get("Energy")) is not None else "--", "⚡", "#10B981", "#D1FAE5"),
        ("Exercise", f"{int(latest['ExerciseMinutes'])} min" if _valid_number(latest.get("ExerciseMinutes")) is not None else "--", "🏃", "#06B6D4", "#CFFAFE"),
        ("Screen Time", f"{latest['ScreenTime']} hrs" if _valid_number(latest.get("ScreenTime")) is not None else "--", "💻", "#8B5CF6", "#EDE9FE"),
    ]

    breakdown_cols = st.columns(6, gap="small")
    for col, (f_name, f_val, f_icon, f_color, f_bg) in zip(breakdown_cols, factor_items):
        with col:
            st.html(
                f"""
                <div style="
                    background: #FFFFFF;
                    border: 1px solid #E2E8F0;
                    border-radius: 12px;
                    padding: 12px 10px;
                    text-align: center;
                    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
                ">
                    <div style="
                        width: 28px;
                        height: 28px;
                        border-radius: 7px;
                        background: {f_bg};
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        margin: 0 auto 5px auto;
                        font-size: 14px;
                    ">
                        {f_icon}
                    </div>
                    <div style="font-size: 11px; font-weight: 600; color: #64748B; margin-bottom: 2px;">
                        {f_name}
                    </div>
                    <div style="font-size: 14.5px; font-weight: 750; color: #0F172A;">
                        {f_val}
                    </div>
                </div>
                """
            )

    with st.expander("🔍 View Wellness Radar & Area Breakdown", expanded=False):
        radar_col, areas_col = st.columns([1.3, 1], gap="medium")
        with radar_col:
            st.plotly_chart(
                radar_chart,
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

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # ==========================================
    # 5. TODAY'S BIGGEST DISCOVERY
    # Dark premium glass-style insight card
    # ==========================================
    st.html(
        f"""
        <div style="
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
            border: 1px solid #334155;
            border-left: 4px solid #6366F1;
            border-radius: 12px;
            padding: 12px 18px;
            display: flex;
            align-items: center;
            gap: 14px;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.12);
        ">
            <div style="font-size: 20px; line-height: 1;">💡</div>
            <div style="flex: 1;">
                <div style="font-size: 10.5px; font-weight: 750; text-transform: uppercase; letter-spacing: 0.08em; color: #818CF8; margin-bottom: 2px;">
                    Key Behavioral Discovery
                </div>
                <div style="font-size: 13.5px; font-weight: 500; color: #F1F5F9; line-height: 1.4;">
                    {biggest_discovery}
                </div>
            </div>
        </div>
        """
    )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # ==========================================
    # 6. RECOMMENDED ACTION PLAN
    # Focus -> Why -> What to do
    # Left: Primary Daily Focus (WHAT) | Right: AI Insight (WHY) + Recommended Action (HOW)
    # ==========================================
    st.markdown(
        """
        <div style="font-size: 17px; font-weight: 700; color: #0F172A; margin-bottom: 8px; letter-spacing: -0.01em;">
            🎯 Recommended Action Plan
        </div>
        """,
        unsafe_allow_html=True
    )

    top_recs = [r for r in recommendations if r != today_focus][:3]
    action_col_today, action_col_prep = st.columns([1.18, 1], gap="medium")

    with action_col_today:
        supporting_items = "".join([
            f'<div style="font-size: 12.5px; color: #334155; display: flex; align-items: baseline; gap: 7px; line-height: 1.35;"><span style="color: #059669; font-weight: 700;">✓</span><span>{rec}</span></div>'
            for rec in top_recs
        ])
        st.html(
            f"""
            <div style="
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 12px;
                padding: 14px 16px;
                box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
                box-sizing: border-box;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                min-height: 154px;
            ">
                <div>
                    <div style="font-size: 10.5px; font-weight: 750; text-transform: uppercase; letter-spacing: 0.08em; color: #059669; margin-bottom: 4px;">
                        🎯 Primary Daily Focus
                    </div>
                    <div style="font-size: 14px; font-weight: 650; color: #0F172A; margin-bottom: 10px; line-height: 1.35;">
                        {today_focus}
                    </div>
                </div>
                <div>
                    <div style="font-size: 10.5px; font-weight: 750; text-transform: uppercase; letter-spacing: 0.06em; color: #64748B; margin-bottom: 5px;">
                        Key Supporting Habits
                    </div>
                    <div style="display: flex; flex-direction: column; gap: 5px;">
                        {supporting_items}
                    </div>
                </div>
            </div>
            """
        )

    with action_col_prep:
        if prediction:
            ai_forecast = generate_ai_forecast_insight(prediction)
            st.html(
                f"""
                <div style="display: flex; flex-direction: column; gap: 8px;">
                    <div style="
                        background: #F8FAFC;
                        border: 1px solid #E2E8F0;
                        border-left: 4px solid #3B82F6;
                        border-radius: 10px;
                        padding: 10px 14px;
                        box-sizing: border-box;
                    ">
                        <div style="font-size: 10px; font-weight: 750; text-transform: uppercase; letter-spacing: 0.07em; color: #2563EB; margin-bottom: 2px;">
                            💡 AI Insight — Why This Matters
                        </div>
                        <div style="font-size: 12.5px; color: #334155; line-height: 1.4;">
                            {ai_forecast}
                        </div>
                    </div>
                    <div style="
                        background: #F8FAFC;
                        border: 1px solid #E2E8F0;
                        border-left: 4px solid #0D9488;
                        border-radius: 10px;
                        padding: 10px 14px;
                        box-sizing: border-box;
                    ">
                        <div style="font-size: 10px; font-weight: 750; text-transform: uppercase; letter-spacing: 0.07em; color: #0D9488; margin-bottom: 2px;">
                            ⚡ Recommended Action — What to Do
                        </div>
                        <div style="font-size: 12.5px; color: #334155; line-height: 1.4;">
                            Reduce screen time before bed while maintaining your current sleep schedule to support tomorrow's energy and recovery.
                        </div>
                    </div>
                </div>
                """
            )
        else:
            st.html(
                """
                <div style="
                    background: #F8FAFC;
                    border: 1px dashed #CBD5E1;
                    border-radius: 12px;
                    padding: 16px;
                    min-height: 154px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    text-align: center;
                    box-sizing: border-box;
                ">
                    <div style="font-size: 13.5px; font-weight: 600; color: #475569; margin-bottom: 4px;">
                        Forward Preparation Guidance
                    </div>
                    <div style="font-size: 12px; color: #64748B; max-width: 250px; line-height: 1.4;">
                        Log consistent check-ins to receive forward-looking preparation guidance.
                    </div>
                </div>
                """
            )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # ==========================================
    # 7. BEHAVIOUR TRENDS
    # Compact Balanced Layout + Segmented Tab Control + Sized Charts
    # ==========================================
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;">
            <div style="font-size: 17px; font-weight: 700; color: #0F172A; letter-spacing: -0.01em;">
                📈 Behaviour Trends
            </div>
            <div style="font-size: 12.5px; color: #64748B;">
                7-day trajectory and habit consistency
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    trend_sum_col, trend_charts_col = st.columns([1, 1.65], gap="medium")

    with trend_sum_col:
        trend_pos = trend_analysis.get("positive", "Keep logging to unlock trend analysis.") if trend_analysis else "Keep logging."
        trend_watch = trend_analysis.get("watch", "No major changes to monitor.") if trend_analysis else "No data."
        trend_risk = trend_analysis.get("risk", "No significant risks detected.") if trend_analysis else "No data."
        trend_rec = trend_analysis.get("recommendation", "Maintain routine.") if trend_analysis else "Maintain routine."

        st.html(
            f"""
            <div style="
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 12px;
                padding: 14px 16px;
                box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
            ">
                <div style="font-size: 11px; font-weight: 750; color: #1E293B; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.06em;">
                    🧠 AI Trend Assessment
                </div>
                <div style="display: flex; flex-direction: column; gap: 7px;">
                    <div style="background: #F0FDF4; border-left: 3px solid #22C55E; padding: 7px 10px; border-radius: 6px;">
                        <div style="font-size: 10px; font-weight: 750; color: #16A34A; text-transform: uppercase; letter-spacing: 0.04em;">Positive Shift</div>
                        <div style="font-size: 12px; color: #166534; margin-top: 1px; line-height: 1.35;">{trend_pos}</div>
                    </div>
                    <div style="background: #FEF3C7; border-left: 3px solid #F59E0B; padding: 7px 10px; border-radius: 6px;">
                        <div style="font-size: 10px; font-weight: 750; color: #D97706; text-transform: uppercase; letter-spacing: 0.04em;">Area to Watch</div>
                        <div style="font-size: 12px; color: #92400E; margin-top: 1px; line-height: 1.35;">{trend_watch}</div>
                    </div>
                    <div style="background: #EFF6FF; border-left: 3px solid #3B82F6; padding: 7px 10px; border-radius: 6px;">
                        <div style="font-size: 10px; font-weight: 750; color: #2563EB; text-transform: uppercase; letter-spacing: 0.04em;">Guidance</div>
                        <div style="font-size: 12px; color: #1E40AF; margin-top: 1px; line-height: 1.35;">{trend_rec}</div>
                    </div>
                </div>
            </div>
            """
        )

    with trend_charts_col:
        with st.container(key="trends_tabs"):
            sleep_tab, mood_tab, stress_tab, screen_tab = st.tabs([
                "😴 Sleep",
                "😊 Mood",
                "😌 Stress",
                "📱 Screen"
            ])
            with sleep_tab:
                fig_sleep = plot_sleep_trend(data)
                fig_sleep.update_layout(height=245, margin=dict(l=20, r=20, t=35, b=25))
                st.plotly_chart(fig_sleep, use_container_width=True)
            with mood_tab:
                fig_mood = plot_mood_trend(data)
                fig_mood.update_layout(height=245, margin=dict(l=20, r=20, t=35, b=25))
                st.plotly_chart(fig_mood, use_container_width=True)
            with stress_tab:
                fig_stress = plot_stress_trend(data)
                fig_stress.update_layout(height=245, margin=dict(l=20, r=20, t=35, b=25))
                st.plotly_chart(fig_stress, use_container_width=True)
            with screen_tab:
                fig_screen = plot_screen_time_trend(data)
                fig_screen.update_layout(height=245, margin=dict(l=20, r=20, t=35, b=25))
                st.plotly_chart(fig_screen, use_container_width=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # ==========================================
    # 8. HABIT RELATIONSHIPS
    # Connection Cards + Advanced Heatmap Expander
    # ==========================================
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;">
            <div style="font-size: 17px; font-weight: 700; color: #0F172A; letter-spacing: -0.01em;">
                🧩 Habit Relationships
            </div>
            <div style="font-size: 12.5px; color: #64748B;">
                Key connections discovered across your recorded check-ins
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

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
                title_color="#059669",
                progress_color="#22C55E",
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
                title_color="#DC2626",
                progress_color="#EF4444",
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
            st.plotly_chart(plot_correlation_heatmap(correlation_matrix), use_container_width=True)
        else:
            st.info(
                "Not enough valid variation in your check-in data to display "
                "the correlation heatmap yet. Record more check-ins with "
                "varied values to reveal relationships."
            )


    if weekly_ai_analysis:
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.markdown("## 🧠 Weekly AI Analysis")

        sections = weekly_ai_analysis.split("\n")

        for line in sections:
            line = line.strip()

            if not line:
                continue

            if line.startswith("WEEKLY OVERVIEW") or "WEEKLY SUMMARY" in line:
                st.markdown("### 📊 Weekly Overview")
            elif line.startswith("KEY TREND") or "METRIC COMPARISONS" in line:
                st.markdown("### 📈 Key Trend")
            elif line.startswith("MAIN FACTOR") or "ROUTINE CHANGES" in line:
                st.markdown("### 🎯 Main Factor")
            elif line.startswith("NEXT WEEK PRIORITY") or "NEXT WEEK" in line:
                st.markdown("### 🚀 Next Week Priority")
            else:
                st.write(line)


if selected_tab == "📊 Reports":
    show_reports(monthly_report)

if selected_tab == "ℹ️ About":
    show_about()
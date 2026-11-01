from src.data_manager import save_data, load_data, normalize_date_series
import streamlit as st
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
from src.weekly_report import generate_weekly_report
from src.correlation import get_correlation_matrix
from src.monthly_report import generate_monthly_report
from src.forecast import (
    predict_next_day,
    generate_forecast_summary,
    generate_forecast_explanation,
    generate_ai_forecast_insight
)
from src.intelligence import (
    get_biggest_discovery,
    get_todays_focus
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


if not data.empty:
    recommendations = generate_recommendations(data)
    patterns = analyze_patterns(data)
    weekly_report = generate_weekly_report(data)
    monthly_report = generate_monthly_report(data)
    correlation_matrix = get_correlation_matrix(data)

    prediction = predict_next_day(data)
    biggest_discovery = get_biggest_discovery(data)
    next_step = get_todays_focus(data)
    today_focus = get_todays_focus(data)
today = datetime.now()

selected_tab = show_navigation()

if selected_tab == "📝 Daily Check-in":
    st.header("📝 Your Daily Check-in")
    st.write("Record your daily habits and reflect on how you felt.")

    if st.session_state.get("checkin_success_message"):
        st.success(st.session_state.pop("checkin_success_message"))

    st.caption(
        "Enter what you know. Mood, energy, stress, and productivity are "
        "personal reflections — there are no right or wrong answers."
    )

    
    with st.form("daily_checkin"):

        # Date: one clear starting point
        date = st.date_input("Check-in date")

        habit_col, reflection_col = st.columns(
            2,
            gap="medium",
            vertical_alignment="top"
        )

        # LEFT PANEL — DAILY HABITS
        with habit_col:
            with st.container(border=True, key="daily_habits_panel"):
                st.subheader("Daily habits")
                st.caption("Record the activities you completed today.")

                sleep_hours = st.slider(
                    "Hours of sleep",
                    min_value=0.0,
                    max_value=14.0,
                    value=7.0,
                    step=0.5,
                    help="How long did you sleep in total?"
                )

                screen_time = st.number_input(
                    "Screen time (hours)",
                    min_value=0.0,
                    max_value=24.0,
                    value=4.0,
                    step=0.5
                )

                exercise_minutes = st.number_input(
                    "Exercise (minutes)",
                    min_value=0,
                    max_value=1440,
                    value=30,
                    step=5
                )

                study_hours = st.number_input(
                    "Study / work (hours)",
                    min_value=0.0,
                    max_value=24.0,
                    value=4.0,
                    step=0.5
                )

        # RIGHT PANEL — PERSONAL REFLECTIONS
        with reflection_col:
            with st.container(
                border=True,
                key="personal_reflections_panel"
            ):
                st.subheader("Personal reflections")
                st.caption("Optional. Choose the description that fits.")

                mood_choice = st.selectbox(
                    "Mood",
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

                energy_choice = st.selectbox(
                    "Energy",
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

                stress_choice = st.selectbox(
                    "Stress",
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

                productivity_choice = st.selectbox(
                    "Productivity",
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

        # OPTIONAL REFLECTION — BELOW BOTH PANELS
        with st.container(key="optional_reflection_panel"):
            st.subheader("Optional reflection")
            st.caption("Add context that numbers and ratings cannot capture.")

            daily_note = st.text_area(
                "What influenced your day?",
                placeholder=(
                    "For example: an exam, a busy day, "
                    "a good night's sleep, or a walk outdoors."
                ),
                help="This note is optional and is not used in calculations yet."
            )

        submitted = st.form_submit_button(
            "Save check-in",
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
    
    
    st.markdown("### Explore your dashboard")

    st.markdown(
        """
        <div style="
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 10px;
            margin-bottom: 24px;
        ">
            <a href="#wellness-section" class="dashboard-jump-link">
                <div style="border:1px solid #D6D9E0;border-radius:9px;padding:10px 4px;text-align:center;color:inherit;">
                    💚 Wellness
                </div>
            </a>
            <a href="#trends-section" class="dashboard-jump-link">
                <div style="border:1px solid #D6D9E0;border-radius:9px;padding:10px 4px;text-align:center;color:inherit;">
                    📈 Trends
                </div>
            </a>
            <a href="#action-section" class="dashboard-jump-link">
                <div style="border:1px solid #D6D9E0;border-radius:9px;padding:10px 4px;text-align:center;color:inherit;">
                    🎯 Action Plan
                </div>
            </a>
            <a href="#forecast-section" class="dashboard-jump-link">
                <div style="border:1px solid #D6D9E0;border-radius:9px;padding:10px 4px;text-align:center;color:inherit;">
                    🔮 Forecast
                </div>
            </a>
            <a href="#relationships-section" class="dashboard-jump-link">
                <div style="border:1px solid #D6D9E0;border-radius:9px;padding:10px 4px;text-align:center;color:inherit;">
                    🧩 Relationships
                </div>
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.html('<div id="wellness-section"></div>')
    st.subheader("🌱 Today's Wellness")

        
    # ----------------------------
    # Score + Category
    # ----------------------------

    with st.container(key="wellness_summary"):
        left, right = st.columns([1, 1], gap="medium")

        with left:
            if wellness_score is not None:
                st.metric("Overall Wellness", f"{wellness_score}/100")
                st.caption(
                    f"Score based on {available_factors} of 7 factors."
                )
            else:
                st.metric("Overall Wellness", "Not available")
                st.caption(
                    "Add valid check-in data to calculate a score."
                )

        with right:
            if wellness_score is not None:
                st.success(wellness_category)
            else:
                st.info("More data needed")

    st.markdown("---")
    # ----------------------------
    # Wellness Overview
    # ----------------------------

    st.markdown("### 📊 Wellness Overview")

    chart_col, insights_col = st.columns([1.4, 1], gap="large")

    with chart_col:
        with st.container(border=True, key="wellness_chart_panel"):
            st.markdown("#### Wellness Profile")
            st.plotly_chart(
                radar_chart,
                use_container_width=True,
                config={"displayModeBar": False}
            )

    with insights_col:
        with st.container(border=True, key="wellness_insights_panel"):
            st.markdown("#### Strongest Areas")

            strongest_areas = [
                (factor, score)
                for factor, score, status in score_breakdown
                if score is not None and score >= 80
            ]

            if strongest_areas:
                for factor, score in strongest_areas:
                    st.markdown(
            f"""
            <div class="wellness-insight-card positive">
                <span>{factor}</span>
                <strong>{score:.0f}/100</strong>
            </div>
            """,
            unsafe_allow_html=True
        )
            else:
                st.caption(
        "No strongest areas identified from the available data."
    )

            st.divider()

            st.markdown("#### Needs Attention")

            attention_areas = [
                (factor, score)
                for factor, score, status in score_breakdown
                if score is not None and score < 60
            ]

            if attention_areas:
                for factor, score in attention_areas:
                    st.markdown(
            f"""
            <div class="wellness-insight-card attention">
                <span>{factor}</span>
                <strong>{score:.0f}/100</strong>
            </div>
            """,
            unsafe_allow_html=True
        )
            else:
                st.caption(
        "No areas currently fall below this threshold."
    )

        st.markdown("---")
    st.markdown("---")
    st.html('<div id="action-section"></div>')
    show_biggest_discovery(biggest_discovery)
    
    st.markdown("---")
    
    # ----------------------------
    # Your Personal Action Plan
    # ----------------------------
    with st.container(key="action_plan_section"):
        
        st.subheader("🎯 Your Personal Action Plan")
        today_col, tomorrow_col = st.columns(2, gap="medium")

        with today_col:
            st.markdown("#### ☀️ Today's Priority")

            supporting_actions = [
                recommendation
                for recommendation in recommendations
                if recommendation not in today_focus
]

            show_action_plan(today_focus, supporting_actions)
        with tomorrow_col:
            st.markdown("#### 🔮 Preparing for Tomorrow")

            if prediction:
                ai_forecast = generate_ai_forecast_insight(prediction)
                show_ai_insight(ai_forecast)
                show_best_action()
            else:
                st.info("Tomorrow's wellness forecast is not available yet.")
    

    st.subheader("📊 Behaviour Summary")
    st.caption("Your average wellbeing across all recorded check-ins.")

    # -----------------------------
    # Dynamic Status Labels
    # -----------------------------

    # Sleep
    avg_sleep = summary.get("avg_sleep") if summary else None
    if avg_sleep is None or (isinstance(avg_sleep, float) and math.isnan(avg_sleep)):
        sleep_status = "No Data"
        sleep_display = "-- hrs"
    elif avg_sleep >= 8:
        sleep_status = "Excellent Sleep"
        sleep_display = f"{avg_sleep} hrs"
    elif avg_sleep >= 7:
        sleep_status = "Healthy Average"
        sleep_display = f"{avg_sleep} hrs"
    else:
        sleep_status = "Needs Improvement"
        sleep_display = f"{avg_sleep} hrs"

    # Mood
    avg_mood = summary.get("avg_mood") if summary else None
    if avg_mood is None or (isinstance(avg_mood, float) and math.isnan(avg_mood)):
        mood_status = "Not Recorded"
        mood_display = "--/10"
    elif avg_mood >= 8:
        mood_status = "Excellent Mood"
        mood_display = f"{avg_mood}/10"
    elif avg_mood >= 6:
        mood_status = "Moderately Positive"
        mood_display = f"{avg_mood}/10"
    else:
        mood_status = "Needs Attention"
        mood_display = f"{avg_mood}/10"

    # Stress (lower is better)
    avg_stress = summary.get("avg_stress") if summary else None
    if avg_stress is None or (isinstance(avg_stress, float) and math.isnan(avg_stress)):
        stress_status = "Not Recorded"
        stress_display = "--/10"
    elif avg_stress <= 3:
        stress_status = "Very Low Stress"
        stress_display = f"{avg_stress}/10"
    elif avg_stress <= 5:
        stress_status = "Manageable Stress"
        stress_display = f"{avg_stress}/10"
    else:
        stress_status = "High Stress"
        stress_display = f"{avg_stress}/10"

    # Wellness
    if wellness_score is None or (isinstance(wellness_score, float) and math.isnan(wellness_score)):
        wellness_status = "Insufficient Data"
        wellness_display = "Not Available"
    elif wellness_score >= 85:
        wellness_status = "Excellent Overall"
        wellness_display = f"{round(wellness_score)}/100"
    elif wellness_score >= 70:
        wellness_status = "Healthy Overall"
        wellness_display = f"{round(wellness_score)}/100"
    elif wellness_score >= 55:
        wellness_status = "Fair Overall"
        wellness_display = f"{round(wellness_score)}/100"
    else:
        wellness_status = "Needs Attention"
        wellness_display = f"{round(wellness_score)}/100"

    col1, col2, col3, col4 = st.columns(4)
    cards = [
        {
            "icon": "🌱",
            "title": "Wellness",
            "value": wellness_display,
            "subtitle": wellness_status,
            "accent": "#10B981",
            "background": "#E0F2FE"
        },
        {
            "icon": "😴",
            "title": "Sleep",
            "value": sleep_display,
            "subtitle": sleep_status,
            "accent": "#6366F1",
            "background": "#EDE9FE"
        },
        {
            "icon": "😊",
            "title": "Mood",
            "value": mood_display,
            "subtitle": mood_status,
            "accent": "#F59E0B",
            "background": "#CCFBF1"
        },
        {
            "icon": "😌",
            "title": "Stress",
            "value": stress_display,
            "subtitle": stress_status,
            "accent": "#EF4444",
            "background": "#FEF3C7"
        }
    ]

    for column, card in zip([col1, col2, col3, col4], cards):

        with column:

            show_stat_card(card)
    st.info(
        "💡 Overall, your recent habits indicate healthy sleep, moderate mood, and manageable stress. Small improvements in physical activity and screen time could further improve your overall wellness."
    )
    st.html('<div id="trends-section"></div>')
    st.markdown("---")
        
    # ==========================================
    # Behaviour Trends
    # ==========================================
    
    st.subheader("📈 Behaviour Trends")
    st.caption("Understand how your daily habits have changed over time.")

    left, right = st.columns([1.1, 2.9], gap="large")

    # -----------------------------
    # LEFT SIDE
    # -----------------------------

    with left:

        st.subheader("🧠 AI Trend Analysis")
        st.caption("Automatically generated from your recent behaviour patterns.")

        show_trend_analysis(trend_analysis)

    # -----------------------------
    # RIGHT SIDE
    # -----------------------------

    with right:

        sleep_tab, mood_tab, stress_tab, screen_tab = st.tabs([
    "😴 Sleep",
    "😊 Mood",
    "😌 Stress",
    "📱 Screen Time"
])

        with sleep_tab:
            st.plotly_chart(
                plot_sleep_trend(data),
                use_container_width=True
            )

        with mood_tab:
            st.plotly_chart(
                plot_mood_trend(data),
                use_container_width=True
            )

        with stress_tab:
            st.plotly_chart(
                plot_stress_trend(data),
                use_container_width=True
            )

        with screen_tab:
            st.plotly_chart(
                plot_screen_time_trend(data),
                use_container_width=True
            )
    st.html('<div id="forecast-section"></div>')
    st.divider()
    
    st.subheader("🔮 Tomorrow's Wellness Forecast")

    st.caption("AI prediction generated from your recent behavioural patterns.")

    if prediction:

        forecast = generate_forecast_summary(prediction)
        explanation = generate_forecast_explanation(prediction)
        

        # -------------------------
        # Overall Forecast
        # -------------------------

        show_forecast(forecast)
        st.markdown("")

        # -------------------------
        # Primary Metrics
        # -------------------------

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "😴 Sleep",
            f'{prediction["SleepHours"]} hrs'
        )

        col2.metric(
            "😊 Mood",
            f'{prediction["Mood"]}/10'
        )

        col3.metric(
            "😌 Stress",
            f'{prediction["Stress"]}/10'
        )


        # -------------------------
        # Secondary Metrics
        # -------------------------

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "⚡ Energy",
            f'{prediction["Energy"]}/10'
        )

        col2.metric(
            "💻 Screen Time",
            f'{prediction["ScreenTime"]} hrs'
        )

        col3.metric(
            "🚀 Productivity",
            f'{prediction["Productivity"]}/10'
        )
    st.html('<div id="relationships-section"></div>')
    st.markdown("---")
    
    
    st.subheader("🧩 Habit Relationships")
    st.caption(
        "Discover which daily habits are most closely connected based on your recorded check-ins."
    )

    with st.expander("📊 Explore correlation heatmap (advanced)", expanded=False):

        if (
            not correlation_matrix.empty
            and correlation_matrix.notna().to_numpy().any()
        ):
            st.caption(
                "This chart shows how strongly your recorded habits are related. "
                "Positive values indicate habits that tend to move together; "
                "negative values indicate habits that tend to move in opposite directions. "
                "Correlation does not necessarily mean one habit causes another."
            )

            st.plotly_chart(
                plot_correlation_heatmap(correlation_matrix),
                use_container_width=True
            )
        else:
            st.info(
                "Not enough valid variation in your check-in data to display "
                "the correlation heatmap yet. Record more check-ins with "
                "varied values to reveal relationships."
            )

    # ---------------------------------------
    # AI Relationship Insights
    # ---------------------------------------

    
    insights = generate_correlation_insights(correlation_matrix)

    positive = insights["positive"] if insights else None
    negative = insights["negative"] if insights else None
    
    st.markdown("### AI Relationship Insights")

    # Generate explanations only when a relationship is available.
    positive_explanation = None
    negative_explanation = None

    if positive is not None:
        positive_explanation = generate_relationship_explanation(
            positive["feature1"],
            positive["feature2"],
            positive["value"],
        )

    if negative is not None:
        negative_explanation = generate_relationship_explanation(
            negative["feature1"],
            negative["feature2"],
            negative["value"],
        )

    col1, col2 = st.columns(2)

    with col1:
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
            st.info(
                "A positive relationship cannot be identified "
                "from the available data yet."
            )

    with col2:
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
            st.info(
                "A negative relationship cannot be identified "
                "from the available data yet."
            )
    
# ----------------------------
# Reports
# ----------------------------

if selected_tab == "📊 Reports":
    show_reports(monthly_report)

if selected_tab == "ℹ️ About":
    show_about()
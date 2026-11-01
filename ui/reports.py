
import pandas as pd
import streamlit as st

from src.data_manager import load_data
from src.weekly_report import generate_weekly_report


def show_reports(monthly_report):

    st.subheader("📊 Wellness Reports")
    st.caption("A compact overview of your wellness progress.")

    # Load existing check-in data
    data = load_data()

    if data.empty:
        st.info("No check-in records available yet. Start by recording your daily check-in.")
        return

    # Prepare reporting data
    data = data.copy()

    # Parse dates using the same formats as app.py
    date_values = data["Date"].astype("string").str.strip()

    parsed_dates = pd.to_datetime(
        date_values,
        format="%Y-%m-%d",
        errors="coerce"
    )

    missing_dates = parsed_dates.isna()

    parsed_dates.loc[missing_dates] = pd.to_datetime(
        date_values.loc[missing_dates],
        format="%d-%m-%Y",
        errors="coerce"
    )

    data["Date"] = parsed_dates

    numeric_columns = [
        "SleepHours",
        "Mood",
        "Stress",
        "Energy",
        "ScreenTime",
        "ExerciseMinutes",
        "StudyHours",
        "Productivity",
    ]

    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["Date"])

    if data.empty:
        st.info("No valid check-in dates are available for reporting.")
        return

    # Reporting period
    period = st.selectbox(
        "Reporting period",
        ["Last 7 days", "Last 30 days", "Last 90 days", "All time"],
        index=1,
        key="reports_period",
    )

    today = pd.Timestamp.today().normalize()

    if period == "Last 7 days":
        start_date = today - pd.Timedelta(days=6)
    elif period == "Last 30 days":
        start_date = today - pd.Timedelta(days=29)
    elif period == "Last 90 days":
        start_date = today - pd.Timedelta(days=89)
    else:
        start_date = data["Date"].min().normalize()

    filtered_data = data[
        (data["Date"] >= start_date)
        & (data["Date"] < today + pd.Timedelta(days=1))
    ].copy()

    if filtered_data.empty:
        st.info("No check-ins found in this reporting period.")
        return

    
    # Key metrics
    total_checkins = len(filtered_data)

    def get_average(column):
        if column in filtered_data.columns:
            return filtered_data[column].mean()
        return None

    metrics = [
        ("📅 Total Check-ins", str(total_checkins)),
        ("😴 Average Sleep", f"{get_average('SleepHours'):.1f} hrs" if pd.notna(get_average("SleepHours")) else "N/A"),
        ("😊 Average Mood", f"{get_average('Mood'):.1f}/10" if pd.notna(get_average("Mood")) else "N/A"),
        ("🧘 Average Stress", f"{get_average('Stress'):.1f}/10" if pd.notna(get_average("Stress")) else "N/A"),
        ("⚡ Average Energy", f"{get_average('Energy'):.1f}/10" if pd.notna(get_average("Energy")) else "N/A"),
        ("🏃 Daily Exercise", f"{get_average('ExerciseMinutes'):.0f} min" if pd.notna(get_average("ExerciseMinutes")) else "N/A"),
        ("📚 Study Time", f"{get_average('StudyHours'):.1f} hrs" if pd.notna(get_average("StudyHours")) else "N/A"),
        ("🎯 Productivity", f"{get_average('Productivity'):.1f}/10" if pd.notna(get_average("Productivity")) else "N/A"),
    ]

    st.markdown("#### Your Wellness Metrics")

    
    with st.container(key="reports_metrics"):
        for row_start in range(0, len(metrics), 4):
            cols = st.columns(4, gap="small")

            for col, (label, value) in zip(
                cols, metrics[row_start:row_start + 4]
            ):
                with col:
                    st.metric(label, value)


    
    # Compact weekly and monthly summaries
    st.markdown("#### Wellness Summaries")

    weekly_start = today - pd.Timedelta(days=6)

    weekly_data = data[
        (data["Date"] >= weekly_start)
        & (data["Date"] < today + pd.Timedelta(days=1))
    ].copy()

    weekly_report = generate_weekly_report(weekly_data)

    weekly_col, monthly_col = st.columns(2, gap="medium")

    with weekly_col:
        with st.container(key="reports_weekly_summary", border=True):
            st.markdown("**📅 Weekly Summary**")

            if weekly_report:
                st.markdown(
                    weekly_report["headline"]
                )

                st.caption(
                    f"🏆 Achievement: {weekly_report['badge']}"
                )

                st.caption(
                    f"🎯 Focus: {weekly_report['mission']}"
                )
            else:
                st.caption(
                    "Not enough recent check-ins for a weekly summary."
                )

    with monthly_col:
        with st.container(key="reports_monthly_summary", border=True):
            st.markdown("**🗓️ Monthly Summary**")

            if monthly_report:
                st.caption(
                    f"🏃 Active on {monthly_report['exercise_days']} "
                    "of your last 30 check-ins."
                )

                st.markdown(
                    f"**💚 Biggest Win:** {monthly_report['biggest_win']}"
                )

                st.markdown(
                    f"**⚠️ Needs Attention:** {monthly_report['challenge']}"
                )

                st.markdown(
                    f"**🚀 Next Opportunity:** {monthly_report['opportunity']}"
                )
            else:
                st.caption("Monthly summary is not available yet.")

    
    # CSV export
    with st.container(key="reports_download"):
        st.download_button(
            label="⬇️ Download Wellness Report (CSV)",
            data=filtered_data.to_csv(index=False).encode("utf-8"),
            file_name="lifelens_wellness_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

import streamlit as st


def show_monthly_score(monthly_report):

    st.html(
        f"""
        <div class="monthly-score-card">
            <div class="monthly-score-label">
                Monthly Wellness Score
            </div>
            <div class="monthly-score-value">
                ⭐ {monthly_report["overall_score"]}/100
            </div>
        </div>
        """
    )
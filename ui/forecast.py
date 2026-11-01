import streamlit as st


def show_forecast(forecast):

    st.html(
        f"""
        <div class="forecast-card">
            <div class="forecast-label">
                AI Forecast
            </div>
            <div class="forecast-title">
                {forecast}
            </div>
            <div class="forecast-description">
                Based on your recent sleep, mood, stress, energy and productivity trends,
                your overall wellbeing is expected to remain consistent tomorrow.
            </div>
        </div>
        """
    )
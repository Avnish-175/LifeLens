
import html
import streamlit as st


def show_ai_insight(ai_forecast):

    safe_forecast = html.escape(str(ai_forecast))

    st.html(
        f"""
        <div class="ai-insight-card">
            <div class="ai-insight-label">
                AI INSIGHT
            </div>
            <div class="ai-insight-text">
                {safe_forecast}
            </div>
        </div>
        """
    )


def show_trend_analysis(trend_analysis):

    insight_cards = [
        {
            "label": "POSITIVE",
            "text": trend_analysis["positive"],
            "class": "trend-positive",
        },
        {
            "label": "WATCH",
            "text": trend_analysis["watch"],
            "class": "trend-watch",
        },
        {
            "label": "BIGGEST RISK",
            "text": trend_analysis["risk"],
            "class": "trend-risk",
        },
        {
            "label": "AI RECOMMENDATION",
            "text": trend_analysis["recommendation"],
            "class": "trend-recommendation",
        },
    ]

    cards_html = ""

    for card in insight_cards:
        safe_label = html.escape(card["label"])
        safe_text = html.escape(str(card["text"]))
        safe_class = card["class"]

        cards_html += f"""
            <div class="trend-insight-card {safe_class}">
                <div class="trend-insight-label">
                    {safe_label}
                </div>
                <div class="trend-insight-text">
                    {safe_text}
                </div>
            </div>
        """

    st.html(
        f"""
        <div class="trend-insight-container">
            {cards_html}
        </div>
        """
    )

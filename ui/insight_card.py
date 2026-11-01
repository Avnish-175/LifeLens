import streamlit as st


def show_insight_card(insight, style):

    st.html(
        f"""
        <div class="insight-card" style="border-left-color:{style['border']};">
            <div class="insight-title">
                {insight['icon']} {insight['title']}
            </div>
            <div class="insight-message">
                {insight['message']}
            </div>
            <span class="insight-badge" style="background:{style['badge']};">
                {style['text']}
            </span>
        </div>
        """
    )
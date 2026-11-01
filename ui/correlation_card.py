import streamlit as st


def show_correlation_card(
    title,
    title_color,
    progress_color,
    icon,
    correlation,
    explanation,
):

    st.html(
        f"""
        <div class="correlation-card">
            <div class="correlation-title" style="color:{title_color};">
                {icon} {title}
            </div>
            <div class="correlation-features">
                {correlation["feature1"]} ↔ {correlation["feature2"]}
            </div>
            <div class="correlation-bar">
                <div class="correlation-progress" style="width:{correlation['percent']}%; background:{progress_color};"></div>
            </div>
            <div class="correlation-strength" style="color:{title_color};">
                {icon} {correlation["strength"]} • {correlation["percent"]}%
            </div>
            <div class="correlation-text">
                {explanation}
            </div>
        </div>
        """
    )
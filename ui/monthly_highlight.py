
import html
import streamlit as st


def show_monthly_highlight(title, message, highlight_type):

    styles = {
        "win": {
            "background": "#ECFDF5",
            "border": "#10B981",
            "title": "#047857",
        },
        "attention": {
            "background": "#FFFBEB",
            "border": "#F59E0B",
            "title": "#B45309",
        },
        "opportunity": {
            "background": "#EFF6FF",
            "border": "#3B82F6",
            "title": "#1D4ED8",
        },
    }

    style = styles.get(highlight_type, styles["opportunity"])

    safe_title = html.escape(str(title))
    safe_message = html.escape(str(message))

    st.html(
        f"""
        <div class="monthly-highlight-card"
             style="
                background:{style['background']};
                border-left:4px solid {style['border']};
             ">
            <div class="monthly-highlight-title"
                 style="color:{style['title']};">
                {safe_title}
            </div>
            <div class="monthly-highlight-message">
                {safe_message}
            </div>
        </div>
        """
    )

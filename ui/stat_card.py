
import html
import streamlit as st


def show_stat_card(card):

    safe_title = html.escape(str(card["title"]))
    safe_value = html.escape(str(card["value"]))
    safe_subtitle = html.escape(str(card["subtitle"]))
    safe_icon = html.escape(str(card["icon"]))
    icon_background = card["background"]

    st.html(
        f"""
        <div class="stat-card">
            <div class="stat-header">
                <div class="stat-icon"
                     style="background:{icon_background};">
                    {safe_icon}
                </div>

                <div class="stat-title">
                    {safe_title}
                </div>
            </div>

            <div class="stat-value">
                {safe_value}
            </div>

            <div class="stat-subtitle">
                {safe_subtitle}
            </div>
        </div>
        """
    )

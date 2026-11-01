
import streamlit as st


def show_navigation():

    pages = [
        "🏠 Dashboard",
        "📝 Daily Check-in",
        "📊 Reports",
        "ℹ️ About",
    ]

    with st.container(key="lifelens_navigation"):

        selected = st.pills(
            "",
            options=pages,
            default="🏠 Dashboard",
            selection_mode="single",
            key="lifelens_page_selector",
        )

    return selected

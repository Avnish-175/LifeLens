import streamlit as st


def show_navigation():
    if "lifelens_theme" not in st.session_state:
        st.session_state["lifelens_theme"] = "dark"

    pages = [
        "🏠 Dashboard",
        "📝 Daily Check-in",
        "📊 Reports",
        "ℹ️ About",
    ]

    theme_options = ["🌙 Dark", "☀️ Light"]
    current_theme = st.session_state.get("lifelens_theme", "dark")
    default_theme_option = "☀️ Light" if current_theme == "light" else "🌙 Dark"

    with st.container(key="lifelens_navigation"):
        nav_col, theme_col = st.columns([3.8, 1.2], gap="small", vertical_alignment="center")

        with nav_col:
            selected = st.pills(
                "Navigation",
                options=pages,
                default="🏠 Dashboard",
                selection_mode="single",
                key="lifelens_page_selector",
                label_visibility="collapsed",
            )

        with theme_col:
            selected_theme = st.segmented_control(
                "Theme Mode",
                options=theme_options,
                default=default_theme_option,
                selection_mode="single",
                key="lifelens_theme_toggle_widget",
                label_visibility="collapsed",
            )

            if selected_theme:
                target_theme = "light" if selected_theme == "☀️ Light" else "dark"
                if target_theme != st.session_state.get("lifelens_theme"):
                    st.session_state["lifelens_theme"] = target_theme
                    st.rerun()

    return selected or "🏠 Dashboard"

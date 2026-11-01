import streamlit as st


def show_biggest_discovery(biggest_discovery):

    st.html(
        f"""
        <div class="discovery-card">
            <h2 class="discovery-title">
                🚨 Today's Biggest Discovery
            </h2>
            <p class="discovery-text">
                {biggest_discovery}
            </p>
        </div>
        """
    )
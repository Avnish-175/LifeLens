
import streamlit as st


def show_best_action():

    st.html(
        """
        <div class="best-action-card">
            <div class="best-action-label">
                RECOMMENDED ACTION
            </div>
            <div class="best-action-text">
                Reduce your screen time this evening while maintaining
                your current sleep schedule. This may help support
                tomorrow's wellness.
            </div>
        </div>
        """
    )

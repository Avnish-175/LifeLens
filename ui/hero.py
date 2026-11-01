
import streamlit as st


def show_hero(greeting, today):
    st.html(
        """
        <div class="hero-card hero-compact">
            <div class="hero-brand">
                <h1 class="hero-title">🌱 LifeLens</h1>
                <p class="hero-subtitle">
                    Your Personal Behaviour Intelligence Platform
                </p>
            </div>
        </div>
        """
    )

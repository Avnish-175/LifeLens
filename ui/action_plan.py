
import html
import streamlit as st


def show_action_plan(today_focus, supporting_actions):

    safe_priority = html.escape(str(today_focus))

    supporting_html = ""

    if supporting_actions:
        supporting_rows = ""

        for recommendation in supporting_actions:
            safe_recommendation = html.escape(str(recommendation))

            supporting_rows += f"""
                <div class="action-plan-supporting-row">
                    <span class="action-plan-check">✓</span>
                    <span>{safe_recommendation}</span>
                </div>
            """

        supporting_html = f"""
            <div class="action-plan-supporting-section">
                <div class="action-plan-supporting-title">
                    Supporting Actions
                </div>
                {supporting_rows}
            </div>
        """

    st.html(
        f"""
        <div class="action-plan-unified-card">

            <div class="action-plan-primary-label">
                TODAY'S TOP PRIORITY
            </div>

            <div class="action-plan-primary-text">
                {safe_priority}
            </div>

            {supporting_html}

        </div>
        """
    )

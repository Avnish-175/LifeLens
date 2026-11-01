import streamlit as st


def show_weekly_snapshot(weekly_report):

    st.subheader("🧠 Weekly Snapshot")
    st.caption("Your week at a glance — no reading required.")

    st.html(
        f"""
        <div class="weekly-card">
            <div class="weekly-label">
                AI WEEKLY SUMMARY
            </div>
            <div class="weekly-headline">
                {weekly_report["headline"]}
            </div>
            <div class="weekly-grid">
                <div class="weekly-stat">
                    <div class="weekly-stat-title">😴 Sleep</div>
                    <div class="weekly-stat-value sleep-good">{weekly_report["sleep_status"]}</div>
                </div>
                <div class="weekly-stat">
                    <div class="weekly-stat-title">😊 Mood</div>
                    <div class="weekly-stat-value mood-stable">{weekly_report["mood_status"]}</div>
                </div>
                <div class="weekly-stat">
                    <div class="weekly-stat-title">😌 Stress</div>
                    <div class="weekly-stat-value stress-high">{weekly_report["stress_status"]}</div>
                </div>
                <div class="weekly-stat">
                    <div class="weekly-stat-title">⚡ Energy</div>
                    <div class="weekly-stat-value energy-low">{weekly_report["energy_status"]}</div>
                </div>
            </div>
            <div class="weekly-takeaway">
                ✨ <b>AI Takeaway</b><br><br>
                {weekly_report["reflection"]}
            </div>
        </div>
        """
    )
    st.divider()
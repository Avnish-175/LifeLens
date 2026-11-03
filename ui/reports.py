import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

from src.data_manager import load_data, normalize_date_series
from src.wellness_score import calculate_wellness_score, _valid_number
from src.weekly_report import generate_weekly_report
from src.monthly_report import generate_monthly_report
from src.nlp_facts import build_weekly_facts
from ml.nlp.reporter import explain_weekly_facts
from src.pdf_generator import generate_wellness_pdf
from src.prediction_tracker import get_actual_vs_predicted_data


def _fmt_delta(diff, unit=""):
    sign = "+" if diff > 0 else ""
    if unit in ("min", "%"):
        return f"{sign}{diff:.0f} {unit}".strip()
    elif unit == "hrs":
        return f"{sign}{diff:.1f} hrs"
    return f"{sign}{diff:.1f}{unit}".strip()


def _create_compact_trend_chart(df, y_col, title, unit, color):
    """
    Helper to generate a clean, compact dual-theme line/scatter trend chart for Reports.
    """
    valid_df = df.dropna(subset=["Date", y_col]).sort_values("Date")
    if valid_df.empty:
        return None

    fig = px.line(
        valid_df,
        x="Date",
        y=y_col,
        markers=True,
        color_discrete_sequence=[color]
    )

    hover_fmt = (
        "<b>%{x|%d %b %Y}</b><br>"
        f"{title}: %{{y:.1f}} {unit}<extra></extra>"
        if unit not in ("/10", "min")
        else f"<b>%{{x|%d %b %Y}}</b><br>{title}: %{{y}}{unit}<extra></extra>"
    )

    fig.update_traces(
        line=dict(width=2.5),
        marker=dict(size=7, line=dict(width=1.5, color="#FFFFFF")),
        hovertemplate=hover_fmt
    )

    theme = st.session_state.get("lifelens_theme", "dark")
    if theme == "light":
        plot_bg = "rgba(0,0,0,0)"
        paper_bg = "rgba(0,0,0,0)"
        font_color = "#29243A"
        title_color = "#29243A"
        grid_color = "rgba(0,0,0,0.06)"
        line_color = "rgba(0,0,0,0.12)"
    else:
        plot_bg = "rgba(0,0,0,0)"
        paper_bg = "rgba(0,0,0,0)"
        font_color = "#AAA5C9"
        title_color = "#F5F2FF"
        grid_color = "rgba(255,255,255,0.06)"
        line_color = "rgba(255,255,255,0.10)"

    fig.update_layout(
        height=210,
        margin=dict(l=15, r=15, t=32, b=15),
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=13, color=title_color),
            x=0.01,
            y=0.96
        ),
        xaxis_title="",
        yaxis_title=unit if unit else "",
        plot_bgcolor=plot_bg,
        paper_bgcolor=paper_bg,
        font=dict(family="Inter, system-ui, sans-serif", size=11, color=font_color),
        hovermode="x unified"
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=grid_color,
        tickformat="%d %b" if len(valid_df) > 1 else "%d %b %Y",
        linecolor=line_color
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=grid_color,
        linecolor=line_color,
        zeroline=False
    )
    return fig


def _compute_period_stats(df, total_days_in_period=7):
    """
    Computes scores, consistency, best/worst days, averages strictly from actual records.
    """
    if df is None or df.empty:
        return None

    sorted_df = df.copy()
    sorted_df["Date"] = pd.to_datetime(sorted_df["Date"])
    sorted_df = sorted_df.sort_values("Date").dropna(subset=["Date"])
    days_recorded = len(sorted_df)

    # Compute daily wellness scores
    daily_scores = []
    for _, row in sorted_df.iterrows():
        sc, cat = calculate_wellness_score(
            row.get("SleepHours"),
            row.get("Mood"),
            row.get("Stress"),
            row.get("Energy"),
            row.get("ExerciseMinutes"),
            row.get("ScreenTime"),
            row.get("Productivity"),
        )
        if sc is not None:
            daily_scores.append({
                "Date": row["Date"].strftime("%Y-%m-%d"),
                "Score": round(sc, 1),
                "Category": cat
            })

    start_score = daily_scores[0]["Score"] if daily_scores else None
    end_score = daily_scores[-1]["Score"] if daily_scores else None
    score_delta = round(end_score - start_score, 1) if (start_score is not None and end_score is not None) else 0.0

    best_day = max(daily_scores, key=lambda x: x["Score"]) if daily_scores else None
    worst_day = min(daily_scores, key=lambda x: x["Score"]) if daily_scores else None

    consistency_pct = round((days_recorded / total_days_in_period) * 100)

    # Metric averages
    def get_avg(col):
        if col in sorted_df.columns:
            s = sorted_df[col].dropna()
            return s.mean() if not s.empty else None
        return None

    return {
        "days_recorded": days_recorded,
        "total_days": total_days_in_period,
        "consistency_pct": consistency_pct,
        "consistency_str": f"{days_recorded}/{total_days_in_period} days ({consistency_pct}%)",
        "start_score": start_score,
        "end_score": end_score,
        "score_delta": score_delta,
        "score_delta_str": _fmt_delta(score_delta),
        "best_day": best_day["Date"] if best_day else "N/A",
        "best_score": best_day["Score"] if best_day else "N/A",
        "worst_day": worst_day["Date"] if worst_day else "N/A",
        "worst_score": worst_day["Score"] if worst_day else "N/A",
        "avg_sleep": f"{get_avg('SleepHours'):.1f} hrs" if get_avg("SleepHours") is not None else "N/A",
        "avg_mood": f"{get_avg('Mood'):.1f}/10" if get_avg("Mood") is not None else "N/A",
        "avg_energy": f"{get_avg('Energy'):.1f}/10" if get_avg("Energy") is not None else "N/A",
        "avg_stress": f"{get_avg('Stress'):.1f}/10" if get_avg("Stress") is not None else "N/A",
        "avg_productivity": f"{get_avg('Productivity'):.1f}/10" if get_avg("Productivity") is not None else "N/A",
        "avg_exercise": f"{get_avg('ExerciseMinutes'):.0f} min" if get_avg("ExerciseMinutes") is not None else "N/A",
        "avg_screen": f"{get_avg('ScreenTime'):.1f} hrs" if get_avg("ScreenTime") is not None else "N/A",
        "avg_study": f"{get_avg('StudyHours'):.1f} hrs" if get_avg("StudyHours") is not None else "N/A",
        "category": daily_scores[-1]["Category"] if daily_scores else "Not recorded"
    }


def show_reports(monthly_report=None, data=None):
    """
    Renders the LifeLens Professional Wellness Report Centre:
    1. Weekly Wellness Report
    2. Monthly Wellness Report
    3. AI Wellness Analysis (on-demand generation with caching)
    4. Actual vs Predicted Analysis
    """
    if data is None:
        data = load_data()

    if data.empty:
        st.html(
            """
            <div class="ll-empty">
                <span class="ll-empty-icon">📊</span>
                <b style="color:var(--ll-text);font-size:1.1rem;">No Check-in Data Available</b><br>
                Please complete your daily check-ins to generate personal wellness reports.
            </div>
            """
        )
        return

    # Clean & normalize dates
    df = data.copy()
    if "Date" in df.columns:
        df["Date"] = normalize_date_series(df["Date"])
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")
        df = df.drop_duplicates(subset=["Date"], keep="last")

    for col in ["SleepHours", "Mood", "Stress", "Energy", "ScreenTime", "ExerciseMinutes", "StudyHours", "Productivity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    today = pd.Timestamp.today().normalize()

    st.html(
        """
        <div class="ll-page-head">
            <span class="ll-eyebrow">📈 Intelligence Hub · Longitudinal Analytics</span>
            <h1>Wellness Report Centre</h1>
            <p>Comprehensive retrospective reports, longitudinal habit trajectories, and predictive accuracy audits.</p>
        </div>
        """
    )

    report_tabs = st.tabs([
        "📅 Weekly Wellness Report",
        "🗓️ Monthly Wellness Report",
        "🤖 AI Wellness Analysis",
        "🎯 Actual vs Predicted Analysis"
    ])

    # =========================================================================
    # TAB 1: WEEKLY WELLNESS REPORT
    # =========================================================================
    with report_tabs[0]:
        weekly_start = today - pd.Timedelta(days=6)
        weekly_df = df[(df["Date"] >= weekly_start) & (df["Date"] <= today)].copy()
        period_str = f"{weekly_start.strftime('%b %d, %Y')} – {today.strftime('%b %d, %Y')}"

        if weekly_df.empty:
            st.info(f"No check-ins found for the current week ({period_str}). Record a daily check-in to unlock your weekly report.")
        else:
            w_stats = _compute_period_stats(weekly_df, total_days_in_period=7)

            # Incomplete period notification
            if w_stats["days_recorded"] < 7:
                st.warning(
                    f"⚠️ **Incomplete Reporting Period**: {w_stats['days_recorded']} of 7 days recorded for this week ({w_stats['consistency_pct']}% consistency). "
                    "Statistics and averages are calculated strictly from your actual recorded days."
                )

            # Executive Score Header Cards
            with st.container(key="reports_exec_metrics"):
                c1, c2, c3, c4 = st.columns(4, gap="small")
                with c1:
                    st.metric("Starting Score", f"{w_stats['start_score']:.0f}/100" if w_stats['start_score'] is not None else "N/A")
                with c2:
                    st.metric("Ending Score", f"{w_stats['end_score']:.0f}/100" if w_stats['end_score'] is not None else "N/A", delta=w_stats["score_delta_str"])
                with c3:
                    st.metric("Check-in Consistency", w_stats["consistency_str"])
                with c4:
                    with st.container(key="reports_category_metric"):
                        st.metric("Wellness Category", w_stats["category"])

            st.markdown("#### Average Weekly Lifestyle Indicators")
            with st.container(key="reports_lifestyle_metrics"):
                m_cols = st.columns(4, gap="small")
                with m_cols[0]:
                    st.metric("😴 Avg Sleep", w_stats["avg_sleep"])
                    st.metric("🏃 Daily Exercise", w_stats["avg_exercise"])
                with m_cols[1]:
                    st.metric("😊 Avg Mood", w_stats["avg_mood"])
                    st.metric("💻 Screen Time", w_stats["avg_screen"])
                with m_cols[2]:
                    st.metric("⚡ Avg Energy", w_stats["avg_energy"])
                    st.metric("📚 Study / Work", w_stats["avg_study"])
                with m_cols[3]:
                    st.metric("🧘 Avg Stress", w_stats["avg_stress"])
                    st.metric("🎯 Productivity", w_stats["avg_productivity"])

            # Peak & Challenging Days
            p1, p2 = st.columns(2, gap="medium")
            with p1:
                st.html(
                    f"""
                    <div class="ll-card" style="padding: 14px 18px;">
                        <span class="ll-chip" style="color:var(--ll-good);">🌟 Peak Wellness Day</span>
                        <div style="font-size: 1.12rem; font-weight: 600; margin-top: 8px;">{w_stats['best_day']}</div>
                        <div style="color:var(--ll-muted); font-size: 0.86rem;">Wellness Score: <b>{w_stats['best_score']}/100</b></div>
                    </div>
                    """
                )
            with p2:
                st.html(
                    f"""
                    <div class="ll-card" style="padding: 14px 18px;">
                        <span class="ll-chip" style="color:var(--ll-warn);">🌱 Recovery / Challenging Day</span>
                        <div style="font-size: 1.12rem; font-weight: 600; margin-top: 8px;">{w_stats['worst_day']}</div>
                        <div style="color:var(--ll-muted); font-size: 0.86rem;">Wellness Score: <b>{w_stats['worst_score']}/100</b></div>
                    </div>
                    """
                )

            # Weekly Trend Visualizations
            st.markdown("#### Weekly Indicator Trajectories")
            ch_cols = st.columns(2, gap="medium")
            with ch_cols[0]:
                fig_sleep = _create_compact_trend_chart(weekly_df, "SleepHours", "😴 Sleep Duration", "hrs", "#8B5CF6")
                if fig_sleep: st.plotly_chart(fig_sleep, use_container_width=True)
                fig_energy = _create_compact_trend_chart(weekly_df, "Energy", "⚡ Vital Energy", "/10", "#FBBF24")
                if fig_energy: st.plotly_chart(fig_energy, use_container_width=True)
            with ch_cols[1]:
                fig_mood = _create_compact_trend_chart(weekly_df, "Mood", "😊 Mood Rating", "/10", "#34D399")
                if fig_mood: st.plotly_chart(fig_mood, use_container_width=True)
                fig_stress = _create_compact_trend_chart(weekly_df, "Stress", "🧘 Stress Level", "/10", "#FB7185")
                if fig_stress: st.plotly_chart(fig_stress, use_container_width=True)

            # PDF Export: explicit preparation & deferred user-triggered download
            st.markdown("#### Export Report")
            w_pdf_key = f"cached_pdf_weekly_{w_stats['days_recorded']}_{w_stats['end_score']}_{w_stats['start_score']}"

            if st.session_state.get("show_weekly_pdf_download") and w_pdf_key in st.session_state:
                dl_col1, dl_col2 = st.columns([3.2, 0.8], gap="small", vertical_alignment="center")
                with dl_col1:
                    st.download_button(
                        label="⬇️ Download Weekly Wellness Report (PDF)",
                        data=st.session_state[w_pdf_key],
                        file_name="LifeLens_Weekly_Report.pdf",
                        mime="application/pdf",
                        key="btn_download_weekly_pdf",
                        on_click="ignore",
                        use_container_width=True
                    )
                with dl_col2:
                    if st.button("Close", key="close_weekly_pdf", use_container_width=True):
                        st.session_state["show_weekly_pdf_download"] = False
                        st.rerun()
            else:
                if st.button("📄 Generate Weekly Wellness Report (PDF)", key="btn_prep_weekly_pdf", type="primary", use_container_width=True):
                    with st.spinner("Generating Weekly PDF Report..."):
                        pdf_data = generate_wellness_pdf(
                            "Weekly",
                            w_stats,
                            period_str,
                            ai_text=f"Weekly consistency reached {w_stats['consistency_pct']}% with an ending Wellness Score of {w_stats['end_score']}/100. Peak day recorded on {w_stats['best_day']}.",
                            recommendations=[
                                "Maintain stable sleep duration to support mental mood balance.",
                                "Keep screen exposure under control during evening hours.",
                                "Incorporate regular 30-minute movement sessions to sustain high energy levels."
                            ]
                        )
                        st.session_state[w_pdf_key] = pdf_data
                        st.session_state["show_weekly_pdf_download"] = True
                        st.rerun()

    # =========================================================================
    # TAB 2: MONTHLY WELLNESS REPORT
    # =========================================================================
    with report_tabs[1]:
        monthly_start = today - pd.Timedelta(days=29)
        monthly_df = df[(df["Date"] >= monthly_start) & (df["Date"] <= today)].copy()
        period_m_str = f"{monthly_start.strftime('%b %d, %Y')} – {today.strftime('%b %d, %Y')}"

        if monthly_df.empty:
            st.info(f"No check-ins found for the past 30 days ({period_m_str}).")
        else:
            m_stats = _compute_period_stats(monthly_df, total_days_in_period=30)

            if m_stats["days_recorded"] < 30:
                st.warning(
                    f"⚠️ **Incomplete Reporting Period**: {m_stats['days_recorded']} of 30 days recorded ({m_stats['consistency_pct']}% consistency). "
                    "Longitudinal statistics are derived strictly from available check-ins without placeholder assumptions."
                )

            with st.container(key="reports_exec_metrics_monthly"):
                mc1, mc2, mc3, mc4 = st.columns(4, gap="small")
                with mc1:
                    st.metric("Starting Score (30d)", f"{m_stats['start_score']:.0f}/100" if m_stats['start_score'] is not None else "N/A")
                with mc2:
                    st.metric("Ending Score", f"{m_stats['end_score']:.0f}/100" if m_stats['end_score'] is not None else "N/A", delta=m_stats["score_delta_str"])
                with mc3:
                    st.metric("Monthly Consistency", m_stats["consistency_str"])
                with mc4:
                    with st.container(key="reports_category_metric_monthly"):
                        st.metric("Overall Category", m_stats["category"])

            st.markdown("#### Monthly Lifestyle Averages")
            with st.container(key="reports_lifestyle_metrics_monthly"):
                m_cols2 = st.columns(4, gap="small")
                with m_cols2[0]:
                    st.metric("😴 Avg Sleep", m_stats["avg_sleep"])
                    st.metric("🏃 Daily Exercise", m_stats["avg_exercise"])
                with m_cols2[1]:
                    st.metric("😊 Avg Mood", m_stats["avg_mood"])
                    st.metric("💻 Screen Time", m_stats["avg_screen"])
                with m_cols2[2]:
                    st.metric("⚡ Avg Energy", m_stats["avg_energy"])
                    st.metric("📚 Study / Work", m_stats["avg_study"])
                with m_cols2[3]:
                    st.metric("🧘 Avg Stress", m_stats["avg_stress"])
                    st.metric("🎯 Productivity", m_stats["avg_productivity"])

            # Monthly Highlights Summary
            if monthly_report:
                st.markdown("#### Monthly Executive Summary")
                st.html(
                    f"""
                    <div class="ll-card">
                        <div style="font-size: 1.05rem; line-height: 1.6; margin-bottom: 10px;">
                            <b>🏆 Biggest Win:</b> {monthly_report.get('biggest_win', 'Consistently tracking habits across the month.')}
                        </div>
                        <div style="font-size: 1.05rem; line-height: 1.6; margin-bottom: 10px; color: var(--ll-warn);">
                            <b>⚠️ Area for Growth:</b> {monthly_report.get('challenge', 'Focus on evening screen time and recovery.')}
                        </div>
                        <div style="font-size: 1.05rem; line-height: 1.6; color: var(--ll-violet);">
                            <b>🚀 Next Horizon:</b> {monthly_report.get('opportunity', 'Aim for 85%+ check-in compliance next month.')}
                        </div>
                    </div>
                    """
                )

            # PDF Export: explicit preparation & deferred user-triggered download
            st.markdown("#### Export Report")
            m_pdf_key = f"cached_pdf_monthly_{m_stats['days_recorded']}_{m_stats['end_score']}_{m_stats['start_score']}"

            if st.session_state.get("show_monthly_pdf_download") and m_pdf_key in st.session_state:
                mdl_col1, mdl_col2 = st.columns([3.2, 0.8], gap="small", vertical_alignment="center")
                with mdl_col1:
                    st.download_button(
                        label="⬇️ Download Monthly Wellness Report (PDF)",
                        data=st.session_state[m_pdf_key],
                        file_name="LifeLens_Monthly_Report.pdf",
                        mime="application/pdf",
                        key="btn_download_monthly_pdf",
                        on_click="ignore",
                        use_container_width=True
                    )
                with mdl_col2:
                    if st.button("Close", key="close_monthly_pdf", use_container_width=True):
                        st.session_state["show_monthly_pdf_download"] = False
                        st.rerun()
            else:
                if st.button("📄 Generate Monthly Wellness Report (PDF)", key="btn_prep_monthly_pdf", type="primary", use_container_width=True):
                    with st.spinner("Generating Monthly PDF Report..."):
                        pdf_data = generate_wellness_pdf(
                            "Monthly",
                            m_stats,
                            period_m_str,
                            ai_text=f"Monthly consistency reached {m_stats['consistency_pct']}% across {m_stats['days_recorded']} recorded check-in days. Net score delta observed: {m_stats['score_delta_str']}.",
                            recommendations=[
                                monthly_report.get("opportunity", "Maintain a balanced sleep and movement schedule.") if monthly_report else "Maintain regular sleep routines.",
                                "Focus on sustainable micro-habits rather than sudden routine disruptions.",
                                "Review monthly correlation heatmaps to recognize triggers between stress and screen time."
                            ]
                        )
                        st.session_state[m_pdf_key] = pdf_data
                        st.session_state["show_monthly_pdf_download"] = True
                        st.rerun()

    # =========================================================================
    # TAB 3: AI WELLNESS ANALYSIS (ON-DEMAND & CACHED)
    # =========================================================================
    with report_tabs[2]:
        st.markdown("### 🤖 V5 AI Wellness Intelligence")
        st.markdown(
            "Synthesize your recent behavioral facts, metric deltas, and habit momentum into actionable, human-like summaries "
            "powered by the fine-tuned FLAN-T5 model. Factual numerical truth is calculated strictly in Python before narrative explanation."
        )

        cache_key = f"cached_ai_analysis_{len(df)}_{df.iloc[-1]['Date'].strftime('%Y%m%d') if not df.empty else 'none'}"

        col_act, col_status = st.columns([1.5, 3], gap="medium", vertical_alignment="center")
        with col_act:
            run_gen = st.button("✨ Generate AI Analysis", type="primary", use_container_width=True)

        if run_gen:
            with st.spinner("🤖 Grounding facts and generating AI synthesis with fine-tuned V5 model..."):
                try:
                    facts = build_weekly_facts(df)
                    analysis_text = explain_weekly_facts(facts)
                    st.session_state[cache_key] = analysis_text
                    st.session_state["cached_reports_ai_analysis"] = analysis_text
                    st.session_state["cached_reports_facts"] = facts
                except Exception as e:
                    st.error(f"AI Generation unavailable: {e}")
                    st.session_state[cache_key] = None

        cached_analysis = st.session_state.get(cache_key) or st.session_state.get("cached_reports_ai_analysis")
        if cached_analysis:
            st.success("✓ AI Synthesis Loaded (Cached from latest execution)")
            st.html(
                f"""
                <div class="ll-card" style="margin-top: 14px;">
                    <div style="font-size: 0.74rem; font-weight: 750; letter-spacing: 0.12em; text-transform: uppercase; color: var(--ll-violet); margin-bottom: 12px;">
                        FLAN-T5 GROUNDED NARRATIVE SYNTHESIS
                    </div>
                    <div style="font-size: 1.02rem; line-height: 1.7; color: var(--ll-text); white-space: pre-line;">
                        {cached_analysis}
                    </div>
                </div>
                """
            )
        else:
            st.info("Click **✨ Generate AI Analysis** above to initiate grounded inference with your current data.")

    # =========================================================================
    # TAB 4: ACTUAL VS PREDICTED ANALYSIS
    # =========================================================================
    with report_tabs[3]:
        st.markdown("### 🎯 Actual Observations vs Machine Learning Forecasts")
        st.markdown(
            "Audit the predictive accuracy of your trained scikit-learn models. "
            "This table strictly compares historical predictions against the actual check-in observations recorded on that exact target date. "
            "**Target dates match observation dates 1:1.** Missing predictions are never displayed as zeros."
        )

        match_df, summary_metrics = get_actual_vs_predicted_data(df)

        if match_df.empty:
            st.html(
                """
                <div class="ll-empty">
                    <span class="ll-empty-icon">🎯</span>
                    <b style="color:var(--ll-text);font-size:1.05rem;">No Matched Prediction Pairs Found Yet</b><br>
                    Predictions are logged when daily tomorrow forecasts are generated.<br>
                    As you record daily check-ins on days with prior forecasts, accuracy audits will appear here.
                </div>
                """
            )
        else:
            # Summary Metrics Cards
            total_target_dates = match_df["TargetDate"].nunique()
            avg_overall_mae = round(match_df["Error"].mean(), 2)

            with st.container(key="avp_metrics"):
                sc1, sc2, sc3 = st.columns(3, gap="small")
                with sc1:
                    st.metric("Target Dates Audited", f"{total_target_dates} dates")
                with sc2:
                    st.metric("Overall Average MAE", f"{avg_overall_mae:.2f}")
                with sc3:
                    st.metric("Accuracy Status", "Calibrated" if avg_overall_mae < 1.8 else "Adapting")

            st.markdown("#### Mean Absolute Error (MAE) by Metric")
            with st.container(key="avp_mae_metrics"):
                mae_cols = st.columns(4, gap="small")
                metrics_list = list(summary_metrics.items())
                for idx, (m_name, m_info) in enumerate(metrics_list):
                    c = mae_cols[idx % 4]
                    with c:
                        st.metric(f"{m_name} Error", f"±{m_info['MAE']:.2f}", help=f"Audited over {m_info['Count']} predictions. Mean Actual: {m_info['MeanActual']}, Mean Predicted: {m_info['MeanPredicted']}")

            st.markdown("#### Historical Predictions vs Actual Observations")
            # Interactive Filter by Metric
            sel_metric = st.selectbox(
                "Filter Comparison by Metric",
                ["All Metrics"] + sorted(match_df["Metric"].unique().tolist()),
                index=0
            )

            display_df = match_df if sel_metric == "All Metrics" else match_df[match_df["Metric"] == sel_metric]

            st.dataframe(
                display_df[[
                    "TargetDate",
                    "PredictionDate",
                    "Metric",
                    "Predicted",
                    "Actual",
                    "Difference",
                    "Error"
                ]].sort_values(["TargetDate", "Metric"], ascending=[False, True]),
                use_container_width=True,
                hide_index=True
            )

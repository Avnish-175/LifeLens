import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_manager import load_data
from src.weekly_report import generate_weekly_report


def _create_compact_trend_chart(df, y_col, title, unit, color):
    """
    Helper to generate a clean, compact line/scatter trend chart for Reports.
    Gracefully handles 1 data point or multiple data points.
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
        if unit != "/10" and unit != "min"
        else f"<b>%{{x|%d %b %Y}}</b><br>{title}: %{{y}}{unit}<extra></extra>"
    )

    fig.update_traces(
        line=dict(width=2.5),
        marker=dict(size=7, line=dict(width=1.5, color="#FFFFFF")),
        hovertemplate=hover_fmt
    )

    fig.update_layout(
        height=220,
        margin=dict(l=15, r=15, t=32, b=15),
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=13, color="#1E3A8A"),
            x=0.01,
            y=0.96
        ),
        xaxis_title="",
        yaxis_title=unit if unit else "",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="system-ui, sans-serif", size=11, color="#475569"),
        hovermode="x unified"
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#F1F5F9",
        tickformat="%d %b" if len(valid_df) > 1 else "%d %b %Y",
        linecolor="#E2E8F0"
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#F1F5F9",
        linecolor="#E2E8F0",
        zeroline=False
    )

    return fig


def _generate_historical_insights(filtered_data, period_label):
    """
    Analyzes earliest vs latest actual recorded values in filtered_data
    and produces 3 deterministic insight cards: Improved, Needs Attention, Biggest Pattern.
    """
    if filtered_data is None or filtered_data.empty or "Date" not in filtered_data.columns:
        return None

    metrics_meta = {
        "SleepHours": ("Sleep Hours", "hrs", True, 8.0, "duration"),
        "Mood": ("Mood Rating", "/10", True, 10.0, "rating"),
        "Energy": ("Energy Level", "/10", True, 10.0, "level"),
        "Productivity": ("Productivity", "/10", True, 10.0, "output"),
        "ExerciseMinutes": ("Exercise Duration", "min", True, 60.0, "time"),
        "ScreenTime": ("Screen Time", "hrs", False, 8.0, "usage"),
        "Stress": ("Stress Level", "/10", False, 10.0, "level"),
    }

    changes = []
    sorted_df = filtered_data.dropna(subset=["Date"]).sort_values("Date")
    if len(sorted_df) < 2:
        return None

    for col, (name, unit, higher_is_better, typical_range, noun) in metrics_meta.items():
        if col in sorted_df.columns:
            valid_series = sorted_df.dropna(subset=[col])
            if len(valid_series) >= 2:
                earliest_val = float(valid_series.iloc[0][col])
                latest_val = float(valid_series.iloc[-1][col])
                diff = latest_val - earliest_val

                # Normalized change for comparison across different scales
                norm_diff = diff / typical_range
                goodness = norm_diff if higher_is_better else -norm_diff

                # Relative change magnitude
                base = abs(earliest_val)
                pct_change = (abs(diff) / (base + 0.5)) * 100

                changes.append({
                    "col": col,
                    "name": name,
                    "unit": unit,
                    "noun": noun,
                    "higher_is_better": higher_is_better,
                    "earliest": earliest_val,
                    "latest": latest_val,
                    "diff": diff,
                    "goodness": goodness,
                    "abs_diff": abs(diff),
                    "pct_change": pct_change,
                })

    if not changes:
        return None

    # Helper formatters
    def fmt_val(val, unit):
        if unit == "min":
            return f"{val:.0f} {unit}"
        elif unit == "hrs":
            return f"{val:.1f} {unit}"
        return f"{val:.1f}{unit}"

    def fmt_delta(diff, unit):
        sign = "+" if diff > 0 else ""
        if unit == "min":
            return f"{sign}{diff:.0f} {unit}"
        elif unit == "hrs":
            return f"{sign}{diff:.1f} {unit}"
        return f"{sign}{diff:.1f}{unit}"

    # 1. Strongest Positive Change
    positive_changes = [c for c in changes if c["goodness"] > 0]
    improved_item = None
    if positive_changes:
        best_pos = max(positive_changes, key=lambda x: x["goodness"])
        diff_str = fmt_delta(best_pos["diff"], best_pos["unit"])
        from_to = f"{fmt_val(best_pos['earliest'], best_pos['unit'])} → {fmt_val(best_pos['latest'], best_pos['unit'])}"

        if best_pos["col"] == "ScreenTime":
            explanation = f"Screen time decreased by {fmt_delta(abs(best_pos['diff']), best_pos['unit'])}, reducing fatigue."
        elif best_pos["col"] == "Stress":
            explanation = f"Stress level dropped by {fmt_delta(abs(best_pos['diff']), best_pos['unit'])}, reflecting calmer days."
        else:
            explanation = f"Your {best_pos['name'].lower()} increased by {diff_str}, showing positive momentum."

        improved_arrow = "↑" if best_pos["diff"] > 0 else "↓"
        improved_item = {
            "title": "⬆️ Improved",
            "metric_title": f"{best_pos['name']} {improved_arrow}",
            "from_to": from_to,
            "delta": diff_str,
            "delta_color": "#059669",
            "bg_tint": "rgba(5, 150, 105, 0.05)",
            "border_color": "#A7F3D0",
            "explanation": explanation
        }
    else:
        improved_item = {
            "title": "⬆️ Improved",
            "metric_title": "Steady Metrics",
            "from_to": "Stable baseline",
            "delta": "0.0",
            "delta_color": "#64748B",
            "bg_tint": "#F8FAFC",
            "border_color": "#E2E8F0",
            "explanation": "No metrics recorded an upward trend over this specific period."
        }

    # 2. Strongest Negative Change
    negative_changes = [c for c in changes if c["goodness"] < 0]
    needs_attention_item = None
    if negative_changes:
        worst_neg = min(negative_changes, key=lambda x: x["goodness"])
        diff_str = fmt_delta(worst_neg["diff"], worst_neg["unit"])
        from_to = f"{fmt_val(worst_neg['earliest'], worst_neg['unit'])} → {fmt_val(worst_neg['latest'], worst_neg['unit'])}"

        if worst_neg["col"] == "ScreenTime":
            explanation = f"Screen time rose by {diff_str}, which may compete with rest."
        elif worst_neg["col"] == "Stress":
            explanation = f"Stress level increased by {diff_str}; consider prioritizing relaxation."
        else:
            explanation = f"Your {worst_neg['name'].lower()} declined by {diff_str}; focus on restoration."

        needs_attention_arrow = "↑" if worst_neg["diff"] > 0 else "↓"
        needs_attention_item = {
            "title": "⬇️ Needs Attention",
            "metric_title": f"{worst_neg['name']} {needs_attention_arrow}",
            "from_to": from_to,
            "delta": diff_str,
            "delta_color": "#DC2626",
            "bg_tint": "rgba(220, 38, 38, 0.04)",
            "border_color": "#FECACA",
            "explanation": explanation
        }
    else:
        needs_attention_item = {
            "title": "⬇️ Needs Attention",
            "metric_title": "All Resilient",
            "from_to": "Healthy levels",
            "delta": "✓ None",
            "delta_color": "#059669",
            "bg_tint": "rgba(5, 150, 105, 0.05)",
            "border_color": "#A7F3D0",
            "explanation": "None of your tracked habits or reflection metrics experienced a decline."
        }

    # 3. Biggest Pattern (Largest relative shift)
    meaningful_changes = [c for c in changes if c["abs_diff"] > 1e-4]
    pattern_item = None
    if meaningful_changes:
        largest_pattern = max(meaningful_changes, key=lambda x: x["pct_change"])
        diff_str = fmt_delta(largest_pattern["diff"], largest_pattern["unit"])
        from_to = f"{fmt_val(largest_pattern['earliest'], largest_pattern['unit'])} → {fmt_val(largest_pattern['latest'], largest_pattern['unit'])}"
        dir_symbol = "↑" if largest_pattern["diff"] > 0 else "↓"

        pattern_item = {
            "title": "🎯 Biggest Pattern",
            "metric_title": f"{largest_pattern['name']} {dir_symbol}",
            "from_to": from_to,
            "delta": f"{diff_str} ({largest_pattern['pct_change']:.0f}%)",
            "delta_color": "#2563EB",
            "bg_tint": "rgba(37, 99, 235, 0.04)",
            "border_color": "#BFDBFE",
            "explanation": f"Showed the most substantial relative fluctuation ({largest_pattern['pct_change']:.0f}% shift) in this timeframe."
        }
    else:
        pattern_item = {
            "title": "🎯 Biggest Pattern",
            "metric_title": "Consistent Routine",
            "from_to": "Steady values",
            "delta": "±0.0",
            "delta_color": "#2563EB",
            "bg_tint": "rgba(37, 99, 235, 0.04)",
            "border_color": "#BFDBFE",
            "explanation": "Your daily habits remained very stable with no wide fluctuations."
        }

    return [improved_item, needs_attention_item, pattern_item]


def _compute_period_comparison(data, period_label, today):
    """
    Computes average metric values for current period vs immediately preceding period.
    Returns (status_type, results_or_message)
    status_type: 'unavailable_all_time' | 'no_previous_data' | 'success'
    """
    if period_label == "All time":
        return "unavailable_all_time", "Period comparison is unavailable for All time as there is no preceding comparative timeframe."

    days_map = {
        "Last 7 days": 7,
        "Last 30 days": 30,
        "Last 90 days": 90,
    }

    n_days = days_map.get(period_label)
    if not n_days:
        return "no_previous_data", "Not enough historical data for period comparison."

    # Current period: [today - (n_days - 1), today + 1 day)
    curr_start = today - pd.Timedelta(days=n_days - 1)
    curr_end = today + pd.Timedelta(days=1)

    # Previous period: [today - (2 * n_days - 1), today - (n_days - 1))
    prev_start = today - pd.Timedelta(days=2 * n_days - 1)
    prev_end = curr_start

    curr_df = data[(data["Date"] >= curr_start) & (data["Date"] < curr_end)].copy()
    prev_df = data[(data["Date"] >= prev_start) & (data["Date"] < prev_end)].copy()

    if prev_df.empty:
        return "no_previous_data", "Not enough historical data for period comparison."

    metrics_def = [
        ("SleepHours", "Sleep", "hrs", True),
        ("Mood", "Mood", "/10", True),
        ("Stress", "Stress", "/10", False),
        ("Energy", "Energy", "/10", True),
        ("ExerciseMinutes", "Exercise", "min", True),
        ("StudyHours", "Study / Work", "hrs", True),
        ("ScreenTime", "Screen Time", "hrs", False),
        ("Productivity", "Productivity", "/10", True),
    ]

    rows = []
    for col, name, unit, higher_is_better in metrics_def:
        if col in data.columns:
            curr_series = curr_df[col].dropna()
            prev_series = prev_df[col].dropna()

            if curr_series.empty and prev_series.empty:
                continue

            curr_avg = curr_series.mean() if not curr_series.empty else None
            prev_avg = prev_series.mean() if not prev_series.empty else None

            if curr_avg is not None and prev_avg is not None:
                diff = curr_avg - prev_avg
                abs_diff = abs(diff)

                if abs_diff < 0.05:
                    dir_label = "→ Steady"
                    badge_style = "color: #64748B; background: #F1F5F9; border: 1px solid #E2E8F0;"
                    diff_str = "0.0"
                else:
                    is_improving = (diff > 0 and higher_is_better) or (diff < 0 and not higher_is_better)
                    if is_improving:
                        dir_label = "↑ Improving"
                        badge_style = "color: #059669; background: rgba(5, 150, 105, 0.08); border: 1px solid rgba(5, 150, 105, 0.2);"
                    else:
                        dir_label = "↓ Declining"
                        badge_style = "color: #DC2626; background: rgba(220, 38, 38, 0.08); border: 1px solid rgba(220, 38, 38, 0.2);"

                    sign = "+" if diff > 0 else "-"
                    if unit == "min":
                        diff_str = f"{sign}{abs_diff:.0f} {unit}"
                    elif unit == "hrs":
                        diff_str = f"{sign}{abs_diff:.1f} {unit}"
                    else:
                        diff_str = f"{sign}{abs_diff:.1f}"

                def fmt_val(v):
                    if unit == "min":
                        return f"{v:.0f} {unit}"
                    elif unit == "hrs":
                        return f"{v:.1f} {unit}"
                    return f"{v:.1f}{unit}"

                curr_str = fmt_val(curr_avg)
                prev_str = fmt_val(prev_avg)
            elif curr_avg is not None:
                curr_str = f"{curr_avg:.1f} {unit}" if unit != "/10" else f"{curr_avg:.1f}/10"
                prev_str = "N/A"
                diff_str = "—"
                dir_label = "— Baseline"
                badge_style = "color: #64748B; background: #F1F5F9; border: 1px solid #E2E8F0;"
            else:
                curr_str = "N/A"
                prev_str = f"{prev_avg:.1f} {unit}" if unit != "/10" else f"{prev_avg:.1f}/10"
                diff_str = "—"
                dir_label = "— Missing"
                badge_style = "color: #64748B; background: #F1F5F9; border: 1px solid #E2E8F0;"

            rows.append({
                "metric": name,
                "current": curr_str,
                "previous": prev_str,
                "change": diff_str,
                "direction": dir_label,
                "badge_style": badge_style
            })

    if not rows:
        return "no_previous_data", "Not enough historical data for period comparison."

    return "success", rows


def show_reports(monthly_report):

    st.subheader("📊 Wellness Reports")
    st.caption("A compact overview of your wellness progress.")

    # Load existing check-in data
    data = load_data()

    if data.empty:
        st.info("No check-in records available yet. Start by recording your daily check-in.")
        return

    # Prepare reporting data
    data = data.copy()

    # Parse dates using the same formats as app.py
    date_values = data["Date"].astype("string").str.strip()

    parsed_dates = pd.to_datetime(
        date_values,
        format="%Y-%m-%d",
        errors="coerce"
    )

    missing_dates = parsed_dates.isna()

    parsed_dates.loc[missing_dates] = pd.to_datetime(
        date_values.loc[missing_dates],
        format="%d-%m-%Y",
        errors="coerce"
    )

    data["Date"] = parsed_dates

    numeric_columns = [
        "SleepHours",
        "Mood",
        "Stress",
        "Energy",
        "ScreenTime",
        "ExerciseMinutes",
        "StudyHours",
        "Productivity",
    ]

    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["Date"])

    if data.empty:
        st.info("No valid check-in dates are available for reporting.")
        return

    # Reporting period
    period = st.selectbox(
        "Reporting period",
        ["Last 7 days", "Last 30 days", "Last 90 days", "All time"],
        index=1,
        key="reports_period",
    )

    today = pd.Timestamp.today().normalize()

    if period == "Last 7 days":
        start_date = today - pd.Timedelta(days=6)
    elif period == "Last 30 days":
        start_date = today - pd.Timedelta(days=29)
    elif period == "Last 90 days":
        start_date = today - pd.Timedelta(days=89)
    else:
        start_date = data["Date"].min().normalize()

    filtered_data = data[
        (data["Date"] >= start_date)
        & (data["Date"] < today + pd.Timedelta(days=1))
    ].copy()

    if filtered_data.empty:
        st.info("No check-ins found in this reporting period.")
        return

    # Key metrics
    total_checkins = len(filtered_data)

    def get_average(column):
        if column in filtered_data.columns:
            return filtered_data[column].mean()
        return None

    metrics = [
        ("📅 Total Check-ins", str(total_checkins)),
        ("😴 Average Sleep", f"{get_average('SleepHours'):.1f} hrs" if pd.notna(get_average("SleepHours")) else "N/A"),
        ("😊 Average Mood", f"{get_average('Mood'):.1f}/10" if pd.notna(get_average("Mood")) else "N/A"),
        ("🧘 Average Stress", f"{get_average('Stress'):.1f}/10" if pd.notna(get_average("Stress")) else "N/A"),
        ("⚡ Average Energy", f"{get_average('Energy'):.1f}/10" if pd.notna(get_average("Energy")) else "N/A"),
        ("🏃 Daily Exercise", f"{get_average('ExerciseMinutes'):.0f} min" if pd.notna(get_average("ExerciseMinutes")) else "N/A"),
        ("📚 Study Time", f"{get_average('StudyHours'):.1f} hrs" if pd.notna(get_average("StudyHours")) else "N/A"),
        ("🎯 Productivity", f"{get_average('Productivity'):.1f}/10" if pd.notna(get_average("Productivity")) else "N/A"),
    ]

    st.markdown("#### Your Wellness Metrics")

    with st.container(key="reports_metrics"):
        for row_start in range(0, len(metrics), 4):
            cols = st.columns(4, gap="small")

            for col, (label, value) in zip(
                cols, metrics[row_start:row_start + 4]
            ):
                with col:
                    st.metric(label, value)

    # -------------------------------------------------------------------------
    # Wellness Trends (Daily trends across actual stored metrics)
    # -------------------------------------------------------------------------
    st.markdown("#### Wellness Trends")

    if total_checkins < 2:
        st.info("💡 Record check-ins across multiple days to reveal continuous trend curves. Showing single check-in point below:")

    # 1. Daily Wellness Reflections Trends
    st.markdown(
        """
        <div style="font-size: 13.5px; font-weight: 600; color: #334155; margin-top: 6px; margin-bottom: 6px;">
            Daily Wellness Trends
        </div>
        """,
        unsafe_allow_html=True
    )

    wellness_configs = [
        ("SleepHours", "😴 Sleep Hours", "hrs", "#4F46E5"),
        ("Mood", "😊 Mood Rating", "/10", "#059669"),
        ("Energy", "⚡ Energy Level", "/10", "#D97706"),
        ("Stress", "🧘 Stress Level", "/10", "#DC2626"),
        ("Productivity", "🎯 Productivity", "/10", "#2563EB"),
    ]

    active_wellness_charts = []
    for col_name, title, unit, color in wellness_configs:
        if col_name in filtered_data.columns and filtered_data[col_name].notna().any():
            fig = _create_compact_trend_chart(filtered_data, col_name, title, unit, color)
            if fig is not None:
                active_wellness_charts.append(fig)

    for i in range(0, len(active_wellness_charts), 2):
        pair = active_wellness_charts[i:i + 2]
        if len(pair) == 2:
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                st.plotly_chart(pair[0], use_container_width=True)
            with c2:
                st.plotly_chart(pair[1], use_container_width=True)
        else:
            c1, _ = st.columns(2, gap="medium")
            with c1:
                st.plotly_chart(pair[0], use_container_width=True)

    # 2. Habit Trends
    st.markdown(
        """
        <div style="font-size: 13.5px; font-weight: 600; color: #334155; margin-top: 10px; margin-bottom: 6px;">
            Habit Trends
        </div>
        """,
        unsafe_allow_html=True
    )

    habit_configs = [
        ("ScreenTime", "💻 Screen Time", "hrs", "#0284C7"),
        ("ExerciseMinutes", "🏃 Exercise Duration", "min", "#16A34A"),
        ("StudyHours", "📚 Study / Focused Work", "hrs", "#7C3AED"),
    ]

    active_habit_charts = []
    for col_name, title, unit, color in habit_configs:
        if col_name in filtered_data.columns and filtered_data[col_name].notna().any():
            fig = _create_compact_trend_chart(filtered_data, col_name, title, unit, color)
            if fig is not None:
                active_habit_charts.append(fig)

    for i in range(0, len(active_habit_charts), 2):
        pair = active_habit_charts[i:i + 2]
        if len(pair) == 2:
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                st.plotly_chart(pair[0], use_container_width=True)
            with c2:
                st.plotly_chart(pair[1], use_container_width=True)
        else:
            c1, _ = st.columns(2, gap="medium")
            with c1:
                st.plotly_chart(pair[0], use_container_width=True)

    # -------------------------------------------------------------------------
    # 📈 What Changed? (Historical Insights Section)
    # -------------------------------------------------------------------------
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 18px; margin-bottom: 8px;">
            <div style="font-size: 17px; font-weight: 700; color: #0F172A; letter-spacing: -0.01em;">
                📈 What Changed?
            </div>
            <div style="font-size: 12px; color: #64748B;">
                Period: <span style="font-weight: 600; color: #2563EB;">{period}</span> · Comparing earliest to latest records
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if total_checkins < 2:
        st.info("More check-ins are needed to identify meaningful changes across this period.")
    else:
        insights = _generate_historical_insights(filtered_data, period)
        if not insights:
            st.info("No comparative metric values are available in the selected period.")
        else:
            card_cols = st.columns(3, gap="medium")
            for col, item in zip(card_cols, insights):
                with col:
                    st.markdown(
                        f"""
                        <div style="
                            background: {item['bg_tint']};
                            border: 1px solid {item['border_color']};
                            border-radius: 12px;
                            padding: 14px 16px;
                            height: 100%;
                            display: flex;
                            flex-direction: column;
                            justify-content: space-between;
                        ">
                            <div>
                                <div style="font-size: 11px; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 4px;">
                                    {item['title']}
                                </div>
                                <div style="font-size: 15px; font-weight: 750; color: #0F172A; margin-bottom: 4px;">
                                    {item['metric_title']}
                                </div>
                                <div style="display: flex; align-items: baseline; gap: 8px; margin-bottom: 6px;">
                                    <span style="font-size: 12.5px; font-weight: 600; color: #334155;">{item['from_to']}</span>
                                    <span style="font-size: 12px; font-weight: 700; color: {item['delta_color']};">{item['delta']}</span>
                                </div>
                            </div>
                            <div style="font-size: 12px; color: #475569; line-height: 1.45; border-top: 1px solid rgba(0,0,0,0.06); padding-top: 6px; margin-top: 4px;">
                                {item['explanation']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    # -------------------------------------------------------------------------
    # 📊 Period Comparison (Current Period vs Preceding Equivalent Period)
    # -------------------------------------------------------------------------
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 22px; margin-bottom: 8px;">
            <div style="font-size: 17px; font-weight: 700; color: #0F172A; letter-spacing: -0.01em;">
                📊 Period Comparison
            </div>
            <div style="font-size: 12px; color: #64748B;">
                Current: <span style="font-weight: 600; color: #2563EB;">{period}</span> vs Immediately Preceding Period
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    status_type, comparison_data = _compute_period_comparison(data, period, today)

    if status_type == "unavailable_all_time":
        st.info("Period comparison is unavailable for All time as there is no preceding comparative timeframe.")
    elif status_type == "no_previous_data":
        st.info("Not enough historical data for period comparison.")
    else:
        # Build clean, responsive HTML comparison table (no leading indentation so markdown does not parse as code)
        table_rows = []
        for row in comparison_data:
            table_rows.append(
                f'<tr style="border-bottom: 1px solid #F1F5F9;">'
                f'<td style="padding: 10px 14px; font-weight: 600; color: #1E293B; font-size: 13.5px;">{row["metric"]}</td>'
                f'<td style="padding: 10px 14px; font-weight: 700; color: #0F172A; font-size: 13px;">{row["current"]}</td>'
                f'<td style="padding: 10px 14px; color: #64748B; font-size: 13px;">{row["previous"]}</td>'
                f'<td style="padding: 10px 14px; font-weight: 600; font-size: 13px;">{row["change"]}</td>'
                f'<td style="padding: 10px 14px;"><span style="display: inline-block; padding: 2px 8px; border-radius: 6px; font-size: 11.5px; font-weight: 700; {row["badge_style"]}">{row["direction"]}</span></td>'
                f'</tr>'
            )
        table_rows_html = "".join(table_rows)

        table_html = (
            '<div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; overflow-x: auto; box-shadow: 0 1px 3px rgba(15,23,42,0.03); margin-bottom: 14px;">'
            '<table style="width: 100%; border-collapse: collapse; text-align: left; min-width: 500px;">'
            '<thead>'
            '<tr style="background: #F8FAFC; border-bottom: 1px solid #E2E8F0;">'
            '<th style="padding: 10px 14px; font-size: 11.5px; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.04em;">Metric</th>'
            '<th style="padding: 10px 14px; font-size: 11.5px; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.04em;">Current Average</th>'
            '<th style="padding: 10px 14px; font-size: 11.5px; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.04em;">Previous Average</th>'
            '<th style="padding: 10px 14px; font-size: 11.5px; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.04em;">Change</th>'
            '<th style="padding: 10px 14px; font-size: 11.5px; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.04em;">Direction</th>'
            '</tr>'
            '</thead>'
            f'<tbody>{table_rows_html}</tbody>'
            '</table>'
            '</div>'
        )
        st.markdown(table_html, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Compact weekly and monthly summaries
    # -------------------------------------------------------------------------
    st.markdown("#### Wellness Summaries")

    weekly_start = today - pd.Timedelta(days=6)

    weekly_data = data[
        (data["Date"] >= weekly_start)
        & (data["Date"] < today + pd.Timedelta(days=1))
    ].copy()

    weekly_report = generate_weekly_report(weekly_data)

    weekly_col, monthly_col = st.columns(2, gap="medium")

    with weekly_col:
        with st.container(key="reports_weekly_summary", border=True):
            st.markdown("**📅 Weekly Summary**")

            if weekly_report:
                st.markdown(
                    weekly_report["headline"]
                )

                st.caption(
                    f"🏆 Achievement: {weekly_report['badge']}"
                )

                st.caption(
                    f"🎯 Focus: {weekly_report['mission']}"
                )
            else:
                st.caption(
                    "Not enough recent check-ins for a weekly summary."
                )

    with monthly_col:
        with st.container(key="reports_monthly_summary", border=True):
            st.markdown("**🗓️ Monthly Summary**")

            if monthly_report:
                st.caption(
                    f"🏃 Active on {monthly_report['exercise_days']} "
                    "of your last 30 check-ins."
                )

                st.markdown(
                    f"**💚 Biggest Win:** {monthly_report['biggest_win']}"
                )

                st.markdown(
                    f"**⚠️ Needs Attention:** {monthly_report['challenge']}"
                )

                st.markdown(
                    f"**🚀 Next Opportunity:** {monthly_report['opportunity']}"
                )
            else:
                st.caption("Monthly summary is not available yet.")

    # CSV export
    with st.container(key="reports_download"):
        st.download_button(
            label="⬇️ Download Wellness Report (CSV)",
            data=filtered_data.to_csv(index=False).encode("utf-8"),
            file_name="lifelens_wellness_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

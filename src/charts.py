import math
import pandas as pd
def plot_sleep_trend(data):
    data = data.sort_values("Date")

    fig = px.line(
        data,
        x="Date",
        y="SleepHours",
        markers=True,
        title="😴 Sleep Trend",
        color_discrete_sequence=["#6366F1"],
        labels={"SleepHours": "Sleep", "Date": "Date"},
    )

    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=8),
        hovertemplate=(
            "<b>%{x|%d %b %Y}</b><br>"
            "Sleep: %{y:.1f} hours"
            "<extra></extra>"
        ),
    )

    fig.update_layout(
        height=360,
        template="plotly_white",
        title=dict(
            x=0.02,
            xanchor="left",
            font=dict(size=19, color="#1E3A8A"),
        ),
        margin=dict(l=20, r=20, t=55, b=25),
        xaxis_title="",
        yaxis_title="Hours",
        hovermode="x unified",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Arial, sans-serif", color="#334155"),
    )

    fig.update_xaxes(gridcolor="#F1F5F9")
    fig.update_yaxes(gridcolor="#E2E8F0", zeroline=False)

    return fig


def plot_mood_trend(data):
    data = data.sort_values("Date")

    fig = px.line(
        data,
        x="Date",
        y="Mood",
        markers=True,
        title="😊 Mood Trend",
        color_discrete_sequence=["#0F766E"],
        labels={"Mood": "Mood", "Date": "Date"},
    )

    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=8),
        hovertemplate=(
            "<b>%{x|%d %b %Y}</b><br>"
            "Mood: %{y:.1f}/10"
            "<extra></extra>"
        ),
    )

    fig.update_layout(
        height=360,
        template="plotly_white",
        title=dict(
            x=0.02,
            xanchor="left",
            font=dict(size=19, color="#1E3A8A"),
        ),
        margin=dict(l=20, r=20, t=55, b=25),
        xaxis_title="",
        yaxis_title="Mood Score",
        hovermode="x unified",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Arial, sans-serif", color="#334155"),
    )

    fig.update_xaxes(gridcolor="#F1F5F9")
    fig.update_yaxes(
        range=[0, 10],
        gridcolor="#E2E8F0",
        zeroline=False,
    )

    return fig


def plot_stress_trend(data):
    data = data.sort_values("Date")

    fig = px.line(
        data,
        x="Date",
        y="Stress",
        markers=True,
        title="😌 Stress Trend",
        color_discrete_sequence=["#D97706"],
        labels={"Stress": "Stress", "Date": "Date"},
    )

    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=8),
        hovertemplate=(
            "<b>%{x|%d %b %Y}</b><br>"
            "Stress: %{y:.1f}/10"
            "<extra></extra>"
        ),
    )

    fig.update_layout(
        height=360,
        template="plotly_white",
        title=dict(
            x=0.02,
            xanchor="left",
            font=dict(size=19, color="#1E3A8A"),
        ),
        margin=dict(l=20, r=20, t=55, b=25),
        xaxis_title="",
        yaxis_title="Stress Level",
        hovermode="x unified",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Arial, sans-serif", color="#334155"),
    )

    fig.update_xaxes(gridcolor="#F1F5F9")
    fig.update_yaxes(
        range=[0, 10],
        gridcolor="#E2E8F0",
        zeroline=False,
    )

    return fig


def plot_screen_time_trend(data):
    data = data.sort_values("Date")

    fig = px.line(
        data,
        x="Date",
        y="ScreenTime",
        markers=True,
        title="📱 Screen Time Trend",
        color_discrete_sequence=["#2563EB"],
        labels={"ScreenTime": "Screen Time", "Date": "Date"},
    )

    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=8),
        hovertemplate=(
            "<b>%{x|%d %b %Y}</b><br>"
            "Screen Time: %{y:.1f} hours"
            "<extra></extra>"
        ),
    )

    fig.update_layout(
        height=360,
        template="plotly_white",
        title=dict(
            x=0.02,
            xanchor="left",
            font=dict(size=19, color="#1E3A8A"),
        ),
        margin=dict(l=20, r=20, t=55, b=25),
        xaxis_title="",
        yaxis_title="Hours",
        hovermode="x unified",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Arial, sans-serif", color="#334155"),
    )

    fig.update_xaxes(gridcolor="#F1F5F9")
    fig.update_yaxes(gridcolor="#E2E8F0", zeroline=False)

    return fig


def plot_correlation_heatmap(correlation_matrix):

    fig = px.imshow(
        correlation_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto"
    )

    fig.update_traces(
        hovertemplate=
        "<b>%{y}</b> ↔ <b>%{x}</b><br>"
        "Correlation: %{z:.2f}<extra></extra>"
    )

    fig.update_layout(

        title=dict(
            text="🧩 Habit Relationship Matrix",
            x=0.02,
            xanchor="left",
            font=dict(size=22)
        ),

        template="plotly_white",

        height=560,

        margin=dict(
            l=20,
            r=20,
            t=70,
            b=20
        ),

        coloraxis_colorbar=dict(
            title="Correlation",
            thickness=18,
            len=0.75
        ),

        xaxis=dict(
            side="bottom",
            tickfont=dict(size=12)
        ),

        yaxis=dict(
            tickfont=dict(size=12)
        )
    )

    return fig

import plotly.express as px


def create_wellness_breakdown_chart(score_breakdown):
    labels = []
    scores = []

    for factor, score, status in score_breakdown:
        labels.append(factor)
        scores.append(score)

    fig = px.bar(
        x=scores,
        y=labels,
        orientation="h",
        text=scores,
    )

    fig.update_layout(
        title="Wellness Breakdown",
        height=350,
        xaxis_title="Score",
        yaxis_title="",
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
    )

    fig.update_traces(textposition="outside")

    return fig


import plotly.graph_objects as go



def create_wellness_radar_chart(score_breakdown, recorded_values=None):
    if recorded_values is None:
        recorded_values = {}

    categories = []
    values = []
    hover_details = []

    for factor, score, status in score_breakdown:
        if score is None:
            continue

        try:
            score = float(score)
        except (TypeError, ValueError):
            continue

        if not math.isfinite(score):
            continue

        # Remove a leading emoji when present, without requiring one.
        name = factor.split(" ", 1)[-1] if " " in factor else factor

        categories.append(name)
        values.append(max(0, min(100, score)))

        # Match the chart label to the original CSV column name.
        column_name = {
            "Sleep": "SleepHours",
            "Mood": "Mood",
            "Stress": "Stress",
            "Energy": "Energy",
            "Exercise": "ExerciseMinutes",
            "Screen Time": "ScreenTime",
            "Productivity": "Productivity",
        }.get(name, name)

        raw_value = recorded_values.get(column_name)

        if raw_value is None:
            recorded_text = "Not available"
        elif name == "Sleep":
            recorded_text = f"{raw_value} hours"
        elif name == "Exercise":
            recorded_text = f"{raw_value} minutes"
        elif name == "Screen Time":
            recorded_text = f"{raw_value} hours"
        elif name in ("Mood", "Stress", "Energy", "Productivity"):
            recorded_text = f"{raw_value}/10"
        else:
            recorded_text = str(raw_value)

        hover_details.append(
            f"Recorded value: {recorded_text}<br>Status: {status}"
        )
    # An empty chart is better than a crash or a misleading polygon.
    if not categories:
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough data to display the wellness chart yet.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=14, color="#64748B"),
        )
        fig.update_layout(
            height=420,
            template="plotly_white",
            showlegend=False,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        return fig

    # Close the radar shape only after confirming that data exists.
    categories.append(categories[0])
    values.append(values[0])
    hover_details.append(hover_details[0])

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name="Available wellness factors",
            line=dict(color="#2563EB", width=2),
            fillcolor="rgba(37, 99, 235, 0.20)",
            marker=dict(size=6, color="#2563EB"),
            customdata=hover_details,
            hovertemplate=(
                "<b>%{theta}</b><br>"
                "Wellness score: %{r:.0f}/100<br>"
                "%{customdata}"
                "<extra></extra>"
            ),
        )
    )

    
    fig.update_layout(
        polar=dict(
            bgcolor="#FFFFFF",
            domain=dict(x=[0.08, 0.92], y=[0.08, 0.92]),
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[20, 40, 60, 80, 100],
                gridcolor="#D9E2EC",
                linecolor="#CBD5E1",
                tickfont=dict(size=12, color="#475569"),
                title=dict(font=dict(size=13, color="#475569")),
            ),
            angularaxis=dict(
                gridcolor="#D9E2EC",
                linecolor="#CBD5E1",
                tickfont=dict(size=14, color="#1E293B"),
            ),
        ),
        showlegend=False,
        height=520,
        autosize=True,
        margin=dict(l=10, r=10, t=15, b=15),
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Arial, sans-serif",
            size=14,
            color="#26364A",
        ),
    )

    return fig

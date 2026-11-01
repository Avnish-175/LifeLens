import streamlit as st


def show_correlation_card(
    title,
    title_color,
    progress_color,
    icon,
    correlation,
    explanation,
):
    if not isinstance(correlation, dict):
        return

    feature1 = correlation.get("feature1", "Habit A")
    feature2 = correlation.get("feature2", "Habit B")
    value = correlation.get("value")

    # Safely compute percent if missing
    percent = correlation.get("percent")
    if percent is None:
        if value is not None:
            try:
                percent = int(round(abs(float(value)) * 100))
            except (ValueError, TypeError):
                percent = 0
        else:
            percent = 0

    # Ensure percent is bounded 0-100
    try:
        percent = max(0, min(100, int(percent)))
    except (ValueError, TypeError):
        percent = 0

    # Safely compute strength if missing
    strength = correlation.get("strength")
    if not strength:
        if value is not None:
            try:
                abs_v = abs(float(value))
                if abs_v >= 0.90:
                    strength = "Very Strong"
                elif abs_v >= 0.70:
                    strength = "Strong"
                elif abs_v >= 0.50:
                    strength = "Moderate"
                else:
                    strength = "Weak"
            except (ValueError, TypeError):
                strength = "Moderate"
        else:
            strength = "Moderate"

    st.html(
        f"""
        <div class="correlation-card">
            <div class="correlation-title" style="color:{title_color};">
                {icon} {title}
            </div>
            <div class="correlation-features">
                {feature1} ↔ {feature2}
            </div>
            <div class="correlation-bar">
                <div class="correlation-progress" style="width:{percent}%; background:{progress_color};"></div>
            </div>
            <div class="correlation-strength" style="color:{title_color};">
                {icon} {strength} • {percent}%
            </div>
            <div class="correlation-text">
                {explanation}
            </div>
        </div>
        """
    )


import streamlit as st


def show_about():

    # Hero section
    with st.container(key="about_hero"):
        st.markdown(
            """
            <div class="about-hero-content">
                <div class="about-hero-label">PERSONAL BEHAVIOUR INTELLIGENCE</div>
                <h1>🌱 About LifeLens</h1>
                <h3>Understand Your Habits. Improve Your Life.</h3>
                <p>
                    A personal wellness platform designed to turn everyday
                    habits into meaningful insights and encourage better
                    lifestyle decisions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Project Overview
    st.markdown('<div class="about-section-title">01 — Project Overview</div>',
                unsafe_allow_html=True)

    
    st.html(
        """
        <div class="about-text-card">
            <p>
                LifeLens transforms everyday behavioural data into meaningful
                information about personal wellbeing. By tracking factors such
                as sleep, mood, stress, energy, physical activity, study time,
                screen time, and productivity, the platform helps users
                understand how their daily routines influence their lifestyle.
            </p>

            <p>
                Rather than simply recording daily activities, LifeLens focuses
                on identifying behavioural patterns, highlighting areas for
                improvement, and encouraging consistent positive habits.
            </p>
        </div>
        """
    )

    # Why LifeLens?
    st.markdown('<div class="about-section-title">02 — Why LifeLens?</div>',
                unsafe_allow_html=True)

    
    st.html(
        """
        <div class="about-text-card">
            <p>
                In our busy daily lives, we often overlook how small habits
                influence our physical wellbeing, mental state, and productivity.
            </p>

            <p>
                LifeLens brings important lifestyle indicators together in one
                platform. It encourages users to become more aware of their
                routines, recognise patterns in their behaviour, and take
                practical steps towards a healthier and more balanced lifestyle.
            </p>
        </div>
        """
    )


    # Key Features
    st.markdown('<div class="about-section-title">03 — Key Features</div>',
                unsafe_allow_html=True)

    features = [
        ("🌿", "Daily Wellness Tracking",
         "Record sleep, mood, stress, energy, exercise, study hours, screen time, and productivity."),
        ("📊", "Wellness Overview",
         "View an overall picture of daily wellbeing through visual indicators and a wellness score."),
        ("🧠", "AI-Powered Insights",
         "Discover behavioural observations and personalised suggestions."),
        ("📈", "Behaviour Trend Analysis",
         "Monitor changes in lifestyle indicators and understand patterns over time."),
        ("🎯", "Personal Action Plans",
         "Receive practical recommendations based on individual behavioural patterns."),
        ("🔮", "Wellness Forecast",
         "Explore estimated next-day wellness indicators."),
    ]

    
    for row_start in range(0, len(features), 3):
        feature_cols = st.columns(3, gap="medium")

        for col, feature in zip(
            feature_cols,
            features[row_start:row_start + 3]
        ):
            icon, title, description = feature

            with col:
                st.markdown(
                    f"""
                    <div class="about-feature-card">
                        <div class="about-feature-icon">{icon}</div>
                        <div class="about-feature-title">{title}</div>
                        <div class="about-feature-description">
                            {description}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # How it works
    st.markdown('<div class="about-section-title">04 — How LifeLens Works</div>',
                unsafe_allow_html=True)

    steps = [
        ("01", "Record", "Enter daily lifestyle and wellness information."),
        ("02", "Analyse", "Process recorded information to calculate wellness indicators."),
        ("03", "Discover", "Explore trends, relationships, and behavioural insights."),
        ("04", "Improve", "Use recommendations to build healthier routines."),
    ]

    step_cols = st.columns(4, gap="small")

    for col, (number, title, description) in zip(step_cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="about-step-card">
                    <div class="about-step-number">{number}</div>
                    <div class="about-step-title">{title}</div>
                    <div class="about-step-description">{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Our Vision
    st.markdown('<div class="about-section-title">05 — Our Vision</div>',
                unsafe_allow_html=True)

    st.markdown(
        """
        <div class="about-vision-card">
            <h3>Every Habit Tells a Story.</h3>
            <p>
                Our vision is to make personal wellbeing more understandable
                through data and intelligent analysis. LifeLens encourages
                individuals to move beyond simply tracking their habits and
                start understanding them.
            </p>
            <p>
                By turning everyday information into meaningful insights,
                we aim to support better self-awareness, healthier routines,
                and continuous personal improvement.
            </p>
            <div class="about-vision-footer">LIFELENS · PERSONAL WELLNESS INTELLIGENCE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

import streamlit as st


def show_about():
    """
    Renders the LifeLens About Page featuring 8 comprehensive modern sections:
    1. Hero
    2. Project Overview
    3. Core Features (8 cards)
    4. How LifeLens Works (8-step architecture pipeline)
    5. Machine Learning Pipeline (7 targets & scikit-learn models)
    6. AI / NLP Explanation Layer (FLAN-T5 grounded reasoning)
    7. Modern Technology Stack
    8. Project Vision & Limitations (Health Disclaimer)
    """

    # -------------------------------------------------------------
    # 01 — HERO SECTION
    # -------------------------------------------------------------
    st.markdown(
        """
        <div class="about-hero-box">
            <span class="about-hero-eyebrow">Personal Behaviour Intelligence</span>
            <h1 class="about-hero-title">🌱 LifeLens – Understand Your Habits. Discover Your Patterns. Improve Your Wellbeing.</h1>
            <p class="about-hero-lead">
                A personal wellness intelligence platform turning everyday lifestyle habits
                into actionable clarity, predictive foresight, and grounded behavioral guidance.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------
    # 02 — PROJECT OVERVIEW
    # -------------------------------------------------------------
    st.markdown(
        """
        <div class="about-section-header">
            <span class="about-sec-num">01</span>
            <h2 class="about-sec-title">Project Overview</h2>
        </div>
        <div class="about-overview-card">
            <p>
                In fast-paced modern routines, subtle lifestyle habits—irregular sleep cycles,
                prolonged screen exposure, fluctuating workplace stress, and inconsistent physical
                movement—compound quietly over days and weeks. Because these subtle behavioral shifts
                accumulate gradually, individuals often struggle to identify why their energy dips or
                productivity stalls until burnout occurs.
            </p>
            <p>
                <strong>Why tracking matters:</strong> Memory is subjective, but data is objective.
                LifeLens bridges the gap between daily lifestyle recording and intelligent analysis.
                By tracking 8 vital lifestyle indicators—sleep duration, subjective mood, stress levels,
                vital energy, physical exercise, productive hours, study time, and screen consumption—the
                system replaces guesswork with verifiable behavioral patterns.
            </p>
            <p>
                <strong>Data-driven wellness:</strong> Rather than serving as an isolated digital diary,
                LifeLens integrates deterministic scoring formulas, interactive multidimensional charts,
                scikit-learn machine learning regression models, and a fine-tuned FLAN-T5 natural language
                model to transform raw numbers into compassionate, actionable lifestyle intelligence.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------
    # 03 — CORE FEATURES (8 Polished Cards)
    # -------------------------------------------------------------
    st.markdown(
        """
        <div class="about-section-header">
            <span class="about-sec-num">02</span>
            <h2 class="about-sec-title">Core System Features</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    features = [
        ("🌿", "Daily Wellness Tracking", "Fast, frictionless multi-dimensional check-in capturing sleep, mood, energy, stress, exercise, study, screen time, and qualitative reflections."),
        ("🎯", "Wellness Score Synthesis", "Normalized composite score (0–100) combining weighted metric evaluations, penalty adjustments, and categorical grading (Excellent, Healthy, Needs Improvement, At Risk)."),
        ("📈", "Historical Trends & Analytics", "Dynamic rolling 7-day, 14-day, and 30-day analytics with interactive dual-theme Plotly visualizations tracking long-term behavioral trajectories."),
        ("🔗", "Correlation & Pattern Mining", "Pearson correlation matrices revealing how physical habits (screen time, rest, exercise) directly govern mental states (stress, mood, and focus)."),
        ("🔮", "ML Tomorrow Forecasting", "Multi-target scikit-learn regression pipeline estimating next-day lifestyle indicators based on recent momentum and rolling behavioral history."),
        ("🤖", "V5 AI Natural Language Reports", "Fine-tuned FLAN-T5 model synthesizing computed facts and deltas into human-like daily reports, weekly summaries, and priority actions."),
        ("💡", "Personalized Action Plans", "Dynamic behavioral nudges highlighting your highest-impact wellness opportunity each day with concrete micro-habits."),
        ("🛡️", "Data Validation & Management", "Atomic CSV persistence, robust date normalization, range bounds validation, and safe duplicate check-in collision guards."),
    ]

    # Render in two rows of 4 columns
    for row_idx in range(0, len(features), 4):
        cols = st.columns(4, gap="medium")
        for col, (icon, title, desc) in zip(cols, features[row_idx:row_idx + 4]):
            with col:
                st.markdown(
                    f"""
                    <div class="about-feature-box">
                        <div class="about-feature-icon">{icon}</div>
                        <div class="about-feature-h">{title}</div>
                        <div class="about-feature-p">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # -------------------------------------------------------------
    # 04 — HOW LIFELENS WORKS (8-Step Architecture Pipeline)
    # -------------------------------------------------------------
    st.markdown(
        """
        <div class="about-section-header">
            <span class="about-sec-num">03</span>
            <h2 class="about-sec-title">How LifeLens Works: Step-by-Step Architecture</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    steps = [
        ("01", "Daily Check-in", "User records subjective ratings (mood, energy, stress) and objective metrics (sleep, screen time, exercise, study)."),
        ("02", "Input Validation", "Data guards enforce valid ranges, numeric conversions, non-empty fields, and confirm overwrite on duplicate dates."),
        ("03", "Data Persistence", "Check-in data is normalized to ISO calendar dates and securely persisted to atomic local CSV storage."),
        ("04", "Score Synthesis", "Mathematical engine calculates the multi-component Wellness Score and determines the holistic health category."),
        ("05", "Historical Analytics", "Pandas and Plotly compute moving averages, period-over-period deltas, and Pearson correlation matrices."),
        ("06", "ML Forecasting", "Scikit-learn multi-output regression models predict estimated next-day values across 7 core lifestyle dimensions."),
        ("07", "NLP Grounding", "Computed facts, deltas, and forecasts are structured into prompt facts for the fine-tuned FLAN-T5 model."),
        ("08", "Actionable Guidance", "Action plans and behavioral suggestions are presented to the user to optimize habits for the day ahead."),
    ]

    st.markdown('<div class="about-workflow-grid">', unsafe_allow_html=True)
    w_cols = st.columns(4, gap="small")
    for i in range(4):
        num, name, desc = steps[i]
        with w_cols[i]:
            st.markdown(
                f"""
                <div class="about-step-node">
                    <div class="about-step-idx">Step {num}</div>
                    <div class="about-step-name">{name}</div>
                    <div class="about-step-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    w_cols2 = st.columns(4, gap="small")
    for i in range(4, 8):
        num, name, desc = steps[i]
        with w_cols2[i - 4]:
            st.markdown(
                f"""
                <div class="about-step-node">
                    <div class="about-step-idx">Step {num}</div>
                    <div class="about-step-name">{name}</div>
                    <div class="about-step-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 05 — MACHINE LEARNING PIPELINE
    # -------------------------------------------------------------
    st.markdown(
        """
        <div class="about-section-header">
            <span class="about-sec-num">04</span>
            <h2 class="about-sec-title">Machine Learning Pipeline (Predictive Forecasting)</h2>
        </div>
        <div class="about-deep-card">
            <p>
                LifeLens employs a dedicated machine learning pipeline built using <strong>scikit-learn</strong>
                regression models to project your trajectory for the upcoming day. The system forecasts
                <strong>7 interconnected lifestyle targets</strong> based on recent history, rolling momentum,
                and cross-metric feature interactions:
            </p>
            <div class="about-targets-grid">
                <span class="about-target-pill">😊 Mood (/10)</span>
                <span class="about-target-pill">⚡ Energy (/10)</span>
                <span class="about-target-pill">🧘 Stress (/10)</span>
                <span class="about-target-pill">🎯 Productivity (/10)</span>
                <span class="about-target-pill">😴 Sleep Hours</span>
                <span class="about-target-pill">📱 Screen Time (hrs)</span>
                <span class="about-target-pill">🏃 Exercise (mins)</span>
            </div>
            <p>
                <strong>Feature Engineering:</strong> The model extracts multi-day lag features, rolling 3-day and
                7-day moving averages, weekend flags, and habit volatility indicators to understand temporal
                patterns unique to your rhythm.
            </p>
            <p>
                <strong>Statistical Estimates, Not Guarantees:</strong> Machine learning forecasts are designed
                to serve as forward-looking indicators of current behavioral momentum. They are probabilistic
                approximations intended to spark proactive habit adjustments, not deterministic prophecies.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------
    # 06 — AI / NLP EXPLANATION LAYER
    # -------------------------------------------------------------
    st.markdown(
        """
        <div class="about-section-header">
            <span class="about-sec-num">05</span>
            <h2 class="about-sec-title">AI & NLP Explanation Layer (Grounded Intelligence)</h2>
        </div>
        <div class="about-deep-card">
            <p>
                A core design principle of LifeLens is the strict boundary between
                <strong>mathematical ground truth</strong> and <strong>natural-language explanation</strong>.
            </p>
            <p>
                <strong>Deterministic Foundation:</strong> The AI model never calculates scores, estimates
                correlations, or guesses metrics. All wellness scores, percentage deltas, moving averages,
                and statistical thresholds are computed deterministically in verified Python code.
            </p>
            <p>
                <strong>Fine-Tuned FLAN-T5 Synthesis:</strong> Our custom fine-tuned V5 FLAN-T5 language model
                operates purely as an empathetic translator and synthesizer. It receives strictly grounded,
                verifiable fact tables and generates clear, structured, and compassionate human-readable
                explanations divided into concise sections: Overall Outlook, Key Strengths, Notable Patterns,
                and Actionable Priorities.
            </p>
            <p>
                This architecture eliminates LLM hallucinations, ensuring every insight you read is 100%
                supported by your recorded lifestyle data.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------
    # 07 — MODERN TECHNOLOGY STACK
    # -------------------------------------------------------------
    st.markdown(
        """
        <div class="about-section-header">
            <span class="about-sec-num">06</span>
            <h2 class="about-sec-title">Modern Technology Stack</h2>
        </div>
        <div class="about-tech-grid">
            <div class="about-tech-item">
                <div class="about-tech-item-name">🐍 Python 3.10+</div>
                <div class="about-tech-item-role">Core backend programming language, mathematical processing, and system orchestration.</div>
            </div>
            <div class="about-tech-item">
                <div class="about-tech-item-name">⚡ Streamlit</div>
                <div class="about-tech-item-role">High-performance reactive frontend UI, session state management, and component architecture.</div>
            </div>
            <div class="about-tech-item">
                <div class="about-tech-item-name">🐼 Pandas & NumPy</div>
                <div class="about-tech-item-role">Vectorized time-series analysis, data manipulation, rolling statistical metrics, and CSV storage.</div>
            </div>
            <div class="about-tech-item">
                <div class="about-tech-item-name">📊 Plotly Express</div>
                <div class="about-tech-item-role">Interactive, responsive dual-theme visualizations (radar score breakdown, heatmaps, and trend curves).</div>
            </div>
            <div class="about-tech-item">
                <div class="about-tech-item-name">🤖 Scikit-Learn</div>
                <div class="about-tech-item-role">Multi-output machine learning regression models for predictive next-day wellness forecasting.</div>
            </div>
            <div class="about-tech-item">
                <div class="about-tech-item-name">🤗 Hugging Face & PyTorch</div>
                <div class="about-tech-item-role">Fine-tuned FLAN-T5 sequence-to-sequence model powering grounded natural-language wellness explanations.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------
    # 08 — PROJECT VISION & HEALTH DISCLAIMER
    # -------------------------------------------------------------
    st.markdown(
        """
        <div class="about-section-header">
            <span class="about-sec-num">07</span>
            <h2 class="about-sec-title">Project Vision & Boundaries</h2>
        </div>
        <div class="about-overview-card" style="margin-bottom: 1.2rem;">
            <p>
                <strong>Our Vision:</strong> To cultivate self-awareness and sustainable behavioral habits
                through transparent, privacy-first personal data intelligence. LifeLens empowers individuals
                to see the tangible impact of small daily choices, helping them build balanced routines that
                foster long-term physical, mental, and productive wellbeing.
            </p>
        </div>
        <div class="about-disclaimer-box">
            <h4>⚠️ Important Health & Medical Disclaimer</h4>
            <p>
                LifeLens is an educational, habit-tracking, and lifestyle analytics tool designed solely for
                personal self-awareness and general wellbeing encouragement. It is <strong>NOT</strong> a medical
                device, diagnostic system, clinical tool, or mental health intervention service.
            </p>
            <p>
                The scores, trends, predictions, and AI-generated insights provided by LifeLens do not constitute
                medical advice, psychiatric diagnosis, or professional clinical treatment. Users experiencing
                chronic sleep disruption, clinical anxiety, depressive disorders, extreme fatigue, or other physical
                or psychological distress should always consult a licensed physician, clinical psychologist, or
                certified healthcare professional.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

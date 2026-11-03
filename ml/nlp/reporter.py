import os

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(BASE_DIR, "ml", "nlp", "lifelens_nlp_model")

_tokenizer = None
_model = None
_device = None


def _load_model():
    global _tokenizer, _model, _device

    if not _DEPS_AVAILABLE:
        return False
    if _model is not None:
        return True
    if not os.path.isdir(MODEL_DIR):
        return False

    try:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
        _model = _model.to(_device)
        _model.eval()
        return True
    except Exception as error:
        print("NLP model load error:", repr(error))
        _tokenizer = None
        _model = None
        _device = None
        return False


def _build_history_text(history):
    if history is None or history.empty:
        return "No recent history available."

    lines = []
    for _, row in history.tail(7).iterrows():
        lines.append(
            f"Date: {row['Date']}, Sleep: {float(row['SleepHours']):.1f}h, "
            f"Mood: {float(row['Mood']):.1f}/10, Energy: {float(row['Energy']):.1f}/10, "
            f"Stress: {float(row['Stress']):.1f}/10, Exercise: {float(row['ExerciseMinutes']):.0f}min, "
            f"Screen Time: {float(row['ScreenTime']):.1f}h, "
            f"Productivity: {float(row['Productivity']):.1f}/10"
        )
    return "\n".join(lines)


def _build_prediction_text(prediction):
    if not prediction:
        return "No tomorrow prediction available."
    return (
        f"Sleep: {prediction.get('SleepHours', 'N/A')}h, "
        f"Mood: {prediction.get('Mood', 'N/A')}/10, "
        f"Energy: {prediction.get('Energy', 'N/A')}/10, "
        f"Stress: {prediction.get('Stress', 'N/A')}/10, "
        f"Exercise: {prediction.get('ExerciseMinutes', 'N/A')}min, "
        f"Screen Time: {prediction.get('ScreenTime', 'N/A')}h, "
        f"Productivity: {prediction.get('Productivity', 'N/A')}/10"
    )


def _resolve_tomorrow_prediction(history, tomorrow_prediction):
    """Use the existing seven-model predictor when forecast context was not supplied."""
    if tomorrow_prediction:
        return tomorrow_prediction
    if history is None or history.empty:
        return None

    try:
        from ml.predictor import predict_next_day_ml
        prediction = predict_next_day_ml(history)
        return prediction if prediction else None
    except Exception as error:
        print("NLP forecast context unavailable:", repr(error))
        return None


def _clean_report(report):
    report = str(report).strip()
    for heading in ("OVERALL", "KEY FACTOR", "PATTERN", "PRIORITY"):
        report = report.replace(f"**{heading}:**", f"{heading}:")
    return report


def generate_wellness_report(
    wellness_score,
    wellness_category,
    latest,
    history=None,
    daily_note=None,
    tomorrow_prediction=None,
):
    """
    Generate a dynamic LifeLens wellness analysis.

    Wellness Score/category remain authoritative. The NLP model only interprets
    the supplied data and existing ML forecast; it does not replace either.
    """
    if wellness_score is None or latest is None or not _load_model():
        return None

    try:
        tomorrow_prediction = _resolve_tomorrow_prediction(
            history, tomorrow_prediction
        )

        note_text = str(daily_note).strip() if daily_note is not None else ""
        if not note_text:
            note_text = "No daily note provided."

        prompt = f"""Analyze this LifeLens user's wellness data.

Current Wellness Score: {wellness_score}/100
Current Category: {wellness_category}

CURRENT CHECK-IN
Sleep: {float(latest['SleepHours']):.1f} hours
Mood: {float(latest['Mood']):.1f}/10
Energy: {float(latest['Energy']):.1f}/10
Stress: {float(latest['Stress']):.1f}/10
Exercise: {float(latest['ExerciseMinutes']):.0f} minutes
Screen Time: {float(latest['ScreenTime']):.1f} hours
Productivity: {float(latest['Productivity']):.1f}/10

RECENT HISTORY (up to 7 latest check-ins)
{_build_history_text(history)}

DAILY NOTE
{note_text}

TOMORROW'S EXISTING ML PREDICTION
{_build_prediction_text(tomorrow_prediction)}

Generate exactly 4 wellness insights using these exact headings:

OVERALL:
Briefly explain the user's current wellness state using the Wellness Score and Category.

KEY FACTOR:
Identify the single most important strength or weakness affecting today's wellness.

PATTERN:
Identify one meaningful relationship or trend from the supplied data. Only mention a pattern supported by the data.

PRIORITY:
Give one practical and specific action the user should focus on next.

Rules:
- Use only the provided LifeLens data.
- Treat the Wellness Score and Category as authoritative.
- Treat tomorrow's values as predictions, not observations.
- Do not invent measurements, events, causes, diagnoses, or trends.
- Do not calculate or replace the Wellness Score.
- Do not replace the existing ML prediction logic.
- Do not repeat the same point in multiple sections.
- Keep each section concise.
- Do not add introductions, conclusions, numbering, or extra headings.
"""

        inputs = _tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )
        inputs = {key: value.to(_device) for key, value in inputs.items()}

        with torch.no_grad():
            output = _model.generate(
                **inputs,
                max_new_tokens=160,
                num_beams=4,
                early_stopping=True,
            )

        report = _tokenizer.decode(
            output[0], skip_special_tokens=True
        ).strip()
        return _clean_report(report) if report else None

    except Exception as error:
        print("NLP report error:", repr(error))
        return None



def explain_tomorrow_forecast(
    prediction,
    latest=None,
    history=None,
    daily_note=None
):
    """
    Generate an AI explanation of Tomorrow's Forecast.

    Uses the V5 hybrid pipeline (ml/nlp/inference.py):
    - Deterministic forecast facts are built from src/nlp_facts.py
    - The V5 FLAN-T5 model explains those facts via generate_nlp_response
    - Falls back to deterministic text if model is unavailable or fails

    Returns a dictionary with 4 keys:
        "Tomorrow Outlook", "Main Factor", "Pattern", "Priority"
    or None if prediction is empty.
    """
    if not prediction:
        return None

    import re as _re

    # Fallback defaults (returned if model fails or no sections found)
    _defaults = {
        "Tomorrow Outlook": "Tomorrow's forecast projects balanced indicators based on your recent habit patterns.",
        "Main Factor": "Core factors remain steady.",
        "Pattern": "Trends reflect current stability across your recorded wellness routines.",
        "Priority": "Maintain regular sleep timing and keep your preparation balanced tomorrow.",
    }

    try:
        from src.nlp_facts import build_forecast_facts
        from ml.nlp.inference import generate_nlp_response

        # Build deterministic forecast facts (authoritative source of truth)
        forecast_facts = build_forecast_facts(
            current=latest if latest is not None else {},
            predicted=prediction,
        )

        if not forecast_facts:
            return dict(_defaults)

        # V5 hybrid pipeline: facts -> NLP explanation string
        raw = generate_nlp_response("forecast_explanation", forecast_facts)

        if not raw or not raw.strip():
            return dict(_defaults)

        # Parse V5 output string into 4 named cards.
        # V5 produces:  FORECAST ... EXPECTED CHANGES ... WATCH FOR ... TOMORROW ...
        section_regexes = [
            ("Tomorrow Outlook", r"(?:^|\s+)(?:FORECAST|TOMORROW(?:'S)?\s+OUTLOOK):?"),
            ("Main Factor",      r"(?:\s+)(?:EXPECTED\s+CHANGES|MAIN\s+FACTOR|KEY\s+FACTOR|BIGGEST\s+(?:IMPROVEMENT|CONCERN)):?"),
            ("Pattern",          r"(?:\s+)(?:WATCH\s+FOR|PATTERN(?:\s+DETECTED)?):?"),
            ("Priority",         r"(?:\s+)(?:TOMORROW(?:'S)?(?!\s+OUTLOOK)(?:\s+(?:FOCUS|PRIORITY|ACTION))?|PRIORITY|NEXT\s+WEEK(?:'S)?\s+PRIORITY):?"),
        ]

        positions = []
        for label, regex in section_regexes:
            m = _re.search(regex, raw, _re.IGNORECASE)
            if m:
                positions.append((m.start(), m.end(), label))

        positions.sort(key=lambda x: x[0])
        explanation = dict(_defaults)

        if positions:
            for i, (_, end_pos, label) in enumerate(positions):
                next_start = positions[i + 1][0] if i + 1 < len(positions) else len(raw)
                content = raw[end_pos:next_start].strip()
                if content:
                    explanation[label] = content
        else:
            # Last-resort: try numbered-item split
            parts = _re.split(r'(?:^|\s+)\d+\.\s+(?=[A-Z])', raw)
            parts = [p.strip() for p in parts if p.strip()]
            for i, label in enumerate(["Tomorrow Outlook", "Main Factor", "Pattern", "Priority"]):
                if i < len(parts) and parts[i]:
                    explanation[label] = parts[i]

        # Contextual refinement if Pattern was not explicitly generated
        if explanation["Pattern"] == _defaults["Pattern"] and forecast_facts:
            improving = [m for m, v in forecast_facts.items() if v.get("direction") == "improving"]
            declining = [m for m, v in forecast_facts.items() if v.get("direction") in ("declining", "worsening")]
            if improving and not declining:
                explanation["Pattern"] = f"Positive momentum projected across {', '.join(improving[:2])}."
            elif declining and not improving:
                explanation["Pattern"] = f"Watch for dips in {', '.join(declining[:2])} during tomorrow's routine."
            elif improving and declining:
                explanation["Pattern"] = f"Mixed trajectory: gains in {improving[0]} balanced by pressure in {declining[0]}."

        return explanation

    except Exception as error:
        print("Forecast explanation error:", repr(error))
        return dict(_defaults)


def explain_weekly_facts(weekly_facts, daily_notes=None):
    """
    Explain Python-calculated weekly facts using the existing local NLP model.

    Python calculates the statistics.
    The NLP model only explains them.
    """
    if not weekly_facts or not _load_model():
        return None

    try:
        metrics = weekly_facts.get("metrics", {})
        days = weekly_facts.get("days_available", 0)

        facts_text = []
        for metric, values in metrics.items():
            facts_text.append(
                f"{metric}: "
                f"current_average={values.get('current_average', 'N/A')}, "
                f"previous_average={values.get('previous_average', 'N/A')}, "
                f"change={values.get('change', 'N/A')}, "
                f"trend={values.get('trend', 'N/A')}"
            )

        facts_text = "\n".join(facts_text)

        notes_text = "No daily notes available."
        if daily_notes:
            if isinstance(daily_notes, (list, tuple)):
                valid_notes = [
                    str(note).strip()
                    for note in daily_notes
                    if str(note).strip()
                ]
                if valid_notes:
                    notes_text = "\n".join(
                        f"- {note}" for note in valid_notes[-7:]
                    )
            else:
                notes_text = str(daily_notes).strip()

        prompt = f"""Analyze these factual LifeLens weekly wellness statistics.

NUMBER OF RECORDED DAYS: {days}

WEEKLY FACTS:
{facts_text}

DAILY NOTES:
{notes_text}

Generate exactly 4 sections using these exact headings:

WEEKLY OVERVIEW:
Summarize the overall week using only the supplied statistics.

KEY TREND:
Identify the most meaningful improving, declining, or stable trend supported by the data.

MAIN FACTOR:
Identify the most important measurable strength or weakness from the supplied facts.

NEXT WEEK PRIORITY:
Give one practical action based on the strongest evidence in the data.

Rules:
- Python-calculated statistics are authoritative.
- Do not invent statistics.
- Do not invent trends.
- Do not claim causation unless the supplied data explicitly supports it.
- Do not calculate a different Wellness Score.
- Do not make medical diagnoses.
- If data is limited, explicitly acknowledge that.
- Keep every section concise.
- Output only the four requested sections.
"""

        inputs = _tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        inputs = {
            key: value.to(_device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            output = _model.generate(
                **inputs,
                max_new_tokens=160,
                num_beams=4,
                early_stopping=True,
            )

        report = _tokenizer.decode(
            output[0],
            skip_special_tokens=True
        ).strip()

        return report if report else None

    except Exception as error:
        print("Weekly NLP analysis error:", repr(error))
        return None

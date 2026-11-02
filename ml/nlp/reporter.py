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

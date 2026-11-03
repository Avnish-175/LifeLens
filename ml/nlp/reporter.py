import os
import re

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# V5 fine-tuned FLAN-T5 model is the single authoritative LLM
MODEL_DIR = os.path.join(BASE_DIR, "ml", "nlp", "lifelens_nlp_model_v5_gpu")

_tokenizer = None
_model = None
_device = None


def _load_model():
    """Route model loading to the authoritative V5 model bundle."""
    from ml.nlp.inference import load_v5_model
    return load_v5_model()


# ---------------------------------------------------------------------------
# Metric-aware direction logic (done in Python, never by the language model)
# ---------------------------------------------------------------------------

# polarity: "higher_better" | "lower_better" | "neutral_range"
_METRICS = {
    "Mood":            {"label": "Mood",        "polarity": "higher_better", "tol": 0.05, "scale": 10.0},
    "Energy":          {"label": "Energy",      "polarity": "higher_better", "tol": 0.05, "scale": 10.0},
    "Productivity":    {"label": "Productivity", "polarity": "higher_better", "tol": 0.05, "scale": 10.0},
    "ExerciseMinutes": {"label": "Exercise",    "polarity": "higher_better", "tol": 0.5,  "scale": 60.0},
    "Stress":          {"label": "Stress",      "polarity": "lower_better",  "tol": 0.05, "scale": 10.0},
    "ScreenTime":      {"label": "Screen time", "polarity": "lower_better",  "tol": 0.05, "scale": 8.0},
    "SleepHours":      {"label": "Sleep",       "polarity": "neutral_range", "tol": 0.05, "scale": 8.0},
}

_METRIC_ORDER = [
    "Mood", "Energy", "Stress", "Productivity",
    "SleepHours", "ExerciseMinutes", "ScreenTime",
]

_SLEEP_HEALTHY_LOW = 7.0
_SLEEP_HEALTHY_HIGH = 9.0

_FORECAST_KEYS = ("Tomorrow Outlook", "Main Factor", "Pattern", "Priority")

_FALLBACK_FORECAST = {
    "Tomorrow Outlook": "No forecast explanation available.",
    "Main Factor": "No forecast explanation available.",
    "Pattern": "No forecast explanation available.",
    "Priority": "No forecast explanation available.",
}


def _lookup(source, key):
    """Case-insensitive value lookup that works for dicts, Series and None."""
    if source is None:
        return None
    try:
        if hasattr(source, "get") and key in source:
            return source.get(key)
    except Exception:
        pass
    try:
        items = source.items()
    except Exception:
        return None
    wanted = key.lower().replace("_", "")
    for k, v in items:
        if str(k).lower().replace("_", "") == wanted:
            return v
    return None


def _to_float(value):
    try:
        if value is None:
            return None
        number = float(value)
        if number != number:  # NaN
            return None
        if number in (float("inf"), float("-inf")):
            return None
        return number
    except (TypeError, ValueError):
        return None


def _fmt(metric, value):
    if metric == "ExerciseMinutes":
        return f"{value:.0f}"
    return f"{value:.1f}"


def _sleep_distance(hours):
    if hours < _SLEEP_HEALTHY_LOW:
        return _SLEEP_HEALTHY_LOW - hours
    if hours > _SLEEP_HEALTHY_HIGH:
        return hours - _SLEEP_HEALTHY_HIGH
    return 0.0


def _classify_change(metric, current, predicted):
    """
    Return (change_type, status).

    change_type: "increase" | "decrease" | "same"   (pure numeric direction)
    status:      "improving" | "declining" | "worsening" | "stable"
                 | "increasing" | "decreasing"       (neutral, sleep only)
    """
    spec = _METRICS[metric]
    delta = predicted - current

    if abs(delta) < spec["tol"]:
        return "same", "stable"

    change_type = "increase" if delta > 0 else "decrease"
    polarity = spec["polarity"]

    if polarity == "higher_better":
        status = "improving" if delta > 0 else "declining"
    elif polarity == "lower_better":
        status = "worsening" if delta > 0 else "improving"
    else:  # sleep: neutral unless it clearly moves toward / away from healthy range
        before = _sleep_distance(current)
        after = _sleep_distance(predicted)
        if after < before:
            status = "improving"
        elif after > before:
            status = "worsening"
        else:
            status = "increasing" if delta > 0 else "decreasing"

    return change_type, status


def _describe_status(metric, change_type, status):
    """Short human phrase for a metric's status, metric-aware."""
    if status == "stable":
        return "stable"
    if metric == "Stress":
        return "increasing stress, worsening" if change_type == "increase" else "decreasing stress, improving"
    if metric == "ScreenTime":
        return ("increasing screen time, potentially worsening"
                if change_type == "increase"
                else "decreasing screen time, potentially improving")
    if metric == "ExerciseMinutes":
        return "increasing activity, improving" if change_type == "increase" else "decreasing activity, declining"
    if metric == "SleepHours":
        verb = "increasing" if change_type == "increase" else "decreasing"
        if status == "improving":
            return f"{verb}, moving toward a healthy range"
        if status == "worsening":
            return f"{verb}, moving away from a healthy range"
        return verb
    return status


def _build_forecast_facts(prediction, latest):
    """
    Build authoritative, Python-calculated forecast facts.
    Returns an ordered list of dicts (one per predicted metric).
    """
    facts = []
    for metric in _METRIC_ORDER:
        current = _to_float(_lookup(latest, metric))
        predicted = _to_float(_lookup(prediction, metric))
        if predicted is None:
            continue
        if current is None:
            facts.append({
                "metric": metric,
                "name": _METRICS[metric]["label"],
                "current": None,
                "predicted": predicted,
                "delta": None,
                "change_type": None,
                "status": None,
                "phrase": None,
                "magnitude": 0.0,
            })
            continue

        change_type, status = _classify_change(metric, current, predicted)
        delta = predicted - current
        facts.append({
            "metric": metric,
            "name": _METRICS[metric]["label"],
            "current": current,
            "predicted": predicted,
            "delta": delta,
            "change_type": change_type,
            "status": status,
            "phrase": _describe_status(metric, change_type, status),
            "magnitude": abs(delta) / _METRICS[metric]["scale"],
        })
    return facts


def _forecast_facts_text(facts):
    lines = []
    for f in facts:
        if f["current"] is None:
            lines.append(f"{f['name']}: predicted {_fmt(f['metric'], f['predicted'])} (no current value)")
        else:
            lines.append(
                f"{f['name']}: {_fmt(f['metric'], f['current'])} -> "
                f"{_fmt(f['metric'], f['predicted'])} ({f['phrase']})"
            )
    return "\n".join(lines)


def _facts_for_v5(facts):
    """Shape Python-calculated facts like the V5 inference prompt expects."""
    out = {}
    for f in facts:
        if f["current"] is None:
            continue
        direction = "unchanged" if f["status"] == "stable" else f["status"]
        out[f["metric"]] = {
            "current": round(f["current"], 1),
            "predicted": round(f["predicted"], 1),
            "change": round(f["delta"], 1),
            "direction": direction,
        }
    return out


def _join_names(names):
    names = list(names)
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def _deterministic_forecast(facts):
    """Safe, always-correct structured forecast built purely from Python facts."""
    comparable = [f for f in facts if f["current"] is not None]
    if not comparable:
        return dict(_FALLBACK_FORECAST)

    improving = [f for f in comparable if f["status"] == "improving"]
    worsening = [f for f in comparable if f["status"] in ("worsening", "declining")]
    neutral = [f for f in comparable if f["status"] in ("increasing", "decreasing")]

    imp_names = [f["name"].lower() for f in improving]
    wor_names = [f["name"].lower() for f in worsening]

    # Tomorrow Outlook
    if improving and worsening:
        if len(improving) > len(worsening):
            tone = "generally positive"
        elif len(worsening) > len(improving):
            tone = "generally cautious"
        else:
            tone = "balanced"
        outlook = (
            f"The forecast is mixed but {tone}, with improvement expected in "
            f"{_join_names(imp_names)}, while {_join_names(wor_names)} may move in a less favourable direction."
        )
    elif improving:
        outlook = f"The forecast is positive, with improvement expected in {_join_names(imp_names)}."
    elif worsening:
        outlook = (
            f"The forecast suggests some pressure tomorrow, with {_join_names(wor_names)} "
            f"moving in a less favourable direction."
        )
    elif neutral:
        outlook = "The forecast is largely steady, with only neutral changes expected."
    else:
        outlook = "The forecast shows stable values across tracked metrics tomorrow."

    # Main Factor: largest normalised change
    moving = [f for f in comparable if f["status"] != "stable"]
    if moving:
        top = max(moving, key=lambda f: f["magnitude"])
        main = (
            f"{top['name']} shows the largest predicted change, moving from "
            f"{_fmt(top['metric'], top['current'])} to {_fmt(top['metric'], top['predicted'])} "
            f"({top['phrase']})."
        )
        others = [f for f in moving if f is not top and f["status"] in ("worsening", "declining")]
        if top["status"] == "improving" and others:
            o = max(others, key=lambda f: f["magnitude"])
            main += (
                f" {o['name']} also changes from {_fmt(o['metric'], o['current'])} "
                f"to {_fmt(o['metric'], o['predicted'])} ({o['phrase']})."
            )
    else:
        main = "No metric is predicted to change meaningfully tomorrow."

    # Pattern
    if improving and worsening:
        if len(improving) >= len(worsening):
            pattern = (
                f"Most movement is positive in {_join_names(imp_names[:3])}, "
                f"but {_join_names(wor_names[:2])} may offset some of those gains."
            )
        else:
            pattern = (
                f"Several metrics ({_join_names(wor_names[:3])}) are trending in a less favourable "
                f"direction, with gains limited to {_join_names(imp_names[:2])}."
            )
    elif improving:
        pattern = f"Predicted changes are consistently positive, led by {_join_names(imp_names[:2])}."
    elif worsening:
        pattern = f"Predicted changes lean toward {_join_names(wor_names[:3])} becoming less favourable."
    else:
        pattern = "Predicted values stay close to today's levels, suggesting a steady day."

    # Priority
    if worsening:
        target = max(worsening, key=lambda f: f["magnitude"])
        if improving:
            keep = max(improving, key=lambda f: f["magnitude"])
            priority = (
                f"Keep supporting {keep['name'].lower()} while taking steps to manage "
                f"{target['name'].lower()} tomorrow."
            )
        else:
            priority = f"Focus on managing {target['name'].lower()} tomorrow."
    elif improving:
        keep = max(improving, key=lambda f: f["magnitude"])
        priority = f"Maintain the habits supporting {keep['name'].lower()} to hold on to the predicted gains."
    else:
        priority = "Keep your routine consistent tomorrow to preserve stable levels."

    return {
        "Tomorrow Outlook": outlook,
        "Main Factor": main,
        "Pattern": pattern,
        "Priority": priority,
    }


# ---------------------------------------------------------------------------
# Section parsing (separates sections only; never judges numbers)
# ---------------------------------------------------------------------------

_HEADING_ALIASES = {
    "Tomorrow Outlook": [
        r"TOMORROW(?:'S|S)?\s+OUTLOOK", r"FORECAST", r"OUTLOOK", r"OVERALL",
    ],
    "Main Factor": [
        r"MAIN\s+FACTOR", r"KEY\s+FACTOR", r"EXPECTED\s+CHANGES",
    ],
    "Pattern": [
        r"PATTERN(?:\s+DETECTED)?", r"WATCH\s+FOR",
    ],
    "Priority": [
        r"PRIORITY", r"TOMORROW(?:'S|S)?\s+FOCUS", r"FOCUS",
    ],
}

_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F]+")


def _parse_forecast_sections(raw):
    """
    Split raw model text into the four forecast fields.
    Returns a dict containing only the sections that were found with content.
    """
    if not raw:
        return {}

    text = str(raw).replace("**", "").replace("__", "")
    text = _EMOJI_RE.sub(" ", text)
    text = re.sub(r"(?m)^[ \t]*#+[ \t]*", "", text)
    # V5 uses a bare "TOMORROW" heading on its own line for the action section.
    text = re.sub(r"(?im)^[ \t]*TOMORROW[ \t]*:?[ \t]*$", "TOMORROW FOCUS:", text)

    alt_to_label = []
    for label, aliases in _HEADING_ALIASES.items():
        for alias in aliases:
            alt_to_label.append((alias, label))
    alt_to_label.sort(key=lambda x: -len(x[0]))  # more specific aliases first
    combined = "|".join(f"(?P<g{i}>{alias})" for i, (alias, _) in enumerate(alt_to_label))

    heading_re = re.compile(
        rf"(?<![A-Za-z'])(?:{combined})(?![A-Za-z'])[ \t]*(?::|(?=\n|$))",
        re.IGNORECASE,
    )

    hits = []
    for m in heading_re.finditer(text):
        label = None
        for i, (_, lab) in enumerate(alt_to_label):
            if m.group(f"g{i}") is not None:
                label = lab
                break
        if label:
            hits.append((m.start(), m.end(), label))

    sections = {}
    for idx, (_, end, label) in enumerate(hits):
        nxt = hits[idx + 1][0] if idx + 1 < len(hits) else len(text)
        content = text[end:nxt].strip(" \t\n:-")
        content = re.sub(r"\s+", " ", content).strip()
        if content and label not in sections:
            sections[label] = content
    return sections


# ---------------------------------------------------------------------------
# Validation of model wording against Python facts
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_WEEKLY_LEAK_MARKERS = (
    "weekly summary", "biggest improvement", "biggest concern",
    "pattern detected", "this week", "weekly overview", "next week",
)
_BAD_WORDS_FOR_IMPROVING = ("declin", "worsen", "deteriorat")
_BAD_WORDS_FOR_HIGHER_BETTER_IMPROVING = ("decreas", "drop", "fall", "fell")
_BAD_WORDS_FOR_WORSENING = ("improv",)


def _allowed_numbers(facts):
    allowed = {"10", "100", "1"}
    for f in facts:
        for key in ("current", "predicted"):
            v = f[key]
            if v is None:
                continue
            allowed.add(_fmt(f["metric"], v))
            allowed.add(f"{v:.0f}")
            allowed.add(f"{v:.1f}")
            allowed.add(str(float(f"{v:.1f}")))
    return allowed


def _direction_consistent(text, facts):
    """
    Reject wording that contradicts Python's direction for a metric.
    Only sentences that mention exactly one metric are checked, so mixed
    sentences such as "energy improves while stress worsens" are left alone.
    """
    comparable = [f for f in facts if f["current"] is not None]
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        low = sentence.lower()
        mentioned = [f for f in comparable if f["name"].lower() in low]
        if len(mentioned) != 1:
            continue
        f = mentioned[0]
        status = f["status"]
        if status == "improving":
            if any(w in low for w in _BAD_WORDS_FOR_IMPROVING):
                return False
            if (_METRICS[f["metric"]]["polarity"] == "higher_better"
                    and any(w in low for w in _BAD_WORDS_FOR_HIGHER_BETTER_IMPROVING)):
                return False
        elif status in ("worsening", "declining"):
            if any(w in low for w in _BAD_WORDS_FOR_WORSENING):
                return False
        elif status == "stable":
            if any(w in low for w in ("improv", "declin", "worsen")):
                return False
    return True


def _section_is_safe(text, allowed, facts):
    """Reject sections that leak weekly content, invent numbers, or contradict directions."""
    if "<" in text or ">" in text:
        return False
    lower = text.lower()
    if any(marker in lower for marker in _WEEKLY_LEAK_MARKERS):
        return False
    for number in _NUMBER_RE.findall(text):
        if number not in allowed:
            return False
    return _direction_consistent(text, facts)


# ---------------------------------------------------------------------------
# Text generation helpers
# ---------------------------------------------------------------------------

def _generate_text(prompt, max_new_tokens=160):
    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {key: value.to(_device) for key, value in inputs.items()}
    with torch.no_grad():
        output = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
        )
    return _tokenizer.decode(output[0], skip_special_tokens=True).strip()


def _generate_v5_forecast_text(facts):
    """
    Wording from the V5 model (ml/nlp/inference.py), fed Python-corrected facts.
    Returns None if V5 is unavailable. Its deterministic fallback text is never
    used here, because it cannot be told apart from model output.
    """
    try:
        from ml.nlp import inference
        v5_facts = _facts_for_v5(facts)
        if not v5_facts or not inference.load_v5_model():
            return None
        prompt = inference.build_prompt_from_facts("forecast_explanation", v5_facts)
        inputs = inference._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=256
        )
        inputs = {k: v.to(inference._device) for k, v in inputs.items()}
        with torch.no_grad():
            output = inference._model.generate(
                **inputs, max_new_tokens=180, num_beams=4, early_stopping=True
            )
        text = inference._tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return text or None
    except Exception as error:
        print("V5 forecast generation unavailable:", repr(error))
        return None


def _generate_legacy_forecast_text(facts):
    """Wording from the model loaded by this module."""
    if not _load_model():
        return None
    facts_text = _forecast_facts_text(facts)
    prompt = f"""Explain tomorrow's predicted wellness changes for a LifeLens user.

TOMORROW'S FORECAST (current -> predicted, with direction already calculated):
{facts_text}

Generate exactly 4 sections using these exact headings:

TOMORROW OUTLOOK:
Summarize the overall forecast for tomorrow.

MAIN FACTOR:
Name the metric with the largest predicted change and describe it.

PATTERN:
Describe how the predicted changes relate to each other.

PRIORITY:
Give one practical action for tomorrow.

Rules:
- The direction labels in parentheses are authoritative; never contradict them.
- Stress and screen time are better when lower.
- Use only the numbers shown above.
- Do not invent causes, events, or diagnoses.
- Do not include weekly summaries or weekly trends.
- Keep each section to one or two sentences.
"""
    return _generate_text(prompt, max_new_tokens=200)


# ---------------------------------------------------------------------------
# Existing helpers
# ---------------------------------------------------------------------------

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


def _build_prediction_text(prediction, latest=None):
    """Tomorrow's prediction; when current values exist, Python-calculated directions are included."""
    if not prediction:
        return "No tomorrow prediction available."

    if latest is not None:
        try:
            facts = _build_forecast_facts(prediction, latest)
            if facts and any(f["current"] is not None for f in facts):
                return _forecast_facts_text(facts)
        except Exception as error:
            print("Prediction direction build error:", repr(error))

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
        from datetime import datetime
        today_str = datetime.now().strftime("%Y-%m-%d")
        last_date = str(history.iloc[-1].get("Date", "")).split()[0]
        if last_date != today_str:
            return None
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
    Generate a dynamic LifeLens wellness analysis using the authoritative V5 LLM.

    Wellness Score/category remain authoritative. The V5 FLAN-T5 model only interprets
    the supplied facts and existing ML forecast; it does not replace either.
    Returns:
        {"status": "ok", "text": str} or failure dict with status and honest message.
        None if wellness_score or latest check-in is not recorded.
    """
    if wellness_score is None or latest is None:
        return None

    try:
        from src.nlp_facts import build_daily_facts
        from ml.nlp.inference import generate_v5_daily_report

        facts = build_daily_facts(
            latest=latest,
            wellness_score=wellness_score,
            wellness_category=wellness_category,
            history=history,
        )

        history_text = _build_history_text(history)
        note_text = str(daily_note).strip() if daily_note is not None else ""

        return generate_v5_daily_report(
            facts=facts,
            history_text=history_text,
            note_text=note_text
        )

    except Exception as error:
        import sys
        print("[LifeLens NLP Error] generate_wellness_report error:", repr(error), file=sys.stderr)
        from ml.nlp.inference import MSG_DAILY_GEN_FAILED
        return {"status": "generation_failed", "message": MSG_DAILY_GEN_FAILED}


def explain_tomorrow_forecast(
    prediction,
    latest=None,
    history=None,
    daily_note=None
):
    """
    Explain Tomorrow's Forecast using the authoritative V5 fine-tuned LLM.

    The numerical forecast comes strictly from ML models and Wellness Score logic.
    The V5 LLM interprets the forecast facts and generates the natural-language explanation.
    Returns:
        dict with status ('ok', 'not_found', 'load_failed', 'generation_failed')
        and the 4 forecast fields when successful:
        "Tomorrow Outlook", "Main Factor", "Pattern", "Priority"
        or None if prediction is empty.
    """
    if not prediction:
        return None

    try:
        # Current values: explicit `latest`, otherwise most recent history row ONLY if it matches today.
        current = latest
        if current is None and history is not None:
            try:
                if not history.empty:
                    from datetime import datetime
                    today_str = datetime.now().strftime("%Y-%m-%d")
                    last_row = history.iloc[-1]
                    last_date = str(last_row.get("Date", "")).split()[0]
                    if last_date == today_str:
                        current = last_row
            except Exception:
                current = None

        from src.nlp_facts import build_forecast_facts
        from ml.nlp.inference import generate_v5_forecast_explanation

        facts = build_forecast_facts(current, prediction)
        history_text = _build_history_text(history)

        return generate_v5_forecast_explanation(
            facts=facts,
            latest=current,
            prediction=prediction,
            history_text=history_text
        )

    except Exception as error:
        import sys
        print("[LifeLens NLP Error] explain_tomorrow_forecast error:", repr(error), file=sys.stderr)
        from ml.nlp.inference import MSG_FORECAST_GEN_FAILED
        return {"status": "generation_failed", "message": MSG_FORECAST_GEN_FAILED}


def explain_weekly_facts(weekly_facts, daily_notes=None):
    """
    Explain Python-calculated weekly facts using the V5 hybrid NLP pipeline.

    Python calculates authoritative statistics (src/nlp_facts.py).
    The V5 model explains them with strict numerical validation and fallback.

    Returns a dictionary with 4 structured sections:
        "Weekly Overview", "Key Trend", "Main Factor", "Next Week Priority"
    or None if weekly_facts is empty.
    """
    if not weekly_facts:
        return None

    try:
        import re
        from src.nlp_facts import METRIC_DISPLAY_NAMES
        from ml.nlp.inference import generate_nlp_response

        # Build authoritative deterministic baseline matching the 4 UI sections
        start = weekly_facts.get("wellness_start", "--")
        end = weekly_facts.get("wellness_end", "--")
        change = weekly_facts.get("wellness_change", 0)
        direction = weekly_facts.get("wellness_direction", "unchanged")
        sign = "+" if isinstance(change, (int, float)) and change > 0 else ""

        metrics = weekly_facts.get("metrics", {})
        metric_strs = []
        for m in ["Mood", "Energy", "Stress", "Productivity"]:
            if m in metrics:
                d = metrics[m]
                display_name = METRIC_DISPLAY_NAMES.get(m, m)
                metric_strs.append(f"{display_name} moved from {d['start']} to {d['end']} ({d['direction']})")
        trend_default = ", ".join(metric_strs) if metric_strs else "Core metrics remained within normal ranges."

        routine_strs = []
        for r in ["SleepHours", "ScreenTime", "ExerciseMinutes"]:
            if r in metrics:
                rd = metrics[r]
                rname = METRIC_DISPLAY_NAMES.get(r, r)
                routine_strs.append(f"{rname} moved from {rd['start']} to {rd['end']}")
        routine_default = ", ".join(routine_strs) if routine_strs else "Routine factors remained steady."

        if direction in ("improved", "improving"):
            overview_default = f"Overall wellness improved from {start} to {end} ({sign}{change} points). Recorded habits reflect positive progression."
            priority_default = "Maintain the consistent habits supporting these gains, especially in your top-performing routines."
        elif direction in ("declining", "worsening"):
            overview_default = f"Overall wellness experienced pressure, moving from {start} to {end} ({change} points). The week presented some routine challenges."
            priority_default = "Prioritize rest and stress reduction next week to stabilize baseline wellness."
        else:
            overview_default = f"Overall wellness held steady from {start} to {end} ({change} points). Balanced daily routines helped maintain equilibrium."
            priority_default = "Keep sleep and screen habits consistent to protect your steady baseline."

        result = {
            "Weekly Overview": overview_default,
            "Key Trend": trend_default + ".",
            "Main Factor": routine_default + ".",
            "Next Week Priority": priority_default,
        }

        # Query V5 NLP model for natural-language interpretation
        raw = generate_nlp_response("weekly_analysis", weekly_facts)
        if raw and isinstance(raw, str) and raw.strip():
            raw_text = raw.strip()
            headings_map = [
                ("Weekly Overview", r"(?:^|\s+)(?:WEEKLY\s+(?:SUMMARY|OVERVIEW)|OVERVIEW):?"),
                ("Key Trend", r"(?:\s+)(?:METRIC\s+COMPARISONS|KEY\s+TREND|TRENDS?):?"),
                ("Main Factor", r"(?:\s+)(?:ROUTINE\s+CHANGES|MAIN\s+FACTOR|KEY\s+FACTOR):?"),
                ("Next Week Priority", r"(?:\s+)(?:NEXT\s+WEEK(?:\s+PRIORITY)?|PRIORITY|FOCUS):?"),
            ]
            positions = []
            for title, regex in headings_map:
                m = re.search(regex, raw_text, re.IGNORECASE)
                if m:
                    positions.append((m.start(), m.end(), title))
            positions.sort(key=lambda x: x[0])

            if positions:
                for idx, (_, end_pos, title) in enumerate(positions):
                    nxt = positions[idx + 1][0] if idx + 1 < len(positions) else len(raw_text)
                    content = raw_text[end_pos:nxt].strip(" \t\n:-")
                    if content and len(content) >= 15:
                        low = content.lower()
                        if "output only the four" not in low and "task:" not in low:
                            if title == "Weekly Overview":
                                if direction in ("improved", "improving") and ("declined" in low or "fell" in low or "dropped" in low):
                                    continue
                                if direction in ("declining", "worsening") and ("improved" in low or "rose" in low or "increased" in low):
                                    continue
                                result[title] = content
                            elif title == "Key Trend":
                                contradiction = False
                                for m, v in metrics.items():
                                    m_low = m.lower()
                                    if m_low in low:
                                        m_dir = v.get("direction")
                                        if m_dir in ("declining", "worsening") and re.search(rf"\b{m_low}\b[^.\n]*\b(?:improved|improving|rose|gains?)\b", low):
                                            contradiction = True
                                            break
                                        if m_dir in ("improved", "improving") and re.search(rf"\b{m_low}\b[^.\n]*\b(?:declined|declining|worsened|worsening|dropped|fell)\b", low):
                                            contradiction = True
                                            break
                                if not contradiction:
                                    result[title] = content
                            else:
                                result[title] = content

        return result

    except Exception as error:
        print("Weekly NLP analysis error:", repr(error))
        return {
            "Weekly Overview": "Overall wellness tracked steadily across recorded days.",
            "Key Trend": "Core metrics remained within expected ranges.",
            "Main Factor": "Daily routine factors contributed to current baseline levels.",
            "Next Week Priority": "Maintain consistent sleep and active recovery habits.",
        }

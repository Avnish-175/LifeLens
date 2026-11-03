"""
LifeLens V5 NLP Inference and Hybrid Integration Layer
ml/nlp/inference.py

Architecture:
- Loads the fine-tuned FLAN-T5 model from `ml/nlp/lifelens_nlp_model_v5_gpu`.
- Uses deterministic Python facts from `src/nlp_facts.py` as authoritative truth.
- Performs strict validation on generated text against factual ground truth.
- Seamlessly falls back to deterministic explanations if validation fails, the model crashes, or CUDA/memory issues occur.
"""

import os
import re
from typing import Dict, Any, Optional

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False

from src.nlp_facts import METRIC_DISPLAY_NAMES

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(BASE_DIR, "ml", "nlp", "lifelens_nlp_model_v5_gpu")

_tokenizer = None
_model = None
_device = None


def load_v5_model() -> bool:
    """Load the V5 GPU model and tokenizer cleanly."""
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
        print("V5 NLP model load failed:", repr(error))
        _tokenizer = None
        _model = None
        _device = None
        return False


# =====================================================================
# DETERMINISTIC FALLBACK GENERATORS
# =====================================================================

def generate_fallback_daily(facts: Dict[str, Any]) -> str:
    """Deterministic fallback for daily analysis."""
    score = facts.get("wellness_score", "--")
    category = facts.get("wellness_category", "Normal")
    sleep = facts.get("sleep_hours", "--")
    energy = facts.get("energy", "--")
    mood = facts.get("mood", "--")
    stress = facts.get("stress", "--")
    change = facts.get("score_change")
    direction = facts.get("score_direction", "unchanged")

    if change is not None:
        if direction == "unchanged" or change == 0:
            change_text = f"\n\n📈 RECENT CHANGE\nYour wellness score remained unchanged at {score}."
        else:
            verb = "improved" if direction == "improved" else "decreased"
            change_text = f"\n\n📈 RECENT CHANGE\nWellness {verb} by {abs(change)} points compared to yesterday."
    else:
        change_text = ""

    concerns = []
    if facts.get("energy") is not None and facts["energy"] <= 4:
        concerns.append(f"energy is low at {energy}/10")
    if facts.get("stress") is not None and facts["stress"] >= 7:
        concerns.append(f"stress is elevated at {stress}/10")
    if facts.get("exercise_minutes") == 0:
        concerns.append("no physical exercise was recorded")
    concern_str = "; ".join(concerns) if concerns else "maintain steady recovery and pacing"

    return (
        f"🌿 OVERALL\n"
        f"Your recorded wellness score is {score}/100 ({category}). "
        f"Sleep was recorded at {sleep} hours with mood at {mood}/10.\n\n"
        f"✅ SUPPORTING SIGNALS\n"
        f"Rest and baseline habits provide a foundation for today's activities.\n\n"
        f"⚠️ MAIN CONCERN\n"
        f"Main considerations: {concern_str}.{change_text}\n\n"
        f"🎯 PRIORITY\n"
        f"Protect recovery time and pace daily activities to support sustained wellness."
    )


def generate_fallback_weekly(facts: Dict[str, Any]) -> str:
    """Deterministic 4-section fallback for weekly analysis matching UI expectations."""
    start = facts.get("wellness_start", "--")
    end = facts.get("wellness_end", "--")
    change = facts.get("wellness_change", 0)
    direction = facts.get("wellness_direction", "unchanged")
    sign = "+" if isinstance(change, (int, float)) and change > 0 else ""

    metrics = facts.get("metrics", {})
    metric_lines = []
    for m in ["Mood", "Energy", "Stress", "Productivity"]:
        if m in metrics:
            d = metrics[m]
            display_name = METRIC_DISPLAY_NAMES.get(m, m)
            metric_lines.append(
                f"{display_name} moved from {d['start']} to {d['end']} ({d['direction']})"
            )
    metrics_str = ", ".join(metric_lines) if metric_lines else "core metrics remained steady within expected ranges"

    routine_lines = []
    for r in ["SleepHours", "ScreenTime", "ExerciseMinutes"]:
        if r in metrics:
            rd = metrics[r]
            rname = METRIC_DISPLAY_NAMES.get(r, r)
            routine_lines.append(f"{rname} moved from {rd['start']} to {rd['end']}")
    routine_str = ", ".join(routine_lines) if routine_lines else "routine habits remained consistent"

    if direction in ("improved", "improving"):
        overview_tone = f"Overall wellness improved by {abs(change)} points from {start} to {end}."
        next_week_action = "Maintain the routines that supported your gains while keeping habit patterns consistent."
    elif direction in ("declining", "worsening"):
        overview_tone = f"Overall wellness experienced pressure, shifting from {start} to {end} ({change} points)."
        next_week_action = "Prioritize recovery and focus on decompressing to rebuild momentum next week."
    else:
        overview_tone = f"Overall wellness remained stable at {end} points throughout the week."
        next_week_action = "Continue consistent sleep and daily pacing to preserve baseline stability."

    return (
        f"WEEKLY SUMMARY\n"
        f"{overview_tone} Your tracked metrics show measurable progress across daily entries.\n\n"
        f"METRIC COMPARISONS\n"
        f"{metrics_str}.\n\n"
        f"ROUTINE CHANGES\n"
        f"{routine_str}.\n\n"
        f"NEXT WEEK\n"
        f"{next_week_action}"
    )


def generate_fallback_forecast(facts: Dict[str, Any]) -> str:
    """Deterministic fallback for forecast explanation."""
    changes = []
    for m in ["Mood", "Energy", "Stress", "Productivity", "SleepHours", "ScreenTime", "ExerciseMinutes"]:
        if m in facts:
            info = facts[m]
            pred = info.get("predicted")
            direction = info.get("direction", "unchanged")
            name = METRIC_DISPLAY_NAMES.get(m, m)
            
            unit = "/10" if m in ["Mood", "Energy", "Stress", "Productivity"] else (" hrs" if "Time" in m or "Hours" in m else " min")
            
            if direction == "unchanged":
                changes.append(f"{name} is expected to remain unchanged at {pred}{unit}")
            elif direction == "improving":
                changes.append(f"{name} is expected to improve to {pred}{unit}")
            elif direction == "worsening":
                changes.append(f"{name} is expected to worsen to {pred}{unit}")
            elif direction == "declining":
                changes.append(f"{name} is expected to decline to {pred}{unit}")

    expected_str = ". ".join(changes) if changes else "Tomorrow's metrics are expected to hold steady"

    return (
        f"🔮 FORECAST\n"
        f"Tomorrow's forecast projects balanced indicators based on your recent habit patterns.\n\n"
        f"📊 EXPECTED CHANGES\n"
        f"{expected_str}.\n\n"
        f"🎯 TOMORROW\n"
        f"Prepare ahead for planned tasks and keep tonight's wind-down routine consistent."
    )


def generate_fallback_daily_note(facts: Dict[str, Any]) -> str:
    """Deterministic fallback for daily note analysis."""
    note = facts.get("note", "")
    score = facts.get("wellness_score", "--")
    classifications = facts.get("classifications", {})
    stress_lbl = classifications.get("stress_label", "moderate")
    mood_lbl = classifications.get("mood_label", "moderate")

    return (
        f"💬 NOTE CONTEXT\n"
        f"Your daily note ('{note}') provides valuable subjective context to your day.\n\n"
        f"🧠 WELLNESS CONNECTION\n"
        f"With a wellness score of {score}/100, stress is classified as {stress_lbl} and mood as {mood_lbl}.\n\n"
        f"🎯 SUGGESTED FOCUS\n"
        f"Acknowledge how your mindset connects with physical metrics and take time to decompress."
    )


# =====================================================================
# OUTPUT VALIDATION
# =====================================================================

def validate_nlp_output(text: Optional[str], task: str, facts: Dict[str, Any]) -> bool:
    """
    Validation checking for instruction leakage, numerical mismatches, and directional hallucinations.
    Returns True if valid, False if fallback/retry should be used.
    """
    if not text or len(text.strip()) < 25:
        return False

    t = text.lower()

    # Reject echoed prompt instructions
    forbidden_phrases = [
        "output only the four requested sections",
        "use the provided facts exactly",
        "do not recalculate numbers",
        "do not invent numbers",
        "task: weekly_analysis",
        "current data:",
        "generate exactly 4",
    ]
    if any(phrase in t for phrase in forbidden_phrases):
        return False

    if task == "daily_analysis":
        score_dir = facts.get("score_direction")
        if score_dir == "unchanged":
            if "decreased" in t or "declining" in t or "increased" in t:
                return False

    elif task == "weekly_analysis":
        # 1. Directional consistency
        direction = facts.get("wellness_direction")
        if direction in ["declining", "worsening"]:
            if "rose sharply" in t or "improved sharply" in t or "wellness rose" in t or "wellness increased" in t:
                return False
        elif direction in ["improved", "improving"]:
            if "declined sharply" in t or "wellness fell" in t or "wellness decreased" in t or "wellness dropped" in t:
                return False
        elif direction == "unchanged":
            if "wellness rose" in t or "wellness fell" in t or "wellness increased" in t or "wellness decreased" in t:
                return False

        # 2. Check for start and end wellness score hallucination if mentioned
        start = facts.get("wellness_start")
        end = facts.get("wellness_end")
        change = facts.get("wellness_change")
        
        match_scores = re.search(r"from\s+(\d+)\s+to\s+(\d+)", t)
        if match_scores and start is not None and end is not None:
            found_start, found_end = int(match_scores.group(1)), int(match_scores.group(2))
            if found_start != int(start) or found_end != int(end):
                return False

        match_change = re.search(r"([+-]?\d+)\s*points", t)
        if match_change and change is not None:
            found_change = int(match_change.group(1))
            if abs(found_change) != abs(int(change)):
                return False

        # 3. Metric direction validation across all tracked metrics
        metrics = facts.get("metrics", {})
        for m, v in metrics.items():
            m_low = m.lower()
            if m_low in t:
                m_dir = v.get("direction")
                if m_dir in ("declining", "worsening"):
                    if re.search(rf"\b{m_low}\b[^.\n]*\b(?:improved|improving|rose|gains?)\b", t):
                        return False
                elif m_dir in ("improved", "improving"):
                    if re.search(rf"\b{m_low}\b[^.\n]*\b(?:declined|declining|worsened|worsening|dropped|fell)\b", t):
                        return False
                elif m_dir == "unchanged":
                    if re.search(rf"\b{m_low}\b[^.\n]*\b(?:rose|fell|dropped|improved|worsened)\b", t):
                        return False

    elif task == "forecast_explanation":
        for m in ["Mood", "Stress", "Energy", "Productivity"]:
            if m in facts:
                d = facts[m].get("direction")
                if d == "unchanged":
                    if f"{m.lower()} rose" in t or f"{m.lower()} fell" in t:
                        return False
                elif d == "worsening" and m == "Stress":
                    if "stress improving" in t:
                        return False

    return True


# =====================================================================
# PROMPT FORMATTERS (USING DETERMINISTIC FACTS AS CONTEXT)
# =====================================================================

def build_prompt_from_facts(task: str, facts: Dict[str, Any]) -> str:
    """Build structured prompt injecting deterministic facts."""
    instruction = (
        "Use the provided facts exactly.\n"
        "Do not recalculate numbers.\n"
        "Do not invent numbers.\n"
        "Do not change metric values.\n"
        "Do not change directions.\n"
        "Explain the provided facts naturally and concisely.\n"
    )

    if task == "daily_analysis":
        score = facts.get("wellness_score", 50)
        prev_score = facts.get("previous_wellness_score", score)
        change = facts.get("score_change", 0)
        direction = facts.get("score_direction", "unchanged")
        return (
            f"Task: daily_analysis\n\n"
            f"{instruction}\n"
            f"Current Data:\n"
            f"Wellness Score: {score}\n"
            f"Previous Wellness Score: {prev_score}\n"
            f"Change: {change}\n"
            f"Direction: {direction}\n"
            f"Category: {facts.get('wellness_category', 'Normal')}\n"
            f"SleepHours: {facts.get('sleep_hours', 7.0)}\n"
            f"ScreenTime: {facts.get('screen_time', 4.0)}\n"
            f"ExerciseMinutes: {facts.get('exercise_minutes', 0)}\n"
            f"StudyHours: {facts.get('study_hours', 0)}\n"
            f"Mood: {facts.get('mood', 5)}\n"
            f"Energy: {facts.get('energy', 5)}\n"
            f"Stress: {facts.get('stress', 5)}\n"
            f"Productivity: {facts.get('productivity', 5)}\n"
        )

    elif task == "weekly_analysis":
        metrics = facts.get("metrics", {})
        metric_lines = []
        for m, v in metrics.items():
            name = METRIC_DISPLAY_NAMES.get(m, m)
            metric_lines.append(
                f"{name}: {v['start']} -> {v['end']} (change: {v['change']}, direction: {v['direction']})"
            )
        metrics_block = "\n".join(metric_lines)
        return (
            f"Task: weekly_analysis\n\n"
            f"{instruction}\n"
            f"Wellness Score: {facts.get('wellness_start')} -> {facts.get('wellness_end')}\n"
            f"Change: {facts.get('wellness_change')}\n"
            f"Direction: {facts.get('wellness_direction')}\n\n"
            f"Metrics:\n{metrics_block}\n"
        )

    elif task == "forecast_explanation":
        metric_lines = []
        for m, v in facts.items():
            name = METRIC_DISPLAY_NAMES.get(m, m)
            metric_lines.append(
                f"{name}: {v.get('current')} -> {v.get('predicted')} (change: {v.get('change')}, direction: {v.get('direction')})"
            )
        metrics_block = "\n".join(metric_lines)
        return (
            f"Task: forecast_explanation\n\n"
            f"{instruction}\n"
            f"Predicted Changes:\n{metrics_block}\n"
        )

    elif task == "daily_note_analysis":
        classes = facts.get("classifications", {})
        class_lines = [f"{k}: {v}" for k, v in classes.items()]
        return (
            f"Task: daily_note_analysis\n\n"
            f"{instruction}\n"
            f"DailyNote: {facts.get('note', '')}\n"
            f"Wellness Score: {facts.get('wellness_score', 50)}\n"
            f"Classifications: {', '.join(class_lines)}\n"
        )

    return f"Task: {task}\n\n{instruction}\n"


# =====================================================================
# CORE REUSABLE INFERENCE FUNCTION
# =====================================================================

def generate_nlp_response(task: str, facts: Dict[str, Any], retry: bool = True) -> str:
    """
    Main entry point for hybrid NLP generation:
    1. Check facts
    2. Try loading model and running inference
    3. Validate output against facts
    4. If validation fails, retry once with stricter prompt before falling back
    """
    if not facts:
        return ""

    fallback_fn = {
        "daily_analysis": generate_fallback_daily,
        "weekly_analysis": generate_fallback_weekly,
        "forecast_explanation": generate_fallback_forecast,
        "daily_note_analysis": generate_fallback_daily_note,
    }.get(task, lambda f: "Analysis temporarily unavailable.")

    if not load_v5_model():
        return fallback_fn(facts)

    try:
        prompt = build_prompt_from_facts(task, facts)

        inputs = _tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        inputs = {key: value.to(_device) for key, value in inputs.items()}

        with torch.no_grad():
            output = _model.generate(
                **inputs,
                max_new_tokens=180,
                num_beams=4,
                early_stopping=True,
            )

        generated = _tokenizer.decode(
            output[0],
            skip_special_tokens=True
        ).strip()

        if validate_nlp_output(generated, task, facts):
            if task == "weekly_analysis":
                for h in ["WEEKLY SUMMARY", "METRIC COMPARISONS", "ROUTINE CHANGES", "NEXT WEEK"]:
                    generated = re.sub(rf"(?<!\n)(?={h})", "\n", generated)
                    generated = re.sub(rf"(?<={h})[ 	]*:?[ 	]*", "\n", generated)
            return generated

        # Validation failed: retry once with stricter prompt if requested
        if retry:
            stricter_prompt = (
                f"{prompt}\n"
                f"STRICT CONSTRAINT: State exact numbers from facts only. "
                f"Wellness Score is {facts.get('wellness_start')} to {facts.get('wellness_end')} ({facts.get('wellness_change')} change, {facts.get('wellness_direction')}). "
                f"Do NOT output meta instructions or altered numbers."
            )
            inputs_retry = _tokenizer(
                stricter_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=300,
            )
            inputs_retry = {key: value.to(_device) for key, value in inputs_retry.items()}

            with torch.no_grad():
                out_retry = _model.generate(
                    **inputs_retry,
                    max_new_tokens=180,
                    num_beams=4,
                    early_stopping=True,
                )

            gen_retry = _tokenizer.decode(
                out_retry[0],
                skip_special_tokens=True
            ).strip()

            if validate_nlp_output(gen_retry, task, facts):
                if task == "weekly_analysis":
                    for h in ["WEEKLY SUMMARY", "METRIC COMPARISONS", "ROUTINE CHANGES", "NEXT WEEK"]:
                        gen_retry = re.sub(rf"(?<!\n)(?={h})", "\n", gen_retry)
                        gen_retry = re.sub(rf"(?<={h})[ \t]*:?[ \t]*", "\n", gen_retry)
                return gen_retry

        # Safe fallback if model generation or retry is not fact-compliant
        return fallback_fn(facts)

    except Exception as error:
        print(f"Inference error for {task}:", repr(error))
        return fallback_fn(facts)

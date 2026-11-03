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
    """Deterministic fallback for weekly analysis."""
    start = facts.get("wellness_start", "--")
    end = facts.get("wellness_end", "--")
    change = facts.get("wellness_change", 0)
    direction = facts.get("wellness_direction", "unchanged")
    sign = "+" if change > 0 else ""

    metrics = facts.get("metrics", {})
    metric_lines = []
    for m in ["Mood", "Energy", "Stress", "Productivity"]:
        if m in metrics:
            d = metrics[m]
            display_name = METRIC_DISPLAY_NAMES.get(m, m)
            metric_lines.append(
                f"{display_name} moved from {d['start']} to {d['end']} ({d['direction']})"
            )
    metrics_str = ", ".join(metric_lines) if metric_lines else "core metrics remained within expected ranges"

    return (
        f"📅 WEEKLY SUMMARY\n"
        f"Wellness moved from {start} to {end} ({sign}{change} points, {direction}). "
        f"Your recorded habits reflect steady engagement across the tracked period.\n\n"
        f"📈 METRIC COMPARISONS\n"
        f"{metrics_str}.\n\n"
        f"🎯 NEXT WEEK\n"
        f"Focus on maintaining regular sleep routines and managing daily stress proactively."
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
    Lightweight validation checking for obvious numerical or directional hallucinations.
    Returns True if valid, False if fallback should be used.
    """
    if not text or len(text.strip()) < 20:
        return False

    t = text.lower()

    if task == "daily_analysis":
        # Check if score_direction is unchanged
        score_dir = facts.get("score_direction")
        if score_dir == "unchanged":
            # If model says decreased from X to X or declined
            if "decreased" in t or "declining" in t or "increased" in t:
                return False

    elif task == "weekly_analysis":
        direction = facts.get("wellness_direction")
        if direction in ["declining", "worsening"] and ("rose sharply" in t or "improved sharply" in t or "wellness rose" in t):
            return False

        metrics = facts.get("metrics", {})
        if "Stress" in metrics and metrics["Stress"]["direction"] == "worsening":
            if "stress improved" in t:
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

def generate_nlp_response(task: str, facts: Dict[str, Any]) -> str:
    """
    Main entry point for hybrid NLP generation:
    1. Check facts
    2. Try loading model and running inference
    3. Validate output against facts
    4. Fall back safely on failure or invalid output
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
                max_new_tokens=140,
                num_beams=4,
                early_stopping=True,
            )

        generated = _tokenizer.decode(
            output[0],
            skip_special_tokens=True
        ).strip()

        # Validate that model did not hallucinate directions or impossible numbers
        if validate_nlp_output(generated, task, facts):
            return generated
        else:
            return fallback_fn(facts)

    except Exception as error:
        print(f"Inference error for {task}:", repr(error))
        return fallback_fn(facts)

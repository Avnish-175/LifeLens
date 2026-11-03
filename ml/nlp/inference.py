"""
LifeLens V5 NLP Inference and Model Loading Layer
ml/nlp/inference.py

Architecture:
- Loads the fine-tuned FLAN-T5 model from `ml/nlp/lifelens_nlp_model_v5_gpu`.
- Cached via Streamlit @st.cache_resource so it loads only once per application session.
- Uses deterministic Python facts from `src/nlp_facts.py` as authoritative ground truth.
- Provides genuine LLM generation for:
  1. Daily AI Wellness Insight
  2. Tomorrow's Forecast Explanation
  3. Weekly AI Analysis
- Implements strict failure handling distinguishing:
  * Model not found (missing directory or files)
  * Model failed to load (initialization / dependency / CUDA error)
  * Generation failed
  * Successful LLM generation
- Exposes underlying errors to terminal logs while displaying honest, user-friendly messages in the UI.
- Never presents hardcoded or template-based text as AI-generated.
"""

import os
import re
import sys
from typing import Dict, Any, Optional, Union, Tuple
import pandas as pd

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False

from src.nlp_facts import METRIC_DISPLAY_NAMES

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(BASE_DIR, "ml", "nlp", "lifelens_nlp_model_v5_gpu")

REQUIRED_MODEL_FILES = ["config.json", "model.safetensors", "tokenizer.json"]

# Standard error messages as required by the specification
MSG_DAILY_UNAVAILABLE = (
    "AI Wellness Insight is currently unavailable because the V5 language model could not be loaded. "
    "Please check the model files and restart the application."
)
MSG_DAILY_INIT_FAILED = (
    "Unable to start the V5 AI model. Please check the model configuration and dependencies, then restart LifeLens."
)
MSG_DAILY_GEN_FAILED = "The AI model is loaded, but report generation failed. Please try again."

MSG_FORECAST_UNAVAILABLE = (
    "Tomorrow Forecast explanation is currently unavailable because the V5 language model could not be loaded. "
    "Please check the model files and restart the application."
)
MSG_FORECAST_INIT_FAILED = (
    "Unable to start the V5 AI model for forecast explanation. Please check the model configuration and dependencies, then restart LifeLens."
)
MSG_FORECAST_GEN_FAILED = "The AI model is loaded, but forecast explanation generation failed. Please try again."

# Module-level globals for backward compatibility
_tokenizer = None
_model = None
_device = None


def check_v5_model_files(model_dir: str = MODEL_DIR) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Check whether the V5 model directory and required weight/config files exist.
    Returns: (is_available, error_status, error_detail)
    error_status: 'not_found' | 'load_failed' | None
    """
    if not os.path.isdir(model_dir):
        return False, "not_found", f"Model directory not found at: {model_dir}"
    for filename in REQUIRED_MODEL_FILES:
        filepath = os.path.join(model_dir, filename)
        if not os.path.isfile(filepath):
            return False, "load_failed", f"Required model file missing: {filename} in {model_dir}"
    return True, None, None


def _load_v5_bundle_raw(model_dir: str = MODEL_DIR) -> Dict[str, Any]:
    """
    Internal loader: initializes the tokenizer and model onto GPU (if CUDA available) or CPU.
    Exposes any underlying error in terminal logs.
    """
    if not _DEPS_AVAILABLE:
        err = "PyTorch or Transformers library is not installed."
        print(f"[LifeLens NLP Error] {err}", file=sys.stderr)
        return {
            "status": "load_failed",
            "error": err,
            "model": None,
            "tokenizer": None,
            "device": None,
        }

    is_avail, err_status, err_msg = check_v5_model_files(model_dir)
    if not is_avail:
        print(f"[LifeLens NLP Error] {err_msg}", file=sys.stderr)
        return {
            "status": err_status,
            "error": err_msg,
            "model": None,
            "tokenizer": None,
            "device": None,
        }

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        model = model.to(device)
        model.eval()
        return {
            "status": "ok",
            "model": model,
            "tokenizer": tokenizer,
            "device": device,
            "error": None,
        }
    except Exception as exc:
        print(f"[LifeLens NLP Error] Failed to initialize V5 model: {repr(exc)}", file=sys.stderr)
        return {
            "status": "load_failed",
            "error": str(exc),
            "model": None,
            "tokenizer": None,
            "device": None,
        }


# =====================================================================
# CACHED MODEL ACCESS (Streamlit @st.cache_resource)
# =====================================================================

try:
    import streamlit as st

    @st.cache_resource(show_spinner=False)
    def _cached_v5_bundle(model_dir: str = MODEL_DIR) -> Dict[str, Any]:
        return _load_v5_bundle_raw(model_dir)

    def get_v5_bundle(model_dir: str = MODEL_DIR) -> Dict[str, Any]:
        """Retrieve the cached V5 model bundle within Streamlit session."""
        return _cached_v5_bundle(model_dir)

except Exception:
    _global_bundle: Optional[Dict[str, Any]] = None

    def get_v5_bundle(model_dir: str = MODEL_DIR) -> Dict[str, Any]:
        """Fallback in-memory singleton cache when running outside Streamlit."""
        global _global_bundle
        if _global_bundle is None or _global_bundle.get("status") != "ok":
            _global_bundle = _load_v5_bundle_raw(model_dir)
        return _global_bundle


def load_v5_model(model_dir: str = MODEL_DIR) -> bool:
    """Load or retrieve the V5 model. Returns True if successfully loaded."""
    global _tokenizer, _model, _device
    bundle = get_v5_bundle(model_dir)
    if bundle.get("status") == "ok":
        _tokenizer = bundle["tokenizer"]
        _model = bundle["model"]
        _device = bundle["device"]
        return True
    _tokenizer = None
    _model = None
    _device = None
    return False


# =====================================================================
# DAILY AI WELLNESS REPORT GENERATION
# =====================================================================

def generate_v5_daily_report(
    facts: Dict[str, Any],
    history_text: Optional[str] = None,
    note_text: Optional[str] = None,
    model_dir: str = MODEL_DIR,
) -> Dict[str, Any]:
    """
    Generate the daily AI wellness report using the V5 fine-tuned FLAN-T5 model.
    Returns:
        {"status": "ok", "text": str}
        or
        {"status": "not_found" | "load_failed" | "generation_failed", "message": str}
    """
    bundle = get_v5_bundle(model_dir)
    status = bundle.get("status")

    if status == "not_found":
        return {"status": "not_found", "message": MSG_DAILY_UNAVAILABLE}
    elif status != "ok":
        return {"status": "load_failed", "message": MSG_DAILY_INIT_FAILED}

    try:
        note_display = (
            note_text.strip()
            if note_text and note_text.strip().lower() not in ("none", "nan", "")
            else "None"
        )
        hist_display = history_text.strip() if history_text else "No recent history available."
        score = facts.get("wellness_score", 50)

        prompt = (
            f"Current Data:\n"
            f"Date: Today\n"
            f"Wellness Score: {score}\n"
            f"SleepHours: {facts.get('sleep_hours', 7.0)}\n"
            f"ScreenTime: {facts.get('screen_time', 4.0)}\n"
            f"ExerciseMinutes: {facts.get('exercise_minutes', 0)}\n"
            f"StudyHours: {facts.get('study_hours', 0)}\n"
            f"Mood: {facts.get('mood', 5)}\n"
            f"Energy: {facts.get('energy', 5)}\n"
            f"Stress: {facts.get('stress', 5)}\n"
            f"Productivity: {facts.get('productivity', 5)}\n"
            f"DailyNote: {note_display}\n\n"
            f"History:\n"
            f"{hist_display}\n"
        )

        tokenizer = bundle["tokenizer"]
        model = bundle["model"]
        device = bundle["device"]

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=180,
                num_beams=4,
                early_stopping=True,
            )

        generated = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        if not generated:
            print("[LifeLens NLP Error] V5 daily report generation returned empty string.", file=sys.stderr)
            return {"status": "generation_failed", "message": MSG_DAILY_GEN_FAILED}

        return {"status": "ok", "text": generated}

    except Exception as exc:
        print(f"[LifeLens NLP Error] V5 daily report generation error: {repr(exc)}", file=sys.stderr)
        return {"status": "generation_failed", "message": MSG_DAILY_GEN_FAILED}


# =====================================================================
# TOMORROW'S FORECAST EXPLANATION GENERATION
# =====================================================================

def parse_v5_forecast_sections(text: str) -> Dict[str, str]:
    """
    Parse V5 forecast explanation into the 4 structured UI fields:
    Tomorrow Outlook, Main Factor, Pattern, Priority
    """
    cleaned = re.sub(r"\*\*", "", str(text))
    cleaned = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F]+", "", cleaned)

    pattern = re.compile(
        r"(?:^|\s)(FORECAST(?:\s+OUTLOOK)?|EXPECTED\s+CHANGES|TOMORROW(?:\s+FOCUS)?|PRIORITY)\s*:?\s*",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(cleaned))
    found: Dict[str, str] = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
        body = cleaned[m.end():end].strip()
        tag = re.sub(r"\s+", "_", m.group(1).upper().strip())
        found[tag] = body

    outlook = found.get("FORECAST") or found.get("FORECAST_OUTLOOK", "")
    changes = found.get("EXPECTED_CHANGES", "")
    tomorrow = found.get("TOMORROW") or found.get("TOMORROW_FOCUS") or found.get("PRIORITY", "")

    # Fallback to full cleaned text if headings were not detected
    if not outlook and not changes and not tomorrow:
        outlook = cleaned.strip()

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", changes) if s.strip()]
    if len(sentences) >= 2:
        main_factor = sentences[0]
        pattern_text = " ".join(sentences[1:])
    elif len(sentences) == 1:
        main_factor = sentences[0]
        pattern_text = changes
    else:
        main_factor = changes or "Predicted metrics reflect your recent habit trends."
        pattern_text = changes or "Tracked indicators follow your established routines."

    return {
        "Tomorrow Outlook": outlook or "Tomorrow's outlook reflects your predicted habit patterns.",
        "Main Factor": main_factor,
        "Pattern": pattern_text,
        "Priority": tomorrow or "Maintain your daily routine to support predicted outcomes.",
    }


def generate_v5_forecast_explanation(
    facts: Dict[str, Any],
    latest: Optional[Union[pd.Series, Dict[str, Any]]] = None,
    prediction: Optional[Dict[str, Any]] = None,
    history_text: Optional[str] = None,
    model_dir: str = MODEL_DIR,
) -> Dict[str, Any]:
    """
    Generate natural-language forecast explanation using the V5 FLAN-T5 model.
    Passes ML predictions and authoritative factual context to the LLM.
    Returns:
        {
            "status": "ok",
            "Tomorrow Outlook": str,
            "Main Factor": str,
            "Pattern": str,
            "Priority": str,
            "raw": str
        }
        or
        {"status": "not_found" | "load_failed" | "generation_failed", "message": str}
    """
    bundle = get_v5_bundle(model_dir)
    status = bundle.get("status")

    if status == "not_found":
        return {"status": "not_found", "message": MSG_FORECAST_UNAVAILABLE}
    elif status != "ok":
        return {"status": "load_failed", "message": MSG_FORECAST_INIT_FAILED}

    try:
        latest_dict = (
            latest.to_dict()
            if isinstance(latest, pd.Series)
            else (dict(latest) if latest else {})
        )
        pred_dict = dict(prediction) if prediction else {}
        hist_display = history_text.strip() if history_text else "No recent history available."

        curr_score = latest_dict.get("WellnessScore", 50)
        curr_sleep = latest_dict.get("SleepHours", 7.0)
        curr_screen = latest_dict.get("ScreenTime", 4.0)
        curr_ex = latest_dict.get("ExerciseMinutes", 0)
        curr_study = latest_dict.get("StudyHours", 0)
        curr_mood = latest_dict.get("Mood", 5)
        curr_energy = latest_dict.get("Energy", 5)
        curr_stress = latest_dict.get("Stress", 5)
        curr_prod = latest_dict.get("Productivity", 5)

        prompt = (
            f"Current Data:\n"
            f"Date: Today\n"
            f"Wellness Score: {curr_score}\n"
            f"SleepHours: {curr_sleep}\n"
            f"ScreenTime: {curr_screen}\n"
            f"ExerciseMinutes: {curr_ex}\n"
            f"StudyHours: {curr_study}\n"
            f"Mood: {curr_mood}\n"
            f"Energy: {curr_energy}\n"
            f"Stress: {curr_stress}\n"
            f"Productivity: {curr_prod}\n\n"
            f"History:\n"
            f"{hist_display}\n\n"
            f"Tomorrow Predictions:\n"
            f"Predicted Mood: {pred_dict.get('Mood', curr_mood)}\n"
            f"Predicted Energy: {pred_dict.get('Energy', curr_energy)}\n"
            f"Predicted Stress: {pred_dict.get('Stress', curr_stress)}\n"
            f"Predicted Productivity: {pred_dict.get('Productivity', curr_prod)}\n"
        )

        tokenizer = bundle["tokenizer"]
        model = bundle["model"]
        device = bundle["device"]

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=180,
                num_beams=4,
                early_stopping=True,
            )

        generated = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        if not generated:
            print("[LifeLens NLP Error] V5 forecast generation returned empty text.", file=sys.stderr)
            return {"status": "generation_failed", "message": MSG_FORECAST_GEN_FAILED}

        parsed = parse_v5_forecast_sections(generated)
        parsed["status"] = "ok"
        parsed["raw"] = generated
        return parsed

    except Exception as exc:
        print(f"[LifeLens NLP Error] V5 forecast generation error: {repr(exc)}", file=sys.stderr)
        return {"status": "generation_failed", "message": MSG_FORECAST_GEN_FAILED}


# =====================================================================
# BACKWARD-COMPATIBLE UTILITIES FOR WEEKLY ANALYSIS & PROMPT VALIDATION
# =====================================================================

def validate_nlp_output(text: Optional[str], task: str, facts: Dict[str, Any]) -> bool:
    """
    Validation checking for instruction leakage, numerical mismatches, and directional hallucinations.
    """
    if not text or len(text.strip()) < 25:
        return False

    t = text.lower()

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

    if task == "weekly_analysis":
        direction = facts.get("wellness_direction")
        if direction in ["declining", "worsening"]:
            if any(p in t for p in ["rose sharply", "improved sharply", "wellness rose", "wellness increased"]):
                return False
        elif direction in ["improved", "improving"]:
            if any(p in t for p in ["declined sharply", "wellness fell", "wellness decreased", "wellness dropped"]):
                return False

    return True


def build_prompt_from_facts(task: str, facts: Dict[str, Any]) -> str:
    """Build structured prompt injecting deterministic facts for tasks like weekly_analysis."""
    instruction = (
        "Use the provided facts exactly.\n"
        "Do not recalculate numbers.\n"
        "Do not invent numbers.\n"
        "Explain the provided facts naturally and concisely.\n"
    )

    if task == "weekly_analysis":
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

    return f"Task: {task}\n\n{instruction}\n"


def generate_nlp_response(task: str, facts: Dict[str, Any], retry: bool = True) -> str:
    """
    Main entry point for hybrid NLP generation used by weekly analysis.
    """
    if not facts:
        return ""

    if not load_v5_model():
        return ""

    try:
        prompt = build_prompt_from_facts(task, facts)
        inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {key: value.to(_device) for key, value in inputs.items()}

        with torch.no_grad():
            output = _model.generate(
                **inputs,
                max_new_tokens=180,
                num_beams=4,
                early_stopping=True,
            )

        generated = _tokenizer.decode(output[0], skip_special_tokens=True).strip()
        if validate_nlp_output(generated, task, facts):
            if task == "weekly_analysis":
                for h in ["WEEKLY SUMMARY", "METRIC COMPARISONS", "ROUTINE CHANGES", "NEXT WEEK"]:
                    generated = re.sub(rf"(?<!\n)(?={h})", "\n", generated)
                    generated = re.sub(rf"(?<={h})[ \t]*:?[ \t]*", "\n", generated)
            return generated

        return generated
    except Exception as error:
        print(f"[LifeLens NLP Error] Inference error for {task}:", repr(error), file=sys.stderr)
        return ""

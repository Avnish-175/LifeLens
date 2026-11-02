import os



try:

    import torch

    from transformers import (

        AutoTokenizer,

        AutoModelForSeq2SeqLM

    )

    _DEPS_AVAILABLE = True

except ImportError:

    _DEPS_AVAILABLE = False





BASE_DIR = os.path.abspath(

    os.path.join(os.path.dirname(__file__), "..", "..")

)



MODEL_DIR = os.path.join(

    BASE_DIR,

    "ml",

    "nlp",

    "lifelens_nlp_model"

)





_tokenizer = None

_model = None

_device = None





def _load_model():

    global _tokenizer

    global _model

    global _device



    if not _DEPS_AVAILABLE:

        return False



    if _model is not None:

        return True



    if not os.path.exists(MODEL_DIR):

        return False



    try:

        _device = "cuda" if torch.cuda.is_available() else "cpu"



        _tokenizer = AutoTokenizer.from_pretrained(

            MODEL_DIR

        )



        _model = AutoModelForSeq2SeqLM.from_pretrained(

            MODEL_DIR

        )



        _model = _model.to(_device)

        _model.eval()



        return True



    except Exception:

        _tokenizer = None

        _model = None

        return False





def generate_wellness_report(
    wellness_score,
    wellness_category,
    latest,
    history=None,
    daily_note=None,
    tomorrow_prediction=None
):
    """
    Generate a dynamic LifeLens wellness analysis.

    The Wellness Score remains the source of truth.
    ML predictions are only used as additional context.
    """

    if wellness_score is None:
        return None

    if not _load_model():
        return None

    try:
        # -----------------------------
        # Recent history
        # -----------------------------
        history_text = "No recent history available."

        if history is not None and not history.empty:

            recent = history.tail(7).copy()

            history_lines = []

            for _, row in recent.iterrows():

                history_lines.append(
                    f"Date: {row['Date']}, "
                    f"Sleep: {float(row['SleepHours']):.1f}h, "
                    f"Mood: {float(row['Mood']):.1f}/10, "
                    f"Energy: {float(row['Energy']):.1f}/10, "
                    f"Stress: {float(row['Stress']):.1f}/10, "
                    f"Exercise: {float(row['ExerciseMinutes']):.0f}min, "
                    f"Screen Time: {float(row['ScreenTime']):.1f}h, "
                    f"Productivity: {float(row['Productivity']):.1f}/10"
                )

            history_text = "\n".join(history_lines)

        # -----------------------------
        # Daily note
        # -----------------------------
        note_text = (
            str(daily_note).strip()
            if daily_note is not None
            else ""
        )

        if not note_text:
            note_text = "No daily note provided."

        # -----------------------------
        # Tomorrow prediction
        # -----------------------------
        prediction_text = "No tomorrow prediction available."

        if tomorrow_prediction:
            prediction_text = (
                f"Mood: {tomorrow_prediction.get('Mood', 'N/A')}/10, "
                f"Energy: {tomorrow_prediction.get('Energy', 'N/A')}/10, "
                f"Stress: {tomorrow_prediction.get('Stress', 'N/A')}/10, "
                f"Productivity: {tomorrow_prediction.get('Productivity', 'N/A')}/10"
            )

        # -----------------------------
        # LLM prompt
        # -----------------------------
        prompt = f"""Analyze this LifeLens user's wellness data.

            Current Wellness Score: {wellness_score}/100
            Current Category: {wellness_category}

            CURRENT CHECK-IN
            Sleep: {float(latest["SleepHours"]):.1f} hours
            Mood: {float(latest["Mood"]):.1f}/10
            Energy: {float(latest["Energy"]):.1f}/10
            Stress: {float(latest["Stress"]):.1f}/10
            Exercise: {float(latest["ExerciseMinutes"]):.0f} minutes
            Screen Time: {float(latest["ScreenTime"]):.1f} hours
            Productivity: {float(latest["Productivity"]):.1f}/10

            RECENT HISTORY
            {history_text}

            DAILY NOTE
            {note_text}

            TOMORROW'S PREDICTION
            {prediction_text}

            Generate exactly 4 wellness insights using these exact headings:

            OVERALL
            KEY FACTOR
            PATTERN
            PRIORITY

            OVERALL:
            Briefly explain the user's current wellness state using the Wellness Score and Category.

            KEY FACTOR:
            Identify the single most important strength or weakness affecting the user's wellness.

            PATTERN:
            Identify one meaningful relationship or trend from the current metrics or recent history.

            PRIORITY:
            Give one practical and specific action the user should focus on next.

            Rules:
            - Use the provided LifeLens data.
            - Use recent history when it reveals a meaningful pattern.
            - Do not invent information.
            - Do not repeat the same point in multiple sections.
            - Keep each section concise.
            - Do not add any other headings, introductions, conclusions, numbering, or commentary.
            """

        inputs = _tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
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
                early_stopping=True
            )

        report = _tokenizer.decode(
            output[0],
            skip_special_tokens=True
        ).strip()

        if not report:
            return None

        return report

    except Exception as error:
        print("NLP report error:", repr(error))
        return None
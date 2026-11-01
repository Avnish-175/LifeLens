import math
def _valid_number(value):
    """Return a finite number, or None for a missing/invalid value."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if math.isfinite(number) else None


def calculate_wellness_score(
    sleep,
    mood,
    stress,
    energy,
    exercise,
    screen_time,
    productivity,
):
    
    # Each entry is (earned points, maximum available points).
    components = []

    # Sleep — maximum 25 points
    sleep = _valid_number(sleep)
    if sleep is not None and 0 <= sleep <= 24:
        if 7 <= sleep <= 9:
            points = 25
        elif 6 <= sleep < 7 or 9 < sleep <= 10:
            points = 20
        elif 5 <= sleep < 6:
            points = 15
        else:
            points = 8
        components.append((points, 25))

    # Mood — maximum 20 points
    mood = _valid_number(mood)
    if mood is not None and 1 <= mood <= 10:
        components.append(((mood / 10) * 20, 20))

    # Stress — maximum 20 points; lower stress scores better
    stress = _valid_number(stress)
    if stress is not None and 1 <= stress <= 10:
        components.append((((10 - stress) / 10) * 20, 20))

    # Energy — maximum 15 points
    energy = _valid_number(energy)
    if energy is not None and 1 <= energy <= 10:
        components.append(((energy / 10) * 15, 15))

    # Exercise — maximum 10 points
    exercise = _valid_number(exercise)
    if exercise is not None and 0 <= exercise <= 1440:
        if exercise >= 30:
            points = 10
        elif exercise >= 20:
            points = 8
        elif exercise >= 10:
            points = 5
        else:
            points = 2
        components.append((points, 10))

    # Screen time — maximum 5 points
    screen_time = _valid_number(screen_time)
    if screen_time is not None and 0 <= screen_time <= 24:
        if screen_time <= 2:
            points = 5
        elif screen_time <= 4:
            points = 4
        elif screen_time <= 6:
            points = 3
        else:
            points = 1
        components.append((points, 5))

    # Productivity — maximum 5 points
    productivity = _valid_number(productivity)
    if productivity is not None and 1 <= productivity <= 10:
        components.append(((productivity / 10) * 5, 5))

    # Do not invent a score if no usable factors exist.
    if not components:
        return None, "Insufficient data"

    earned_points = sum(points for points, _ in components)
    available_points = sum(maximum for _, maximum in components)

    # Normalize the available factors to a 100-point scale.
    score = round((earned_points / available_points) * 100, 1)

    if score >= 85:
        category = "Excellent"
    elif score >= 70:
        category = "Healthy"
    elif score >= 55:
        category = "Needs Improvement"
    else:
        category = "At Risk"

    return score, category



def get_score_breakdown(
    sleep,
    mood,
    stress,
    energy,
    exercise,
    screen_time,
    productivity,
):
    breakdown = []

    sleep = _valid_number(sleep)
    mood = _valid_number(mood)
    stress = _valid_number(stress)
    energy = _valid_number(energy)
    exercise = _valid_number(exercise)
    screen_time = _valid_number(screen_time)
    productivity = _valid_number(productivity)

    # Sleep: use the same point rules as the overall score.
    if sleep is not None and 0 <= sleep <= 24:
        if 7 <= sleep <= 9:
            score = 100
            status = "Excellent"
        elif 6 <= sleep < 7 or 9 < sleep <= 10:
            score = 80
            status = "Good"
        elif 5 <= sleep < 6:
            score = 60
            status = "Fair"
        else:
            score = 32
            status = "Needs Improvement"

        breakdown.append(("Sleep", score, status))

    # Mood: higher is better.
    if mood is not None and 1 <= mood <= 10:
        score = mood * 10
        status = "Good" if mood >= 7 else "Needs Improvement"
        breakdown.append(("Mood", score, status))

    # Stress: lower is better.
    if stress is not None and 1 <= stress <= 10:
        score = (10 - stress) / 9 * 100
        status = "Good" if stress <= 4 else "Needs Improvement"
        breakdown.append(("Stress", round(score, 1), status))

    # Energy: higher is better.
    if energy is not None and 1 <= energy <= 10:
        score = energy * 10
        status = "Good" if energy >= 7 else "Needs Improvement"
        breakdown.append(("Energy", score, status))

    # Exercise.
    if exercise is not None and 0 <= exercise <= 1440:
        if exercise >= 30:
            score, status = 100, "Excellent"
        elif exercise >= 20:
            score, status = 80, "Good"
        elif exercise >= 10:
            score, status = 50, "Fair"
        else:
            score, status = 20, "Needs Improvement"

        breakdown.append(("Exercise", score, status))

    # Screen time: lower is better.
    if screen_time is not None and 0 <= screen_time <= 24:
        if screen_time <= 2:
            score, status = 100, "Excellent"
        elif screen_time <= 4:
            score, status = 80, "Good"
        elif screen_time <= 6:
            score, status = 60, "Fair"
        else:
            score, status = 20, "Needs Improvement"

        breakdown.append(("Screen Time", score, status))

    # Productivity: higher is better.
    if productivity is not None and 1 <= productivity <= 10:
        score = productivity * 10
        status = "Good" if productivity >= 7 else "Needs Improvement"
        breakdown.append(("Productivity", score, status))

    return breakdown

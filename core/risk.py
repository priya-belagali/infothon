"""Rule-based baseline risk score (0-100). Transparent on purpose:
the ML model has to beat THIS before we claim it adds value."""


def _clip01(x):
    return max(0.0, min(1.0, float(x)))


def rule_score(delay_days, buffer_days, cover_days):
    """delay_days  : how late the shipment will be
    buffer_days : spare days between shipment ETA and the customer's promised date
    cover_days  : how many days current stock can keep production/orders going"""
    if delay_days <= 0:
        return 0
    gap = max(0.0, delay_days - buffer_days)        # days beyond the delivery buffer
    shortage = max(0.0, delay_days - cover_days)    # days stock cannot bridge
    both = 1.0 if (gap > 0 and shortage > 0) else 0.0
    return int(round(100 * (0.45 * _clip01(gap / 2) + 0.35 * _clip01(shortage / 2) + 0.20 * both)))


def risk_label(score):
    if score >= 50:
        return "HIGH"
    if score >= 20:
        return "MEDIUM"
    return "LOW"


def is_breach(delay_days, buffer_days, cover_days):
    """A delivery breach = delay beats the buffer AND stock cannot bridge the gap."""
    return bool(delay_days > buffer_days and cover_days < delay_days)

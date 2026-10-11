"""Recovery options. All numbers come from these deterministic rules - never from an LLM.
ASSUMPTIONS (change them freely): reroute = 12,000 + 30/unit, +1 day, 92% feasible;
transfer = 1,500 + 30/unit, 0 days, needs enough alternate stock."""
from dataclasses import dataclass, asdict


@dataclass
class Option:
    name: str
    cost: float
    delay_days: float
    feasibility: float          # 0-1; 0 means not possible
    breach: bool                # does the customer promise still break?
    note: str
    breach_risk: float = None   # only used for "No action" (ML probability)

    def to_dict(self):
        d = asdict(self)
        d["cost"] = f"Rs {self.cost:,.0f}"
        d["feasibility"] = f"{self.feasibility:.0%}"
        d["breach"] = "Yes" if self.breach else "No"
        return d


def generate_options(row, alt_units):
    qty, buffer = int(row["qty"]), float(row["buffer_days"])

    reroute = Option("Reroute shipment", 12000 + 30 * qty, 1.0, 0.92, 1.0 > buffer,
                     "Check capacity + transit time.")

    if alt_units >= qty:
        feas = 0.81 if alt_units >= 1.25 * qty else 0.60
    else:
        feas = 0.0
    transfer = Option("Transfer inventory", 1500 + 30 * qty, 0.0, feas, False,
                      "Check stock + replenishment risk." if feas else "Not enough alternate stock.")

    no_action = Option("No action", 0.0, float(row["delay_days"]), 1.0, bool(row["breach"]),
                       "Baseline for comparison.", breach_risk=row.get("ml_prob"))
    return [reroute, transfer, no_action]


def recommend(options):
    """Cheapest feasible option that avoids a breach. None = escalate to a human."""
    no_action = next(o for o in options if o.name == "No action")
    if not no_action.breach:
        return no_action                      # nothing to fix; just monitor
    feasible = [o for o in options if o.name != "No action" and o.feasibility > 0]
    safe = [o for o in feasible if not o.breach]
    pool = safe or feasible
    return min(pool, key=lambda o: o.cost) if pool else None

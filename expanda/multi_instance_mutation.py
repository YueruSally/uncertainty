"""Mask positions before selection without changing the five mutation operators."""
import math

import baseline_uncertainty as base
from learning_mutation import enumerate_targets
from multi_instance_policy import rule_probabilities


def candidates_for(ind, batches, operator, path_lib, timetable, arc_lookup, reliable_options):
    rows = enumerate_targets(ind, batches, operator, path_lib, timetable, arc_lookup)
    by_id = {b.batch_id: b for b in batches}
    for row in rows:
        batch = by_id[row["target"].batch_id]
        key = (batch.origin, batch.destination, batch.batch_id)
        allocs = ind.od_allocations.get(key, [])
        paths = path_lib.get((batch.origin, batch.destination), [])
        unallocated = [p for p in paths if p not in {a.path for a in allocs}]
        row["unallocated_path_count"] = len(unallocated)
        for name, attribute in (
                ("cost", "base_cost_per_teu"),
                ("emission", "base_emission_per_teu"),
                ("time", "base_travel_time_h")):
            values = [float(getattr(path, attribute)) for path in unallocated]
            row[f"alternative_min_{name}"] = min(values, default=0.0)
            row[f"alternative_mean_{name}"] = (
                sum(values) / len(values) if values else 0.0)
        options = (reliable_options or {}).get(key, [])
        row["available_reliable_options"] = len(options)
        probabilities = [float(option.on_time_probability) for option in options]
        finite_lateness = [float(option.max_lateness_h) for option in options
                           if math.isfinite(float(option.max_lateness_h))]
        row["reliable_best_on_time_probability"] = max(probabilities, default=0.0)
        row["reliable_mean_on_time_probability"] = (
            sum(probabilities) / len(probabilities) if probabilities else 0.0)
        row["reliable_min_max_lateness_h"] = min(finite_lateness, default=0.0)
        row["reliable_mean_max_lateness_h"] = (
            sum(finite_lateness) / len(finite_lateness) if finite_lateness else 0.0)
        if operator == "replace":
            if not options:
                row["eligible"] = False
                reason = "no_reliable_option"
            elif len(allocs) == 1 and allocs[0].share == 1.0 and all(
                    o.path == allocs[0].path for o in options):
                row["eligible"] = False
                reason = "same_only_replacement"
            else:
                row["eligible"] = True
                reason = None
        elif operator == "add" and not row["unallocated_path_count"]:
            # Existing add fallback can execute but replaces an existing path;
            # it is not an actual addition. Mask this guaranteed no-add case.
            row["eligible"] = False
            reason = "no_new_path"
        elif not row["eligible"]:
            reason = {"del": "single_or_no_path", "mod": "share_cannot_change",
                      "mode": "no_alternative_mode", "add": "no_path_library"}.get(operator,
                                                                                "structural")
        else:
            reason = None
        row["mask_reason"] = reason
    rule = rule_probabilities(rows)
    for row, p in zip(rows, rule):
        row["p_rule"] = p
    return rows

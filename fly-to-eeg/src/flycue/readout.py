"""Documented motor-population to actuator-drive mapping.

The two-channel mapping is an added movement interface, not a fitted muscle
model. It receives only actual motor-neuron states. No cue, requested side,
reward or target trajectory is accepted.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class MotorReadout:
    groups: tuple[np.ndarray, np.ndarray]
    modules: tuple[str, ...] = ("coxa swing", "femur/tr extend")
    rate_scale: float = 20.0

    @classmethod
    def from_graph(cls, graph, rate_scale: float = 20.0):
        modules = ("coxa swing", "femur/tr extend")
        groups = tuple(np.array([i for i, n in enumerate(graph.nodes)
                                if n.get("role") == "motor"
                                and n.get("motor_side") == side
                                and n.get("motor_module") in modules], dtype=int)
                       for side in ("L", "R"))
        if any(len(g) == 0 for g in groups):
            raise ValueError("Verified bilateral swing-related motor groups required")
        if not np.isfinite(rate_scale) or rate_scale <= 0:
            raise ValueError("Positive finite readout rate_scale required")
        return cls(groups, modules, float(rate_scale))

    @property
    def monitor_indices(self):
        return np.unique(np.concatenate(self.groups))

    def transform(self, activity: np.ndarray, monitor_indices: np.ndarray) -> np.ndarray:
        a = np.asarray(activity, dtype=float)
        if a.ndim != 2 or a.shape[1] != len(monitor_indices) or not np.isfinite(a).all():
            raise ValueError("Finite samples-by-monitors activity required")
        if np.any(a < -1e-12):
            raise ValueError("Readout expects nonnegative rate-model state")
        lookup = {int(index): col for col, index in enumerate(monitor_indices)}
        # Saturating sum gives a bounded drive without changing neural activity.
        population_sum = np.stack([a[:, [lookup[int(i)] for i in group]].sum(axis=1)
                                   for group in self.groups], axis=-1)
        return -np.expm1(-population_sum / self.rate_scale)

    def metadata(self, graph):
        return dict(type="fixed_saturating_motor_population_sum",
                    neural_modules=list(self.modules), rate_scale=self.rate_scale,
                    formula="drive = 1 - exp(-sum(nonnegative_motor_state)/rate_scale)",
                    output_units="dimensionless_actuator_drive",
                    empirical_muscle_calibration=False,
                    body_ids={side: graph.ids[group].tolist()
                              for side, group in zip(("L", "R"), self.groups)})


def candidate_inputs(graph, readout: MotorReadout, per_side: int = 5) -> list[int]:
    """Rank real excitatory upstream cells by anatomical motor-group preference.

    This is a controllability screen. It does not train or prescribe cue-to-leg
    associations; a subsequent policy chooses among tested input cells.
    """
    sums = np.stack([np.asarray(graph.W[g, :].sum(axis=0)).ravel()
                     for g in readout.groups], axis=0)
    eligible = np.array([n.get("role") != "motor" and float(n.get("sign", 0)) > 0
                         for n in graph.nodes])
    chosen: list[int] = []
    for side in (0, 1):
        target = np.maximum(sums[side], 0)
        other = np.maximum(sums[1-side], 0)
        score = np.log1p(target) * (target - other) / (target + other + 1)
        score[~eligible | (target == 0)] = -np.inf
        ranked = np.argsort(score)[::-1]
        for i in ranked[:per_side]:
            if np.isfinite(score[i]) and int(i) not in chosen:
                chosen.append(int(i))
    for i, node in enumerate(graph.nodes):
        if node.get("type") == "DNg100" and i not in chosen:
            chosen.append(i)
    return chosen

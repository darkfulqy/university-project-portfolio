"""Continuous connectome-constrained rate dynamics, not spikes or voltage.

The nonlinearity and parameter distributions follow Eq. 1 / Methods pp. 23–25
of Pugliese et al., bioRxiv 2025.09.12.675944, version 2026-04-30. Input and
threshold are arbitrary units. The state is an abstract continuous activity
reported in nominal Hz by that model; it is not an empirical firing rate.

W uses (postsynaptic, presynaptic) ordering and signed synapse counts. The
caller must document the graph and the source of ``normalized_sizes``. In
particular, an incoming-synapse proxy is NOT measured morphological volume.
This implementation offers exponential-leak Euler and classical RK4, rather
than the paper's adaptive Dopri5; applications must assess time-step convergence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RateParameters:
    gain: FloatArray
    threshold: FloatArray
    rmax: FloatArray
    tau_s: FloatArray
    seed: int

    def as_dict(self) -> dict:
        """Return the complete parameter ensemble for trial provenance."""
        return {
            "gain": self.gain.tolist(),
            "threshold": self.threshold.tolist(),
            "rmax": self.rmax.tolist(),
            "tau_s": self.tau_s.tolist(),
            "seed": self.seed,
        }


@dataclass(frozen=True)
class SimulationResult:
    time_s: FloatArray
    activity: FloatArray
    monitor_indices: NDArray[np.int64]
    final_state: FloatArray
    all_activity: FloatArray | None
    diagnostics: dict
    metadata: dict


def _positive_vector(value: ArrayLike, name: str, n: int | None = None) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 1 or not array.size or (n is not None and array.size != n):
        raise ValueError(f"{name} must be a nonempty vector of length {n}")
    if not np.isfinite(array).all() or np.any(array <= 0):
        raise ValueError(f"{name} must contain finite, strictly positive values")
    return array


def _positive_normal(rng: np.random.Generator, mean: float, sd: float, n: int) -> FloatArray:
    """Rejection sampling implements truncation, without a point mass at zero."""
    draws = rng.normal(mean, sd, n)
    invalid = draws <= 0
    while invalid.any():
        draws[invalid] = rng.normal(mean, sd, int(invalid.sum()))
        invalid = draws <= 0
    return draws


def sample_parameters(normalized_sizes: ArrayLike, seed: int) -> RateParameters:
    """Draw one independent cell-parameter ensemble; sizes are already normalized.

    All Gaussian SDs are absolute, not variances. Size scaling is gain = a*/s
    and threshold = theta* × s. No normalization or size substitution is hidden.
    """
    sizes = _positive_vector(normalized_sizes, "normalized_sizes")
    rng = np.random.default_rng(seed)
    n = sizes.size
    return RateParameters(
        gain=_positive_normal(rng, 1.0, 0.1, n) / sizes,
        threshold=_positive_normal(rng, 7.5, 0.6, n) * sizes,
        rmax=_positive_normal(rng, 200.0, 10.0, n),
        tau_s=_positive_normal(rng, 0.020, 0.002, n),
        seed=int(seed),
    )


def _indices(value: ArrayLike, n: int, name: str) -> NDArray[np.int64]:
    raw = np.asarray(value)
    if raw.ndim != 1 or (raw.size and not np.issubdtype(raw.dtype, np.integer)):
        raise ValueError(f"{name} must be a one-dimensional integer vector")
    indices = raw.astype(np.int64)
    if np.any(indices < 0) or np.any(indices >= n) or len(np.unique(indices)) != len(indices):
        raise ValueError(f"{name} must contain distinct indices in [0, {n})")
    return indices


class RateNetwork:
    """Fixed CSR connectome with seeded, heterogeneous continuous dynamics."""

    def __init__(
        self,
        W: sparse.spmatrix | ArrayLike,
        normalized_sizes: ArrayLike,
        *,
        seed: int,
        b: float = 0.03,
        parameters: RateParameters | None = None,
    ) -> None:
        self.W = sparse.csr_matrix(W, dtype=np.float64, copy=True)
        if self.W.shape[0] != self.W.shape[1] or self.W.shape[0] == 0:
            raise ValueError("W must be a nonempty square matrix in (post, pre) order")
        self.W.sum_duplicates()
        self.W.sort_indices()
        if not np.isfinite(self.W.data).all():
            raise ValueError("W contains non-finite weights")
        self.n = self.W.shape[0]
        self.normalized_sizes = _positive_vector(normalized_sizes, "normalized_sizes", self.n)
        if not np.isfinite(b) or b < 0:
            raise ValueError("b must be finite and nonnegative")
        self.b = float(b)
        p = parameters or sample_parameters(self.normalized_sizes, seed)
        if int(p.seed) != int(seed):
            raise ValueError("parameter seed and network seed must match")
        self.parameters = RateParameters(
            _positive_vector(p.gain, "gain", self.n),
            _positive_vector(p.threshold, "threshold", self.n),
            _positive_vector(p.rmax, "rmax", self.n),
            _positive_vector(p.tau_s, "tau_s", self.n),
            int(seed),
        )

    def _target(self, state: FloatArray, indices: NDArray[np.int64], drive: FloatArray) -> FloatArray:
        p = self.parameters
        # Model rectification is intentional. Numerical failures are never clipped.
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            net_input = self.b * (self.W @ state) - p.threshold
            net_input[indices] += drive
            if not np.isfinite(net_input).all():
                raise FloatingPointError("Non-finite recurrent or external input")
            scaled_input = (p.gain / p.rmax) * net_input
            return p.rmax * np.maximum(np.tanh(scaled_input), 0.0)

    def simulate(
        self,
        *,
        duration_s: float,
        dt_s: float,
        stimulation_indices: ArrayLike = (),
        stimulation: ArrayLike | Callable[[float], ArrayLike] | None = None,
        monitor_indices: ArrayLike | None = None,
        output_every: int = 1,
        initial_state: ArrayLike | None = None,
        record_all: bool = False,
        solver: str = "exponential_leak_euler",
    ) -> SimulationResult:
        """Integrate a trial, with inputs held constant on each [t, t + dt) step.

        ``stimulation`` is a vector of length n_stim, an array with shape
        (n_steps, n_stim), or a callable of time returning that vector. A pulse
        boundary must lie on the dt grid. All cells start at zero unless an
        explicit initial state is provided. Returned samples include t=0 and
        the final time, even if output_every does not divide n_steps.

        Monitoring affects storage only. All network cells are always evolved.
        This routine does not assign behavior labels or create complete trial
        provenance; the caller supplies graph/cell IDs, stimulus and split group.

        ``solver`` is "exponential_leak_euler" or "rk4". RK4 holds external
        input fixed through all four stages of each step (zero-order hold).
        It rejects dt > min(tau) and any state leaving the model bounds; this
        conservative leak check is not a recurrent-network convergence proof.
        """
        if not np.isfinite(duration_s) or duration_s <= 0 or not np.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("duration_s and dt_s must be finite and strictly positive")
        n_steps = int(round(duration_s / dt_s))
        if n_steps < 1 or not np.isclose(n_steps * dt_s, duration_s, rtol=1e-10, atol=1e-12):
            raise ValueError("duration_s must be an integer multiple of dt_s")
        if isinstance(output_every, bool) or not isinstance(output_every, (int, np.integer)) or output_every < 1:
            raise ValueError("output_every must be a positive integer")
        stim_idx = _indices(stimulation_indices, self.n, "stimulation_indices")
        mon_idx = np.arange(self.n, dtype=np.int64) if monitor_indices is None else _indices(
            monitor_indices, self.n, "monitor_indices"
        )
        if stimulation is None:
            stimulation = np.zeros(stim_idx.size)
        dynamic = callable(stimulation)
        if not dynamic:
            stimulation = np.asarray(stimulation, dtype=np.float64)
            if stimulation.shape not in ((stim_idx.size,), (n_steps, stim_idx.size)):
                raise ValueError("stimulation must have shape (n_stim,) or (n_steps, n_stim)")
            if not np.isfinite(stimulation).all():
                raise ValueError("stimulation contains non-finite values")
        state = np.zeros(self.n) if initial_state is None else np.array(initial_state, dtype=np.float64, copy=True)
        p = self.parameters
        if solver not in ("exponential_leak_euler", "rk4"):
            raise ValueError("solver must be exponential_leak_euler or rk4")
        if solver == "rk4" and dt_s > p.tau_s.min():
            raise ValueError("rk4 requires dt_s <= min(tau_s); additionally check convergence")
        if state.shape != (self.n,) or not np.isfinite(state).all():
            raise ValueError("initial_state must be a finite vector of length n_cells")
        if np.any(state < 0) or np.any(state > p.rmax):
            raise ValueError("initial_state must lie between zero and each cell's rmax")
        alpha = -np.expm1(-dt_s / p.tau_s)
        sample_steps = np.arange(0, n_steps + 1, output_every, dtype=np.int64)
        if sample_steps[-1] != n_steps:
            sample_steps = np.append(sample_steps, n_steps)
        activity = np.empty((len(sample_steps), len(mon_idx)))
        all_activity = np.empty((len(sample_steps), self.n)) if record_all else None
        activity[0] = state[mon_idx]
        if all_activity is not None:
            all_activity[0] = state
        peak = float(state.max())
        peak_fraction_rmax = float(np.max(state / p.rmax))
        ever_active = state > 0.01
        ever_saturated = state >= 0.95 * p.rmax
        saturated_cell_steps = int(ever_saturated.sum())
        target_saturated_cell_steps = 0
        out = 1
        for step in range(n_steps):
            if dynamic:
                drive = np.asarray(stimulation(step * dt_s), dtype=np.float64)
                if drive.shape != (stim_idx.size,) or not np.isfinite(drive).all():
                    raise ValueError(f"stimulation callback returned invalid input at step {step}")
            else:
                drive = stimulation if stimulation.ndim == 1 else stimulation[step]
            target = self._target(state, stim_idx, drive)
            target_saturated_cell_steps += int(np.count_nonzero(target >= 0.95 * p.rmax))
            if solver == "exponential_leak_euler":
                # Convex update preserves the bounded ODE state without clipping.
                state = (1.0 - alpha) * state + alpha * target
            else:
                k1 = (target - state) / p.tau_s
                stage2 = state + 0.5 * dt_s * k1
                target2 = self._target(stage2, stim_idx, drive)
                k2 = (target2 - stage2) / p.tau_s
                stage3 = state + 0.5 * dt_s * k2
                target3 = self._target(stage3, stim_idx, drive)
                k3 = (target3 - stage3) / p.tau_s
                stage4 = state + dt_s * k3
                target4 = self._target(stage4, stim_idx, drive)
                k4 = (target4 - stage4) / p.tau_s
                state = state + dt_s / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
                target_saturated_cell_steps += sum(int(np.count_nonzero(t >= 0.95 * p.rmax))
                                                   for t in (target2, target3, target4))
            if not np.isfinite(state).all():
                raise FloatingPointError(f"Non-finite state at step {step + 1}")
            if np.any(state < 0) or np.any(state > p.rmax):
                raise FloatingPointError(f"State outside [0, rmax] at step {step + 1}; reduce dt_s")
            peak = max(peak, float(state.max()))
            ratio = state / p.rmax
            peak_fraction_rmax = max(peak_fraction_rmax, float(ratio.max()))
            saturated = ratio >= 0.95
            saturated_cell_steps += int(saturated.sum())
            ever_saturated |= saturated
            ever_active |= state > 0.01
            if step + 1 == sample_steps[out]:
                activity[out] = state[mon_idx]
                if all_activity is not None:
                    all_activity[out] = state
                out += 1
        return SimulationResult(
            time_s=sample_steps * dt_s,
            activity=activity,
            monitor_indices=mon_idx,
            final_state=state.copy(),
            all_activity=all_activity,
            diagnostics={
                "all_finite": True,
                "peak_abstract_activity_nominal_hz": peak,
                "peak_fraction_of_rmax": peak_fraction_rmax,
                "ever_active_cell_fraction": float(ever_active.mean()),
                "ever_saturated_cell_fraction": float(ever_saturated.mean()),
                "saturated_cell_time_fraction": saturated_cell_steps / (self.n * (n_steps + 1)),
                "target_saturated_cell_time_fraction": target_saturated_cell_steps / (
                    self.n * n_steps * (4 if solver == "rk4" else 1)),
                "target_saturation_observations_per_step": 4 if solver == "rk4" else 1,
                "saturation_threshold_fraction_of_rmax": 0.95,
                "dt_over_min_tau": float(dt_s / p.tau_s.min()),
            },
            metadata={
                "model": "pugliese_rectified_tanh_rate_equation_1",
                "paper_version": "2026-04-30",
                "activity_kind": "abstract_continuous_rate",
                "activity_units": "arbitrary_units",
                "paper_activity_label": "nominal_Hz_abstract_activity_not_empirical_spike_rate",
                "input_units": "arbitrary_units",
                "time_units": "s",
                "solver": solver,
                "dt_s": float(dt_s),
                "output_every": int(output_every),
                "duration_s": float(duration_s),
                "parameter_seed": p.seed,
                "synaptic_scaling_b": self.b,
                "weight_orientation": "postsynaptic_by_presynaptic",
                "initial_state": "zero" if initial_state is None else "caller_supplied",
                "input_timing": "left_endpoint_held_for_dt",
                "state_clipping": False,
                "size_provenance": "caller_must_record",
                "n_cells": self.n,
                "n_steps": n_steps,
            },
        )

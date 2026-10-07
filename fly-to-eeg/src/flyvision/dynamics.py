"""New, uncalibrated whole-CNS continuous activity model (arbitrary units).

This is not the existing leg model and is not an LFP forward model. Contact
counts constrain relative coupling; leak, tonic input and gains are assumptions.
For max_i sum_j |W_ij| <= 1 and coupling < 1 the fixed-point map is a contraction.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class Parameters:
    tau_s: np.ndarray
    tonic_input: np.ndarray
    coupling: float
    seed: int


def parameters(n: int, seed: int, coupling: float = 0.8,
               tau_range_s=(0.018, 0.022), tonic_range_au=(0.225, 0.275)) -> Parameters:
    if n < 1 or not 0 <= coupling < 1:
        raise ValueError("positive cell count and contractive coupling required")
    tau = np.asarray(tau_range_s, dtype=float)
    tonic = np.asarray(tonic_range_au, dtype=float)
    if (tau.shape != (2,) or tonic.shape != (2,) or not np.isfinite(tau).all()
            or not np.isfinite(tonic).all() or tau[0] <= 0 or tau[1] < tau[0]
            or tonic[1] < tonic[0]):
        raise ValueError("finite ordered parameter ranges and positive tau required")
    rng = np.random.default_rng(seed)
    return Parameters(rng.uniform(*tau, n), rng.uniform(*tonic, n), float(coupling), int(seed))


def flash_train(duration_s: float, dt_s: float, onset_s: float, offset_s: float,
                frequency_hz: float = 10, duty: float = 0.5) -> np.ndarray:
    """Return interval inputs. Pulse boundaries must lie on the integration grid."""
    if not (0 < dt_s and 0 <= onset_s < offset_s <= duration_s and
            frequency_hz > 0 and 0 < duty < 1):
        raise ValueError("invalid stimulus timing")
    quantities = np.array([duration_s, onset_s, offset_s, 1 / frequency_hz,
                           duty / frequency_hz]) / dt_s
    if not np.allclose(quantities, np.round(quantities), rtol=0, atol=1e-8):
        raise ValueError("duration, pulse and onset/offset boundaries must align with dt")
    steps, on, off, period, high = np.round(quantities).astype(int)
    k = np.arange(steps)
    return ((k >= on) & (k < off) & ((k - on) % period < high)).astype(float)


class ContinuousCNS:
    def __init__(self, weights, p: Parameters):
        self.W = sparse.csr_matrix(weights, dtype=np.float64)
        self.n = self.W.shape[0]
        if self.W.shape != (self.n, self.n) or self.n < 1:
            raise ValueError("square nonempty graph required")
        if not np.isfinite(self.W.data).all():
            raise ValueError("nonfinite coupling")
        self.row_bound = float(abs(self.W).sum(axis=1).max())
        if self.row_bound > 1 + 1e-6:
            raise ValueError("absolute row sums must be <= 1")
        if not 0 <= p.coupling < 1 or p.tau_s.shape != (self.n,) or p.tonic_input.shape != (self.n,):
            raise ValueError("invalid parameter dimensions or coupling")
        if not (np.isfinite(p.tau_s).all() and np.isfinite(p.tonic_input).all()) or np.any(p.tau_s <= 0):
            raise ValueError("finite parameters and positive tau required")
        self.p = p

    def target(self, state, input_indices, drive):
        current = self.p.tonic_input + self.p.coupling * (self.W @ state)
        current[input_indices] += drive
        return np.maximum(np.tanh(current), 0)

    def resting_state(self, tolerance=1e-11):
        state = np.maximum(np.tanh(self.p.tonic_input), 0)
        empty = np.array([], dtype=int)
        for iteration in range(1000):
            nxt = self.target(state, empty, 0)
            residual = float(np.max(np.abs(nxt - state)))
            state = nxt
            if residual < tolerance:
                return state, {"iterations": iteration + 1, "fixed_point_residual": residual}
        raise RuntimeError("resting state did not converge")

    def simulate(self, stimulus, dt_s, input_indices, observer, *, amplitude=0.8,
                 initial_state=None, output_every=1, progress=None):
        stimulus = np.asarray(stimulus, dtype=float)
        indices = np.asarray(input_indices, dtype=int)
        observer = sparse.csr_matrix(observer, dtype=np.float64)
        if stimulus.ndim != 1 or not stimulus.size or not np.isfinite(stimulus).all():
            raise ValueError("finite 1D stimulus required")
        if (indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= self.n)
                or len(np.unique(indices)) != len(indices)):
            raise ValueError("distinct valid input indices required")
        if observer.shape[1] != self.n or not np.isfinite(observer.data).all():
            raise ValueError("invalid observation matrix")
        if not np.isfinite(dt_s) or dt_s <= 0 or dt_s > self.p.tau_s.min():
            raise ValueError("0 < dt <= min(tau) required; also check convergence")
        if not np.isfinite(amplitude) or output_every < 1 or int(output_every) != output_every:
            raise ValueError("finite amplitude and positive integer output_every required")
        state, equilibration = self.resting_state() if initial_state is None else (np.array(initial_state, dtype=float, copy=True), None)
        if state.shape != (self.n,) or not np.isfinite(state).all() or np.any(state < 0) or np.any(state > 1):
            raise ValueError("state must be finite, length n and within [0, 1]")
        initial = state.copy()
        sample_steps = np.arange(0, len(stimulus) + 1, output_every)
        if sample_steps[-1] != len(stimulus):
            raise ValueError("output interval must divide trial duration")
        observed = np.empty((len(sample_steps), observer.shape[0]))
        observed[0] = observer @ state
        alpha = -np.expm1(-dt_s / self.p.tau_s)
        max_change = np.zeros(self.n)
        start = time.perf_counter()
        report_at = start + 25
        out = 1
        for k, value in enumerate(stimulus):
            target = self.target(state, indices, amplitude * value)
            state += alpha * (target - state)
            max_change = np.maximum(max_change, np.abs(state - initial))
            if (k + 1) % output_every == 0:
                observed[out] = observer @ state
                out += 1
            now = time.perf_counter()
            if progress and now >= report_at:
                progress(f"{k + 1}/{len(stimulus)} steps; simulated {(k + 1) * dt_s:.3f}s; wall {now - start:.1f}s")
                report_at = now + 25
        if not np.isfinite(state).all() or not np.isfinite(observed).all():
            raise FloatingPointError("nonfinite result")
        return {"time_s": sample_steps * dt_s, "population_activity_au": observed,
                "initial_state_au": initial, "final_state_au": state,
                "max_cell_change_au": max_change,
                "diagnostics": {"wall_time_s": time.perf_counter() - start,
                    "n_integrated_cells": self.n, "n_steps": len(stimulus),
                    "all_finite": True, "equilibration": equilibration,
                    "absolute_weight_row_sum_max": self.row_bound,
                    "contractivity_bound": self.p.coupling * self.row_bound,
                    "max_state_change_au": float(max_change.max()),
                    "responding_cells_above_1e_6_au": int(np.count_nonzero(max_change > 1e-6)),
                    "maximum_final_activity_au": float(state.max())}}

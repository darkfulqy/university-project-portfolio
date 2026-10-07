"""Shared, label-honest reference policies for Paper B.

Rows are evaluation cells (for example subject x seed x target session) and
columns are adaptation methods. Gains are relative to R0, whose gain is zero.
The default estimand gives each subject equal mass, then each cell within a
subject equal mass. Method ties follow column order; a tie with R0 favours R0.

Inputs used to choose a policy must themselves be free of held-out-label
information. In particular, globally fitted empirical-Bayes posterior means
are not a valid substitute for raw gains when selecting the training-fold
fixed method.
"""
from __future__ import annotations

import numpy as np


def _subjects(subjects):
    subjects = np.asarray(subjects)
    if subjects.ndim != 1 or len(subjects) == 0:
        raise ValueError("subjects must be a nonempty one-dimensional array")
    if any(s is None or s != s for s in subjects):
        raise ValueError("subjects must not contain missing values")
    return subjects


def subject_balanced_weights(subjects):
    """Return normalized cell weights, giving every subject equal total mass."""
    subjects = _subjects(subjects)
    _, inverse, counts = np.unique(subjects, return_inverse=True, return_counts=True)
    return 1.0 / (len(counts) * counts[inverse])


def cell_weights(subjects, weighting="subject"):
    """Normalized weights for either the subject- or cell-average estimand."""
    subjects = _subjects(subjects)
    if weighting == "subject":
        return subject_balanced_weights(subjects)
    if weighting == "cell":
        return np.full(len(subjects), 1.0 / len(subjects))
    raise ValueError("weighting must be 'subject' or 'cell'")


def _gains(gains, subjects):
    gains = np.asarray(gains, dtype=float)
    if gains.ndim != 2 or gains.shape[0] != len(subjects) or gains.shape[1] == 0:
        raise ValueError("gains must have shape (len(subjects), n_methods >= 1)")
    if not np.isfinite(gains).all():
        raise ValueError("gains must contain only finite values")
    return gains


def select_fixed_method(train_gains, train_subjects, include_no_adapt=True,
                        weighting="subject"):
    """Return one training-only choice: method-column index, or -1 for R0.

    This primitive is also suitable for inner folds of a nested gate or
    selector. Only the supplied training cells are used to choose a method.
    A training fold may contain a single subject.
    """
    train_subjects = _subjects(train_subjects)
    train_gains = _gains(train_gains, train_subjects)
    means = cell_weights(train_subjects, weighting) @ train_gains
    method = int(np.argmax(means))
    return method if not include_no_adapt or means[method] > 0.0 else -1


def loso_fixed_policy(gains, subjects, include_no_adapt=True, weighting="subject"):
    """Select a fixed policy in each training fold; return held-out outcomes.

    Returns ``(realized_gains, choices)`` with one entry per input row.
    ``choices == -1`` means R0, otherwise the value is a zero-based method
    column index. The same choice is applied to all cells of a held-out
    subject. Negative held-out outcomes are preserved: test labels never
    determine whether the selected method runs.

    The training-fold mean uses the requested weighting, recomputed using
    only that fold. R0 participates in *training* selection when enabled.
    Exact ties with R0 choose R0; adaptation-method ties choose the first
    column. At least two subjects and one method are required.
    """
    subjects = _subjects(subjects)
    gains = _gains(gains, subjects)
    unique_subjects = np.unique(subjects)
    if len(unique_subjects) < 2:
        raise ValueError("LOSO requires at least two subjects")
    # Validate the estimand even before entering the folds.
    cell_weights(subjects, weighting)
    realized = np.zeros(len(subjects), dtype=float)
    choices = np.full(len(subjects), -1, dtype=int)
    for subject in unique_subjects:
        test = subjects == subject
        train = ~test
        method = select_fixed_method(gains[train], subjects[train],
                                     include_no_adapt=include_no_adapt,
                                     weighting=weighting)
        if method != -1:
            choices[test] = method
            realized[test] = gains[test, method]
    return realized, choices

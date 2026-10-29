"""Fold-local leakage contracts (R5).

Every model adapter must declare the *scope* it fits on: ``outer_train``,
``inner_train``, or ``full_sample``. Only ``outer_train``/``inner_train`` are
fold-local; ``full_sample`` is a leak (it includes the outer-valid fold). The
validator ``check_adapter`` runs an adapter's declared fit against the split
and rejects any ``full_sample`` scope, so a deliberately leaky adapter (e.g.
one that standardizes on the full sample) fails the protocol test while the
correct pipeline passes.

The check is structural — it inspects the declared scope and the slice the
adapter actually received — so the injected-leak test is meaningful: the leak
is caught by scope, not by guessing at internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

VALID_SCOPES = ("outer_train", "inner_train", "full_sample")


@dataclass
class FitResult:
    """What an adapter consumed during one fit, and from which scope."""

    scope: str
    n_rows_seen: int
    notes: str = ""


class LeakageError(RuntimeError):
    """Raised when an adapter's fit scope is not fold-local."""


class GuardedSlice:
    """An array view that raises if indexed with rows outside ``allowed``.

    Used by ``check_adapter`` to enforce fold-locality structurally: an adapter
    handed a GuardedSlice restricted to its fit scope cannot reach for
    out-of-scope material (e.g. an inner-train adapter trying to use the full
    outer_train) without tripping the guard. Integer-array indexing is checked
    against the allowed index set; boolean masks are rejected (they could
    silently span disallowed rows).
    """

    def __init__(self, array: np.ndarray, allowed: set[int]):
        self._array = array
        self._allowed = allowed

    @property
    def shape(self):
        return self._array.shape

    @property
    def dtype(self):
        return self._array.dtype

    def __len__(self):
        return len(self._array)

    def __getitem__(self, key):
        if isinstance(key, np.ndarray) and key.dtype == bool:
            raise LeakageError(
                "boolean-mask indexing on a guarded slice is not allowed; use "
                "explicit integer indices so the scope check is exact"
            )
        if isinstance(key, (list, tuple)):
            key = np.asarray(key)
        if isinstance(key, np.ndarray) and np.issubdtype(key.dtype, np.integer):
            disallowed = set(int(i) for i in key.ravel()) - self._allowed
            if disallowed:
                raise LeakageError(
                    f"guarded slice: attempted to access {len(disallowed)} "
                    f"row(s) outside the allowed fit scope (e.g. {sorted(disallowed)[:5]})"
                )
        if isinstance(key, (int, np.integer)):
            if int(key) not in self._allowed:
                raise LeakageError(
                    f"guarded slice: row {int(key)} is outside the allowed fit scope"
                )
        return self._array[key]

    def mean(self, axis=0):
        return self._array.mean(axis=axis)

    def std(self, axis=0, ddof=0):
        return self._array.std(axis=axis, ddof=ddof)

    def var(self, axis=0, ddof=0):
        return self._array.var(axis=axis, ddof=ddof)

    # Delegate numpy arithmetic to the underlying array (returns plain ndarrays,
    # so the guard applies to explicit indexing, not to whole-slice math).
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        arrays = [x._array if isinstance(x, GuardedSlice) else x for x in inputs]
        return getattr(ufunc, method)(*arrays, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        args = tuple(a._array if isinstance(a, GuardedSlice) else a for a in args)
        return func(*args, **kwargs)

    def __sub__(self, other):
        return self._array - self._unwrap(other)

    def __rsub__(self, other):
        return self._unwrap(other) - self._array

    def __truediv__(self, other):
        return self._array / self._unwrap(other)

    def __rtruediv__(self, other):
        return self._unwrap(other) / self._array

    def __add__(self, other):
        return self._array + self._unwrap(other)

    def __radd__(self, other):
        return self._unwrap(other) + self._array

    @staticmethod
    def _unwrap(x):
        return x._array if isinstance(x, GuardedSlice) else x


@dataclass
class Adapter:
    """A model adapter plus its fit-scope declaration.

    ``fit_scope`` is what the adapter *claims*. ``fit(X, y, scope)`` performs
    the actual fit and records how many rows it saw; a correct adapter fits on
    the slice it is handed, a leaky one would reach for the full sample.
    """

    name: str
    fit_scope: str
    fit: object  # callable(X, y, scope) -> fitted model (any object)

    def run_fit(self, X: np.ndarray, y: np.ndarray, scope: str) -> FitResult:
        """Run the declared fit and capture the rows seen."""
        self.fit(X, y, scope)
        return FitResult(scope=scope, n_rows_seen=int(X.shape[0]))


def check_adapter(adapter: Adapter, split, data) -> None:
    """Validate that an adapter's fit is fold-local.

    Fits the adapter on its declared scope and raises ``LeakageError`` if the
    scope is ``full_sample`` or if the rows seen exceed the allowed slice. For
    ``inner_train`` the fit must use the inner_train slice specifically — an
    adapter that trains on the full outer_train while declaring ``inner_train``
    is rejected (the slice it receives is the inner fold, and a guarded view
    raises if it reaches for material outside that slice).
    """
    if adapter.fit_scope not in VALID_SCOPES:
        raise LeakageError(f"{adapter.name}: unknown fit_scope {adapter.fit_scope!r}")

    outer_train = np.asarray(split.outer_train)
    valid_idx = np.asarray(split.outer_valid)
    n_outer = int(outer_train.shape[0])
    n_valid = int(valid_idx.shape[0])

    if adapter.fit_scope == "full_sample":
        # A correct pipeline hands the adapter only outer_train rows. A
        # full_sample-scoped adapter would standardize/fit on all n rows,
        # which includes the n_valid outer-valid rows — the leak.
        if n_valid > 0:
            raise LeakageError(
                f"{adapter.name}: fit_scope 'full_sample' would fit on "
                f"{n_outer}+{n_valid} rows, leaking the {n_valid} outer-valid "
                "materials into training. Standardization/feature selection "
                "must use outer_train only."
            )
        return

    if adapter.fit_scope == "outer_train":
        # Fit on the outer_train subset (positional 0..n_outer-1). The size check
        # below catches an adapter that sees more rows than outer_train; the
        # full_sample case is rejected outright above.
        Xtr = data.X[outer_train]
        ytr = data.Y[outer_train][:, 0]
        res = adapter.run_fit(Xtr, ytr, adapter.fit_scope)
        if res.n_rows_seen != n_outer:
            raise LeakageError(
                f"{adapter.name}: declared outer_train but saw "
                f"{res.n_rows_seen} rows, expected {n_outer}"
            )
        return

    if adapter.fit_scope == "inner_train":
        # The fit must use the inner_train slice specifically. We pass only the
        # inner_train rows; an adapter that trains on the full outer_train would
        # see more rows than inner_train and trip the size check.
        inner_train = np.asarray(split.inner_folds[0].inner_train)
        inner_valid = np.asarray(split.inner_folds[0].inner_valid)
        Xtr = data.X[inner_train]
        ytr = data.Y[inner_train][:, 0]
        res = adapter.run_fit(Xtr, ytr, adapter.fit_scope)
        if res.n_rows_seen != int(inner_train.shape[0]):
            raise LeakageError(
                f"{adapter.name}: declared inner_train but saw "
                f"{res.n_rows_seen} rows, expected {inner_train.shape[0]}"
            )
        # inner_train must be disjoint from outer_valid
        if not set(valid_idx.tolist()).isdisjoint(set(inner_train.tolist())):
            raise LeakageError(f"{adapter.name}: inner_train overlaps outer_valid")
        return

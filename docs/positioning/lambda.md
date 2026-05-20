# LAMBDA and ambiguity resolution

The LAMBDA algorithm (Least-squares AMBiguity Decorrelation Adjustment,
Teunissen 1995) is the standard way to fix the float carrier-phase
ambiguities of an RTK or PPP solution to their true integer values. With
the integers fixed, the baseline (RTK) or position (PPP) snaps to its
centimetre-class form.

rinexpy ships the single-frequency LAMBDA in `rinexpy.lambda_ar` and the
dual-frequency LAMBDA-style fix (via wide-lane / narrow-lane) in
`rinexpy.multifreq`.

## The algorithm

The float ambiguity vector `â` has a positive-definite covariance `Q`. The
integer ambiguity `n` minimises the quadratic form
`(â - n)' * Q^{-1} * (â - n)` over `n ∈ Z^k`.

The naive minimisation is exponential in `k` (the number of ambiguities)
because the search lattice has `O(eps^{-k})` candidates. LAMBDA reduces it
to polynomial by first decorrelating `Q` via an integer-valued Z-transform,
and then searching the decorrelated lattice with a branch-and-bound that
prunes by the running best squared residual.

The procedure inside `lambda_resolve`:

1. **LDL decomposition.** Factor `Q = L D L'`. The lower-triangular `L`
   captures the correlations; the diagonal `D` is the variance of each
   decorrelated ambiguity.

2. **Bootstrap.** Round `â` to integers in the decorrelated frame, one
   ambiguity at a time, applying each integer correction to the
   subsequent float ambiguities. The result is a candidate integer
   vector that is usually close to the true solution.

3. **Integer least squares.** A branch-and-bound search around the
   bootstrap finds the `n_cands` best integer vectors by squared
   residual.

4. **Ratio test.** Accept the best candidate if `sq_errors[1] /
   sq_errors[0] >= ratio_threshold`. The conservative default for the
   threshold is 3.0.

## Single-frequency LAMBDA

```python
import numpy as np
from rinexpy.lambda_ar import lambda_resolve

a_float = np.array([5.1, 10.4])
Q = np.array([[1.0, 0.5], [0.5, 1.0]])

out = lambda_resolve(a_float, Q, ratio_threshold=3.0)
print(out)
```

The return dict has:

| Key | Type | Meaning |
| --- | --- | --- |
| `a_int` | `ndarray` | best-integer candidate |
| `ratio` | `float` | `sq_errors[1] / sq_errors[0]` |
| `accepted` | `bool` | passed the ratio test |
| `candidates` | `ndarray` | the `n_cands` best candidates, shape `(n_cands, k)` |
| `sq_errors` | `ndarray` | squared residuals of the candidates |

## ILS internals

If you want to drive the search yourself, the building blocks are public.

```python
from rinexpy.lambda_ar import ldl, bootstrap, integer_least_squares

L, D = ldl(Q)                      # L D L' decomposition

a_int_bootstrap = bootstrap(L, a_float)

candidates, sq_errors, L_used = integer_least_squares(
    a_float, Q,
    n_cands=2,
    max_nodes=100_000,
    max_seconds=None,
)
```

`integer_least_squares` returns the `n_cands` best integer candidates as
an `(n_cands, n)` array, the matching squared residuals as `(n_cands,)`,
and the L factor from the internal LDL decomposition. It accepts two
exit limits: `max_nodes` caps the search-tree expansion and
`max_seconds` caps the wall-clock time. When either limit is hit, the
search raises `ILSAborted` carrying the best partial result.

```python
from rinexpy.lambda_ar import ILSAborted, integer_least_squares

try:
    candidates, sq_errors, _ = integer_least_squares(
        a_float, Q, n_cands=2, max_nodes=10,
    )
except ILSAborted as e:
    print("partial:", e.candidates)
    print("sq err: ", e.sq_errors)
```

For RTK and PPP applications the default `max_nodes` is high enough that
the limit is essentially never hit. For high-dimensional problems
(50+ ambiguities) the limit becomes load-bearing and tuning matters.

## Dual-frequency LAMBDA via WL+NL

For an L1+L2 receiver, fixing the L1 and L2 ambiguities directly is
needlessly hard because the LAMBDA covariance is `2k × 2k` (with `k`
satellites). The Wide-Lane / Narrow-Lane decomposition reduces it to two
`k × k` problems: fix the WL ambiguity first (it has a wavelength of 86
cm, so it is easy), then fix the NL ambiguity given the WL fix.

`rinexpy.multifreq.lambda_dual_freq` is the high-level entry point.

```python
import numpy as np
from rinexpy.multifreq import lambda_dual_freq

# Float ambiguities at L1 and L2 (in cycles, not metres).
a_l1_float = ...    # (n_sv,)
a_l2_float = ...    # (n_sv,)

out = lambda_dual_freq(
    a_l1_float, a_l2_float,
    cov_block=None,                 # joint covariance, optional
    p1_m=p1, p2_m=p2,                # raw pseudoranges in metres
    sigma_threshold=0.25,            # MW gate for WL fix
)
print("Fixed N1:", out["N_L1"])
print("Fixed N2:", out["N_L2"])
print("Fraction fixed:", out["fraction_fixed"])
```

The function uses the Melbourne-Wuebbena combination (geometry-free and
ionosphere-free) to fix WL ambiguities, then back-solves N1 and N2.
Satellites whose WL fix fails the `sigma_threshold` gate are left as
floats.

## Wide-lane resolution alone

```python
from rinexpy.multifreq import (
    melbourne_wubbena,
    resolve_wide_lane,
    split_wl_into_l1_l2,
)

mw = melbourne_wubbena(phi1_cycles, phi2_cycles, p1_m, p2_m)
# mw is the per-SV wide-lane ambiguity in cycles, plus noise

out = resolve_wide_lane(
    phi1_cycles, phi2_cycles, p1_m, p2_m,
    sigma_threshold=0.25,
)
print(out["N_WL"])           # rounded WL integer per SV
print(out["float_WL"])       # the float estimate per SV
print(out["fixed_mask"])     # bool per SV
print(out["fraction_fixed"]) # share of SVs that passed the gate
```

`split_wl_into_l1_l2` recovers `(N1, N2)` from `(N_WL, N_NL)`:

```python
n1, n2 = split_wl_into_l1_l2(n_wl, n_nl)
```

## Three-Carrier Ambiguity Resolution

For receivers tracking L1 + L2 + L5 (or E1 + E5a + E5b for Galileo), the
extra-wide-lane combination on L2 - L5 has a wavelength of nearly 6 metres.
The WL ambiguity is essentially noise-free, and constraints propagate
forward to fix N_WL and N_L1.

```python
from rinexpy.multifreq import (
    LAMBDA_EWL_25,
    extra_wide_lane_phase,
    melbourne_wubbena_ewl,
    resolve_extra_wide_lane,
    tcar_resolve,
)

# The full TCAR chain in one call:
out = tcar_resolve(
    phi1_cycles, phi2_cycles, phi5_cycles,
    p1_m, p2_m, p5_m,
)
print(out["N_EWL"])     # extra-wide-lane integers, almost always fixed
print(out["N_WL"])      # wide-lane integers
print(out["N_L1"])      # L1 integers
print(out["fixed_mask"])# per-SV bool
```

The fix chain is `EWL → WL → L1`, each step using the previous fix to
narrow the search space for the next.

## Constants exposed

The wavelength constants for both LAMBDA and downstream filtering:

| Constant | Value (m) | Meaning |
| --- | --- | --- |
| `LAMBDA_L1` | 0.19029367 | GPS L1 |
| `LAMBDA_L2` | 0.24421021 | GPS L2 |
| `LAMBDA_L5` | 0.25482804 | GPS L5 |
| `LAMBDA_WL` | 0.86191840 | wide-lane (L1 - L2) |
| `LAMBDA_NL` | 0.10695338 | narrow-lane (L1 + L2) |
| `LAMBDA_EWL_15` | 0.75148098 | L1 - L5 extra-wide-lane |
| `LAMBDA_EWL_25` | 5.86103024 | L2 - L5 extra-wide-lane |

Frequencies:

| Constant | Value (Hz) | Meaning |
| --- | --- | --- |
| `F1` | 1.57542e9 | GPS L1 |
| `F2` | 1.22760e9 | GPS L2 |
| `F5` | 1.17645e9 | GPS L5 |

## The ratio test

The conservative default `ratio_threshold=3.0` is the textbook Teunissen
threshold and is what real-time RTK receivers use. Lower thresholds (2.0)
accept more fixes but risk wrong-integer fixes; higher thresholds (5.0)
reject more fixes but rarely accept a wrong one.

The ratio is `sq_errors[1] / sq_errors[0]`: the ratio between the
second-best and the best squared residual. A ratio of 10 says the next
candidate is ten times worse, which is overwhelming evidence that the
best candidate is right. A ratio of 1.1 is borderline and usually
indicates the geometry is too weak for a fix.

When the ratio test fails, the float solution is still usable. The
SequentialRTK class runs partial AR on the most-precise subset when the
full fix fails.

## When LAMBDA struggles

A few common failure modes.

**Too few satellites.** With 4 satellites and a 4-unknown baseline, there
is one redundant satellite and no integer constraint to test. The fix is
formally well-posed but unreliable.

**Geometry collapse.** When all visible satellites are clustered in
azimuth (urban canyon, building on one side), the covariance matrix is
ill-conditioned and the LAMBDA decorrelation has nothing to work with.

**Unmodelled ionosphere.** Single-frequency RTK over a baseline of
20 km or more has a residual ionospheric delay that the DD does not
cancel. The float ambiguities become biased and the LAMBDA fix lands on
a wrong integer.

**Cycle slips just before the fix.** A slip that re-bootstraps an
ambiguity inflates that satellite's variance. If the LAMBDA call happens
on the next epoch the search is much wider and the ratio test fails.

The right response in production is to defer the fix one or two epochs
after a slip, or to run partial AR on the satellites that did not slip.
`SequentialRTK` does this internally.

## Related pages

- [RTK and integer fixing](rtk.md): the high-level user of `lambda_resolve`.
- [Precise point positioning](ppp.md): PPP-AR uses the WL+NL chain.
- [QC and cycle slips](../quality/qc.md): slip detectors feeding the AR loop.

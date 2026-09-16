"""
Uncertainty and sensitivity analysis for the recyclability indicator.
Reproduces the results reported in [paper citation / DOI].

Outputs (written to figures/):
  subweight_distributions.png
  mc_vs_equal_weights_violin.png
  pawn_ranking_heatmap_y1.png
  pawn_sensitivity_heatmap_y1.png
Usage:
  python recyclability_ua_sa.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import combinations
from sklearn.neighbors import KernelDensity

from SALib.analyze import pawn

# Reproducible random state (change to test sensitivity to seed choice)
rng = np.random.default_rng(42)

# Ensure output directory exists
os.makedirs('figures', exist_ok=True)

# Ensure all saved figures default to at least 300 DPI
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300

# ====
# 1. CONFIGURATION
# ====

CATEGORY_CONFIG = {
    'Generic': {
        'mix': {'Typical': 1000, 'Gold Standard': 250, 'Greenwasher': 250, 'Hidden Gem': 250, 'Low Performer': 250}
    },
}

N_WEIGHT_SAMPLES = 5000

# Top-layer priority ordering:
#   w5 > w4 > w1,  w5 > w4 > w3,  w5 > w3 > w1,  w3 > w2
def top_constraint(w):
    return ((w[:, 4] > w[:, 3]) & (w[:, 3] > w[:, 0]) &
            (w[:, 3] > w[:, 2]) & (w[:, 4] > w[:, 2]) &
            (w[:, 2] > w[:, 1]))

# Sub-layer A priority ordering:
#   w2 > w3 > w1,  w2 > w4 > w1
def a_constraint(w):
    return ((w[:, 1] > w[:, 2]) & (w[:, 1] > w[:, 3]) &
            (w[:, 2] > w[:, 0]) & (w[:, 3] > w[:, 0]))

SCALES = {
    'x1x4':   [1.0, 0.8, 0.6, 0.2, 0.0],
    'x5':     [1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
    'x6x7':   [1.0, 0.5, 0.0],
    'x8':     [1.0, 0.8, 0.6, 0.2, 0.0],
    'x9':     [1.0, 0.6, 0.2],
    'x10':    [1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
    'x11':    [1.0, 0.8, 0.6, 0.4, 0.0],
    'binary': [1.0, 0.0],
    'x13':    [1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
    'x15':    [1.0, 0.8, 0.6, 0.4, 0.0],
}

X_TO_SCALE = {
    'X1':'x1x4', 'X2':'x1x4', 'X3':'x1x4', 'X4':'x1x4',
    'X5':'x5',   'X6':'x6x7', 'X7':'x6x7',
    'X8':'x8',   'X9':'x9',   'X10':'x10',
    'X11':'x11', 'X12':'binary',
    'X13':'x13', 'X14':'binary', 'X15':'x15',
}

DOC_COMP = ['X1','X2','X3','X4','X5','X6','X7']
PHYSICAL = ['X8','X9','X10','X11','X12','X13','X14','X15']

HIGH_THRESH, LOW_THRESH = 0.6, 0.4

palette = {'Gold Standard':'gold', 'Greenwasher':'tomato',
           'Hidden Gem':'steelblue', 'Typical':'grey', 'Low Performer':'darkred'}
archetype_order = ['Gold Standard', 'Hidden Gem', 'Typical', 'Greenwasher', 'Low Performer']

weight_labels = {
    'wTop': ['w1', 'w2', 'w3', 'w4', 'w5'],
    'wA':   ['w6 (X1)', 'w7 (X2)', 'w8 (X3)', 'w9 (X4)'],
    'wB':   ['w10 (X5)', 'w11 (X6)', 'w12 (X7)'],
    'wC':   ['w13 (X8)', 'w14 (X9)', 'w15 (X10)', 'w16 (X11)', 'w17 (X12)'],
    'wD':   ['w18 (X13)', 'w19 (X14)', 'w20 (X15)'],
}

# ====
# 2. FUNCTIONS
# ====

def sample_sub_weights(n_features, n_samples=5000, constraint=None):
    """Sample Dirichlet weights with optional inequality constraints.

    Parameters
    ----------
    n_features : int
        Number of weights in this layer.
    n_samples : int
        Number of valid weight sets to return.
    constraint : callable or None
        A function that takes a (batch_size, n_features) array and returns
        a boolean mask of valid rows. None means no constraints (all rows
        accepted).

    Returns
    -------
    weights : ndarray, shape (n_samples, n_features)
    n_accepted : int   — total accepted before truncation to n_samples
    n_discarded : int  — total rejected by the constraint
    """
    valid, discarded = [], 0
    batch_size = max(10_000, n_samples * 10)
    while len(valid) < n_samples:
        batch = rng.dirichlet(np.ones(n_features), size=batch_size)
        if constraint is not None:
            mask = constraint(batch)
            discarded += int((~mask).sum())
            valid.extend(batch[mask][:n_samples - len(valid)])
        else:
            valid.extend(batch[:n_samples - len(valid)])
    return np.array(valid[:n_samples]), len(valid), discarded


def threshold_sample(opts, direction, n):
    opts = np.array(opts)
    filtered = opts[opts >= HIGH_THRESH] if direction == 'high' else opts[opts <= LOW_THRESH]
    if len(filtered) == 0:
        filtered = opts
    return rng.choice(filtered, n)


def _generate_single(n, category, archetype_type):
    data = {}
    for i in range(1, 16):
        xname = f'X{i}'
        opts  = SCALES[X_TO_SCALE[xname]]
        is_doc = xname in DOC_COMP
        if archetype_type == 'Gold Standard':
            data[xname] = threshold_sample(opts, 'high', n)
        elif archetype_type == 'Greenwasher':
            data[xname] = threshold_sample(opts, 'high' if is_doc else 'low', n)
        elif archetype_type == 'Hidden Gem':
            data[xname] = threshold_sample(opts, 'low' if is_doc else 'high', n)
        elif archetype_type == 'Low Performer':
            data[xname] = threshold_sample(opts, 'low', n)
        else:
            data[xname] = rng.choice(opts, n)
    x16_map = {
        'Gold Standard': (0.61, 1.0),
        'Greenwasher':   (0.0, 0.49),
        'Hidden Gem':    (0.5, 1.0),
        'Low Performer': (0.0, 0.49),
        'Typical':       (0.0, 1.0),
    }
    lo, hi = x16_map[archetype_type]
    x16_raw = rng.uniform(lo, hi, n)
    # Discrete scoring scale for recycling rate of priority part (X16),
    # normalised from the original 0/1/3/5-point scale to a 0–1 scale:
    #   0 pts (0.0)  -> cannot be recycled (raw == 0)
    #   1 pt  (0.2)  -> recyclability rate lower than 0.5
    #   3 pts (0.6)  -> recyclability rate between 0.5 and 0.6
    #   5 pts (1.0)  -> recyclability rate higher than 0.6
    x16_scored = np.where(
        x16_raw <= 0.0, 0.0,
        np.where(x16_raw < 0.5, 0.2, np.where(x16_raw <= 0.6, 0.6, 1.0))
    )
    data['X16'] = x16_scored
    data['Category']  = category
    data['Archetype'] = archetype_type
    return pd.DataFrame(data)


def generate_product_data(category='Generic'):
    cfg = CATEGORY_CONFIG[category]
    frames = [_generate_single(count, category, arch)
              for arch, count in cfg['mix'].items()]
    return pd.concat(frames, ignore_index=True)


def calculate_scores_full(df, wTop, wA, wB, wC, wD):
    """Scenario A: fully linear scoring."""
    A = wA @ df[['X1','X2','X3','X4']].values.T
    B = wB @ df[['X5','X6','X7']].values.T
    C = wC @ df[['X8','X9','X10','X11','X12']].values.T
    D = wD @ df[['X13','X14','X15']].values.T
    E = np.tile(df['X16'].values, (wTop.shape[0], 1))
    return (wTop[:,0:1]*A + wTop[:,1:2]*B +
            wTop[:,2:3]*C + wTop[:,3:4]*D + wTop[:,4:5]*E)


def excellence_transform(x, k=5.0):
    """Abrupt (excellence) scale: sigmoid-like transform centred at 0.5.
    Values near 1 are boosted; values near 0 are suppressed."""
    return 1.0 / (1.0 + np.exp(-k * (x - 0.5)))


def calculate_scores_scenarioB(df, wTop, wA, wB, wC, wD):
    """Scenario B: excellence scale applied to ALL sub-indicators."""
    def ex(arr): return excellence_transform(arr)
    A = wA @ ex(df[['X1','X2','X3','X4']].values).T
    B = wB @ ex(df[['X5','X6','X7']].values).T
    C = wC @ ex(df[['X8','X9','X10','X11','X12']].values).T
    D = wD @ ex(df[['X13','X14','X15']].values).T
    E = np.tile(ex(df['X16'].values), (wTop.shape[0], 1))
    return (wTop[:,0:1]*A + wTop[:,1:2]*B +
            wTop[:,2:3]*C + wTop[:,3:4]*D + wTop[:,4:5]*E)


def calculate_scores_scenarioC(df, wTop, wA, wB, wC, wD):
    """Scenario C (Hybrid): linear for A/B (documentation layers),
    excellence scale for C/D/E (physical/technical layers)."""
    def ex(arr): return excellence_transform(arr)
    A = wA @ df[['X1','X2','X3','X4']].values.T
    B = wB @ df[['X5','X6','X7']].values.T
    C = wC @ ex(df[['X8','X9','X10','X11','X12']].values).T
    D = wD @ ex(df[['X13','X14','X15']].values).T
    E = np.tile(ex(df['X16'].values), (wTop.shape[0], 1))
    return (wTop[:,0:1]*A + wTop[:,1:2]*B +
            wTop[:,2:3]*C + wTop[:,3:4]*D + wTop[:,4:5]*E)


def variance_decomposition(scores_arch):
    mean_per_scenario = scores_arch.mean(axis=1)
    var_per_scenario  = scores_arch.var(axis=1)
    var_product = var_per_scenario.mean()
    var_weight  = mean_per_scenario.var()
    var_total   = var_product + var_weight
    pct_product = 100 * var_product / var_total if var_total > 0 else 0
    pct_weight  = 100 * var_weight  / var_total if var_total > 0 else 0
    return {
        'var_total':   var_total,
        'var_weight':  var_weight,
        'var_product': var_product,
        'pct_weight':  pct_weight,
        'pct_product': pct_product,
        'mean_score':  scores_arch.mean(),
        'std_total':   scores_arch.std(),
    }


def overlap_index(s1, s2, n_points=500):
    x = np.linspace(0, 1, n_points).reshape(-1, 1)
    kde1 = KernelDensity(bandwidth=0.03).fit(s1.reshape(-1, 1))
    kde2 = KernelDensity(bandwidth=0.03).fit(s2.reshape(-1, 1))
    p1 = np.exp(kde1.score_samples(x)); p2 = np.exp(kde2.score_samples(x))
    p1 /= p1.sum(); p2 /= p2.sum()
    return np.minimum(p1, p2).sum()


# ====
# 3. SAMPLE WEIGHTS
# ====
print("Sampling weights...")
wTop, wTop_accepted, wTop_discarded = sample_sub_weights(5, n_samples=N_WEIGHT_SAMPLES, constraint=top_constraint)
wA,   wA_accepted,   wA_discarded   = sample_sub_weights(4, n_samples=N_WEIGHT_SAMPLES, constraint=a_constraint)
wB,   wB_accepted,   wB_discarded   = sample_sub_weights(3, n_samples=N_WEIGHT_SAMPLES)
wC,   wC_accepted,   wC_discarded   = sample_sub_weights(5, n_samples=N_WEIGHT_SAMPLES)
wD,   wD_accepted,   wD_discarded   = sample_sub_weights(3, n_samples=N_WEIGHT_SAMPLES)

all_weights = np.hstack([wTop, wA, wB, wC, wD])
all_labels  = (weight_labels['wTop'] + weight_labels['wA'] +
               weight_labels['wB']   + weight_labels['wC'] + weight_labels['wD'])

# Print weight sampling statistics
print("\n" + "="*70)
print("WEIGHT SAMPLING STATISTICS")
print("="*70)
weight_stats = [
    ('Top sub-indicator (w1-w5)', wTop_accepted, wTop_discarded, True),
    ('Sub-indicator A (w6-w9)',   wA_accepted,   wA_discarded,   True),
    ('Sub-indicator B (w10-w12)', wB_accepted,   wB_discarded,   False),
    ('Sub-indicator C (w13-w17)', wC_accepted,   wC_discarded,   False),
    ('Sub-indicator D (w18-w20)', wD_accepted,   wD_discarded,   False),
]

total_accepted = total_discarded = 0
for layer_name, accepted, discarded, has_constraints in weight_stats:
    total_accepted  += accepted
    total_discarded += discarded
    acceptance_rate  = 100 * accepted / (accepted + discarded) if (accepted + discarded) > 0 else 100
    constraint_str   = "(constrained)" if has_constraints else "(no constraints)"
    print(f"{layer_name:<25} | Accepted: {accepted:5d} | Discarded: {discarded:7d} | "
          f"Acceptance Rate: {acceptance_rate:6.2f}% {constraint_str}")

print("-" * 70)
print(f"{'TOTAL':<25} | Accepted: {total_accepted:5d} | Discarded: {total_discarded:7d} | "
      f"Acceptance Rate: {100 * total_accepted / (total_accepted + total_discarded):6.2f}%")
print("="*70 + "\n")

# ====
# 3a-PLOT. SUB-WEIGHT DISTRIBUTIONS (Dirichlet-sampled, constrained) — VIOLIN PLOTS
# ====
print("\nPlotting sub-weight distributions (violin plots)...")

layer_info = {
    'Top sub-indicator':  (wTop, ['w1', 'w2', 'w3', 'w4', 'w5'],              '#2c7bb6'),
    'Sub-indicator A':    (wA,   ['w6', 'w7', 'w8', 'w9'],                    '#d7191c'),
    'Sub-indicator B':    (wB,   ['w10', 'w11', 'w12'],                        '#1a9641'),
    'Sub-indicator C':    (wC,   ['w13', 'w14', 'w15', 'w16', 'w17'],         '#fdae61'),
    'Sub-indicator D':    (wD,   ['w18', 'w19', 'w20'],                        '#762a83'),
}

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 6)
ax_top1 = fig.add_subplot(gs[0, 0:3])
ax_top2 = fig.add_subplot(gs[0, 3:6])
ax_bot1 = fig.add_subplot(gs[1, 0:2])
ax_bot2 = fig.add_subplot(gs[1, 2:4])
ax_bot3 = fig.add_subplot(gs[1, 4:6])
axes = [ax_top1, ax_top2, ax_bot1, ax_bot2, ax_bot3]

for i, (ax, (layer_name, (W, labels, col))) in enumerate(zip(axes, layer_info.items())):
    parts = ax.violinplot(
        [W[:, i2] for i2 in range(W.shape[1])],
        positions=range(len(labels)), showmedians=True, showextrema=True
    )
    for pc in parts['bodies']:
        pc.set_facecolor(col)
        pc.set_alpha(0.6)
    for key in ['cmedians', 'cbars', 'cmins', 'cmaxes']:
        parts[key].set_color('black')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.set_title(layer_name, fontweight='bold', fontsize=10)
    ax.grid(True, alpha=0.3)
    if i in (0, 2):
        ax.set_ylabel('Weighting factor value', fontsize=9)

plt.tight_layout()
plt.savefig('figures/subweight_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# Print median/std summary for the sub-weight distributions
print("\n" + "="*70)
print("SUB-WEIGHT DISTRIBUTION SUMMARY (Median / StdDev)")
print("="*70)
for layer_name, (W, labels, col) in layer_info.items():
    print(f"\n{layer_name.splitlines()[0]}:")
    for j, lbl in enumerate(labels):
        print(f"  {lbl:<14} median={np.median(W[:, j]):.3f}  std={np.std(W[:, j]):.3f}")
print("="*70 + "\n")

# ====
# 3b. GENERATE SYNTHETIC DATA
# ====
print("Generating synthetic product data...")
product_data  = {cat: generate_product_data(category=cat) for cat in CATEGORY_CONFIG}
scores_by_cat = {cat: calculate_scores_full(product_data[cat], wTop, wA, wB, wC, wD)
                 for cat in CATEGORY_CONFIG}
scores_by_cat_B = {cat: calculate_scores_scenarioB(product_data[cat], wTop, wA, wB, wC, wD)
                   for cat in CATEGORY_CONFIG}
scores_by_cat_C = {cat: calculate_scores_scenarioC(product_data[cat], wTop, wA, wB, wC, wD)
                   for cat in CATEGORY_CONFIG}

# ====
# 3b-EQ. EQUAL-WEIGHT BENCHMARK
# ====
wTop_eq = np.array([[1/5] * 5])
wA_eq   = np.array([[1/4] * 4])
wB_eq   = np.array([[1/3] * 3])
wC_eq   = np.array([[1/5] * 5])
wD_eq   = np.array([[1/3] * 3])

scores_by_cat_equal = {
    cat: calculate_scores_full(product_data[cat], wTop_eq, wA_eq, wB_eq, wC_eq, wD_eq)
    for cat in CATEGORY_CONFIG
}

# ====
# 3b-CONV. WEIGHT CONVERGENCE TEST (PER ARCHETYPE)
# ====
print("\n" + "="*70)
print("WEIGHT CONVERGENCE ANALYSIS (PER ARCHETYPE)")
print("="*70)

weight_convergence_checkpoints = [200, 500, 1000, 2000, 3000, 4000, 5000]
weight_convergence_checkpoints = [c for c in weight_convergence_checkpoints if c <= N_WEIGHT_SAMPLES]
if weight_convergence_checkpoints[-1] != N_WEIGHT_SAMPLES:
    weight_convergence_checkpoints.append(N_WEIGHT_SAMPLES)

df_ref = product_data['Generic']
wconv_rows = []

for checkpoint in weight_convergence_checkpoints:
    scores_s = calculate_scores_full(df_ref, wTop[:checkpoint], wA[:checkpoint],
                                     wB[:checkpoint], wC[:checkpoint], wD[:checkpoint])
    row = {'N_weights': checkpoint}
    for arch in archetype_order:
        mask = df_ref['Archetype'].values == arch
        arch_scores = scores_s[:, mask]
        row[f'{arch}_mean'] = arch_scores.mean(axis=1).mean()
        row[f'{arch}_std']  = arch_scores.mean(axis=1).std()
    wconv_rows.append(row)

wconv_df = pd.DataFrame(wconv_rows)

col_w = 14
header_line = "-" * (12 + col_w * len(archetype_order))

print("\nMean Score per Archetype vs. N Weight Scenarios:")
print(header_line)
print(f"{'N_weights':<12}" + "".join(f"{a[:10]:<{col_w}}" for a in archetype_order))
print(header_line)
for _, r in wconv_df.iterrows():
    print(f"{int(r['N_weights']):<12}" + "".join(f"{r[f'{a}_mean']:<{col_w}.4f}" for a in archetype_order))
print(header_line)

print("\nStd of Per-Scenario Archetype Means vs. N Weight Scenarios:")
print(header_line)
print(f"{'N_weights':<12}" + "".join(f"{a[:10]:<{col_w}}" for a in archetype_order))
print(header_line)
for _, r in wconv_df.iterrows():
    print(f"{int(r['N_weights']):<12}" + "".join(f"{r[f'{a}_std']:<{col_w}.4f}" for a in archetype_order))
print(header_line)

print("\nRelative Change in Mean Score (%) per Archetype vs. Previous Checkpoint:")
print(header_line)
print(f"{'N_weights':<12}" + "".join(f"{a[:10]:<{col_w}}" for a in archetype_order))
print(header_line)
for i, (_, r) in enumerate(wconv_df.iterrows()):
    if i == 0:
        print(f"{int(r['N_weights']):<12}" + "".join(f"{'baseline':<{col_w}}" for _ in archetype_order))
    else:
        prev = wconv_df.iloc[i - 1]
        cells = []
        for arch in archetype_order:
            delta = 100 * abs(r[f'{arch}_mean'] - prev[f'{arch}_mean']) / (abs(prev[f'{arch}_mean']) + 1e-10)
            flag = "✓" if delta < 1.0 else " "
            cells.append(f"{flag}{delta:.3f}%".ljust(col_w))
        print(f"{int(r['N_weights']):<12}" + "".join(cells))
print(header_line)
print("  ✓ = relative change < 1% (converged)")

# ====
# 3c. MONTE CARLO 95% CONFIDENCE INTERVALS
# ====
print("\n" + "="*70)
print("MONTE CARLO SCORE DISTRIBUTIONS — 95% CONFIDENCE INTERVALS (Scenario A)")
print("="*70)
print(f"\n{'Archetype':<16} {'Median':>8} {'Mean':>8} {'StdDev':>8} {'P2.5':>8} {'P97.5':>8} {'95% CI Width':>14}")
print("-" * 70)

df_ci     = product_data["Generic"]
scores_ci = scores_by_cat["Generic"]

ci_rows = []
for arch in archetype_order:
    mask      = df_ci['Archetype'].values == arch
    arch_scores = scores_ci[:, mask].flatten()
    p2_5  = np.percentile(arch_scores, 2.5)
    p50   = np.percentile(arch_scores, 50.0)
    p97_5 = np.percentile(arch_scores, 97.5)
    mean  = arch_scores.mean()
    std   = arch_scores.std()
    width = p97_5 - p2_5
    print(f"{arch:<16} {p50:>8.3f} {mean:>8.3f} {std:>8.3f} {p2_5:>8.3f} {p97_5:>8.3f} {width:>14.3f}")
    ci_rows.append({'Archetype': arch, 'Median': p50, 'Mean': mean, 'StdDev': std,
                    'P2.5': p2_5, 'P97.5': p97_5, 'CI_Width': width})

ci_df = pd.DataFrame(ci_rows)
print("\nPresentation-ready format:")
print("-" * 70)
for _, row in ci_df.iterrows():
    print(f"  {row['Archetype']:<16}: {row['Median']:.2f}  (95% CI: {row['P2.5']:.2f} – {row['P97.5']:.2f})")
print("="*70 + "\n")

# ====
# 3c-CMP. STANDARD DEVIATION COMPARISON: Priority (constrained) weights vs Equal weights
# ====
print("\n" + "="*70)
print("STANDARD DEVIATION COMPARISON — PRIORITY (CONSTRAINED) vs EQUAL WEIGHTS")
print("="*70)

df_cmp          = product_data["Generic"]
scores_prio_cmp = scores_by_cat["Generic"]
scores_eq_cmp   = scores_by_cat_equal["Generic"]

print(f"\n{'Archetype':<16} {'Mean(prio)':>12} {'Std(prio)':>12} "
      f"{'Mean(eq)':>12} {'Std(eq)':>12} {'Std reduction':>15}")
print("-" * 85)

stddev_cmp_rows = []
for arch in archetype_order:
    mask      = df_cmp['Archetype'].values == arch
    prio_vals = scores_prio_cmp[:, mask].flatten()
    eq_vals   = scores_eq_cmp[:, mask].flatten()
    mean_prio, std_prio = prio_vals.mean(), prio_vals.std()
    mean_eq,   std_eq   = eq_vals.mean(),   eq_vals.std()
    std_reduction = 100 * (std_prio - std_eq) / std_prio if std_prio > 0 else 0
    print(f"{arch:<16} {mean_prio:>12.4f} {std_prio:>12.4f} "
          f"{mean_eq:>12.4f} {std_eq:>12.4f} {std_reduction:>14.2f}%")
    stddev_cmp_rows.append({
        'Archetype': arch,
        'Mean_priority': mean_prio, 'Std_priority': std_prio,
        'Mean_equal': mean_eq, 'Std_equal': std_eq,
        'Std_reduction_pct': std_reduction,
    })

stddev_cmp_df = pd.DataFrame(stddev_cmp_rows)
print("-" * 85)
print("Std reduction (%) = how much score variability shrinks when moving from the\n"
      "constrained-priority weighting scheme to a fixed equal-weight scheme.\n"
      "Positive values indicate that priority (constrained) weighting introduces\n"
      "additional uncertainty relative to the equal-weight benchmark.")
print("="*70 + "\n")

# ====
# 3d. VIOLIN PLOT: Monte Carlo (priority weights) vs Equal Weights
# ====
print("\nPlotting MC vs Equal Weights violin plot...")

df_viol   = product_data["Generic"]
scores_mc = scores_by_cat["Generic"]
scores_eq = scores_by_cat_equal["Generic"]

violin_rows = []
for arch in archetype_order:
    mask = df_viol["Archetype"].values == arch
    for score_val in scores_mc[:, mask].flatten():
        violin_rows.append({'Archetype': arch, 'Score': score_val, 'Weighting': 'Priority-constrained weights'})
    for score_val in scores_eq[:, mask].flatten():
        violin_rows.append({'Archetype': arch, 'Score': score_val, 'Weighting': 'Equal weights'})

violin_df = pd.DataFrame(violin_rows)

fig, ax = plt.subplots(figsize=(14, 6))
sns.violinplot(
    data=violin_df,
    x='Archetype', y='Score',
    hue='Weighting',
    order=archetype_order,
    hue_order=['Priority-constrained weights', 'Equal weights'],
    palette={'Priority-constrained weights': 'steelblue', 'Equal weights': 'lightgrey'},
    split=True, inner='quartile', linewidth=0.8, ax=ax
)
ax.set_ylim(0, 1)
ax.set_xlabel('')
ax.set_ylabel('Recyclability score', fontsize=10)
ax.legend(title='Weighting scheme', fontsize=9, title_fontsize=9)
ax.grid(True, axis='y', alpha=0.25)
for tick, arch in zip(ax.get_xticklabels(), archetype_order):
    tick.set_color("black")
    tick.set_fontweight('bold')
plt.tight_layout()
plt.savefig('figures/mc_vs_equal_weights_violin.png', dpi=300, bbox_inches='tight')
plt.show()

# ====
# PAWN GLOBAL SENSITIVITY ANALYSIS
# Two Y targets per archetype:
#   - Mean   (location sensitivity):       which weights shift the average score?
#   - StdDev (discrimination sensitivity): which weights affect score spread?
#
# Note on input distribution: PAWN is formally derived for uniformly distributed
# inputs. Here the X matrix contains constrained-Dirichlet weight samples, whose
# marginal distributions are non-uniform. The observable bounds are tightened to
# the sampled range (see below) to improve conditioning-interval coverage, but
# sparse tails in some intervals may still inflate variance of the KS statistic.
# Indices should therefore be interpreted as ordinal rankings rather than exact
# magnitudes.
# ====
print("\n" + "="*70)
print("PAWN GLOBAL SENSITIVITY ANALYSIS (Mean & StdDev)")
print("="*70)

X_pawn = all_weights

# Tighten bounds to observed range of sampled weights (improves PAWN conditioning intervals)
pawn_bounds = []
for i in range(X_pawn.shape[1]):
    lo = float(np.min(X_pawn[:, i]))
    hi = float(np.max(X_pawn[:, i]))
    if hi <= lo:
        hi = lo + 1e-12
    pawn_bounds.append([lo, hi])

pawn_problem = {
    'num_vars': 20,
    'names': all_labels,
    'bounds': pawn_bounds
}

df_cat_pawn = product_data["Generic"]
scores_pawn = scores_by_cat["Generic"]   # Scenario A (linear)

pawn_rows = []
for arch in archetype_order:
    mask        = df_cat_pawn['Archetype'].values == arch
    arch_scores = scores_pawn[:, mask]
    Y_mean = arch_scores.mean(axis=1)
    Y_std  = arch_scores.std(axis=1)
    for metric_name, Y in [('Mean', Y_mean), ('StdDev', Y_std)]:
        Si = pawn.analyze(pawn_problem, X_pawn, Y, S=10, print_to_console=False)
        for j, label in enumerate(all_labels):
            pawn_rows.append({
                'Archetype':   arch,
                'Metric':      metric_name,
                'Weight':      label,
                'PAWN_median': Si['median'][j],
                'PAWN_mean':   Si['mean'][j],
                'PAWN_max':    Si['maximum'][j],
                'CV':          Si['CV'][j],
            })

pawn_df = pd.DataFrame(pawn_rows)

# ====
# WEIGHT CONTRIBUTION RANKING (PAWN)
# ====
print("\n" + "="*70)
print("WEIGHT CONTRIBUTION RANKING (PAWN)")
print("="*70)

ranking_rows = []
for arch in archetype_order:
    for metric in ['Mean', 'StdDev']:
        sub = (
            pawn_df[(pawn_df['Archetype'] == arch) & (pawn_df['Metric'] == metric)]
            .sort_values('PAWN_median', ascending=False)
            .reset_index(drop=True)
        )
        sub['Rank'] = np.arange(1, len(sub) + 1)
        print(f"\n{arch} — {metric} sensitivity")
        print("-" * 60)
        for _, row in sub.iterrows():
            print(f"#{int(row['Rank']):<2} {row['Weight']:<18} PAWN={row['PAWN_median']:.3f}")
            ranking_rows.append({
                'Archetype':   arch,
                'Metric':      metric,
                'Rank':        int(row['Rank']),
                'Weight':      row['Weight'],
                'PAWN_median': row['PAWN_median'],
            })

ranking_df = pd.DataFrame(ranking_rows)

# Sequential weight order w1→w20 (reversed for bottom-to-top y-axis in heatmaps)
w_order = (weight_labels['wTop'] + weight_labels['wA'] +
           weight_labels['wB']   + weight_labels['wC'] + weight_labels['wD'])[::-1]

# Consistent row order across PAWN plots (overall PAWN_median magnitude)
ref_pawn_order = (
    pawn_df.groupby('Weight')['PAWN_median']
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

# ====
# PLOT PAWN 0b: Rank heatmap — Y1 (Mean) only
# ====
print("\nPlotting PAWN Y1 (Mean) ranking heatmap...")

pivot_val_y1 = (
    pawn_df[pawn_df['Metric'] == 'Mean']
    .pivot(index='Weight', columns='Archetype', values='PAWN_median')
    [archetype_order].reindex(w_order)
)
pivot_rank_y1 = pivot_val_y1.fillna(-np.inf).rank(axis=0, ascending=False, method='first').astype(int)

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(
    pivot_rank_y1, ax=ax,
    cmap='RdYlGn_r', vmin=1, vmax=len(all_labels),
    annot=True, fmt='d', annot_kws={'size': 9}, linewidths=0.4,
    cbar_kws={'label': 'Rank (1 = highest contribution)'}
)
ax.tick_params(axis='x', rotation=15)
ax.tick_params(axis='y', labelsize=9)
ax.set_xlabel('')
ax.set_ylabel('Weighting factors')
plt.tight_layout()
plt.savefig('figures/pawn_ranking_heatmap_y1.png', dpi=300, bbox_inches='tight')
plt.show()

# ====
# PLOT PAWN 0c: PAWN median value heatmap — Y1 (Mean) only
# ====
print("\nPlotting PAWN Y1 (Mean) median value heatmap...")

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(
    pivot_val_y1, ax=ax,
    cmap='YlOrRd', vmin=0, vmax=0.6,
    annot=True, fmt='.2f', annot_kws={'size': 9}, linewidths=0.4,
    cbar_kws={'label': 'PAWN method median index'}
)
ax.tick_params(axis='x', rotation=15)
ax.tick_params(axis='y', labelsize=9)
ax.set_xlabel('')
ax.set_ylabel('Weighting factors')
plt.tight_layout()
plt.savefig('figures/pawn_sensitivity_heatmap_y1.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n{'Archetype':<16} {'Metric':<8} {'Weight':<16} "
      f"{'PAWN_median':>12} {'PAWN_mean':>10} {'PAWN_max':>10} {'CV':>8}")
print("-" * 76)
for _, row in pawn_df.iterrows():
    print(f"{row['Archetype']:<16} {row['Metric']:<8} {row['Weight']:<16} "
          f"{row['PAWN_median']:>12.3f} {row['PAWN_mean']:>10.3f} "
          f"{row['PAWN_max']:>10.3f} {row['CV']:>8.3f}")

print("\nAll done!")

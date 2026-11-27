# Generate all publication PDF figures (Figs. 3-9) from plot_data.json.
import numpy as np
import os
import sys
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PLOT_JSON = os.path.join(_PROJECT_ROOT, 'plot_data.json')
OUTPUT_DIR = _PROJECT_ROOT

PLOT_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
}


def save_figure(fig, filename):
    fig.savefig(filename, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"  [Plot] {filename}")


def fmt_num(v):
    f = float(v)
    return str(int(f)) if f == int(f) else str(f)


def _lookup_float_key(mapping, target):
    if not isinstance(mapping, dict):
        return None
    for k, v in mapping.items():
        try:
            if abs(float(k) - float(target)) <= 1e-12:
                return v
        except (TypeError, ValueError):
            continue
    return None


# Plot total cost with +/-1 std band against a sensitivity multiplier (Figs. 3 & 4).
def plot_sensitivity_line(gammas, total_means, total_stds,
                          xlabel, title, filename, logy=False):
    valid = [(g, m, 0.0 if s is None else s)
             for g, m, s in zip(gammas, total_means, total_stds)
             if g is not None and m is not None]
    if not valid:
        print(f"  [SKIP] Empty data for {filename}")
        return

    fig, ax = plt.subplots(figsize=(5.5, 4))

    gammas = np.array([v[0] for v in valid], dtype=float)
    means_k = np.array([v[1] for v in valid], dtype=float) / 1000
    stds_k = np.array([v[2] for v in valid], dtype=float) / 1000

    ax.plot(gammas, means_k, 'o-', color='#2166AC', linewidth=2, markersize=7,
            markerfacecolor='white', markeredgewidth=2, markeredgecolor='#2166AC',
            label='Total Cost', zorder=3)

    if np.any(stds_k > 0):
        lower = np.clip(means_k - stds_k, 1e-9, None) if logy else means_k - stds_k
        ax.fill_between(gammas, lower, means_k + stds_k,
                         color='#2166AC', alpha=0.15, label='$\\pm 1$ std')

    for g, m in zip(gammas, means_k):
        if m == means_k.max():
            dy, va = -12, 'top'
        else:
            dy, va = 10, 'bottom'
        ax.annotate(f'{m:.1f}', xy=(g, m),
                    xytext=(0, dy), textcoords='offset points',
                    ha='center', va=va, fontsize=9, fontweight='bold')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Total Cost (\\$k)')
    ax.set_title(title)
    ax.legend(loc='upper left', framealpha=0.9)

    if logy:
        ax.set_yscale('log')
        ymin, ymax = means_k.min(), means_k.max()
        ax.set_ylim(ymin / 1.15, ymax * 1.35)
    else:
        span = means_k.max() - means_k.min()
        y_pad = span * 0.2 if span > 0 else max(abs(means_k.max()) * 0.1, 1.0)
        ax.set_ylim(means_k.min() - y_pad, means_k.max() + y_pad * 2.2)

    fig.tight_layout()
    save_figure(fig, filename)


# Plot expected recourse cost vs. CVaR_0.95 across lambda (Fig. 5a).
def plot_risk_pareto(lambda_vals, expected_rec_means, cvar_means, filename):
    points = sorted(
        [(l, r / 1000, c / 1000)
         for l, r, c in zip(lambda_vals, expected_rec_means, cvar_means)
         if l is not None and r is not None and c is not None],
        key=lambda t: float(t[0]))
    if not points:
        print(f"  [SKIP] Empty data for {filename}")
        return

    fig, ax = plt.subplots(figsize=(5.5, 4))

    lambda_vals = [p[0] for p in points]
    rec_k = [p[1] for p in points]
    cvar_k = [p[2] for p in points]

    ax.plot(rec_k, cvar_k, 'o-', color='#B2182B', markersize=8, linewidth=2,
            markerfacecolor='white', markeredgewidth=2, markeredgecolor='#B2182B',
            zorder=3)

    label_box = dict(boxstyle='round,pad=0.15', facecolor='white',
                     edgecolor='none', alpha=0.85)
    ax.annotate(f'$\\lambda$={fmt_num(lambda_vals[0])}',
                xy=(rec_k[0], cvar_k[0]), xytext=(-8, -8),
                textcoords='offset points', ha='right', va='top',
                fontsize=8, bbox=label_box, zorder=4)
    if len(lambda_vals) > 1:
        ax.annotate(f'$\\lambda$={fmt_num(lambda_vals[-1])}',
                    xy=(rec_k[-1], cvar_k[-1]), xytext=(8, -8),
                    textcoords='offset points', ha='left', va='top',
                    fontsize=8, bbox=label_box, zorder=4)

        arrow_start = 1 if len(lambda_vals) > 2 else 0
        arrow_end = min(arrow_start + 1, len(lambda_vals) - 1)
        ax.annotate('', xy=(rec_k[arrow_end], cvar_k[arrow_end]),
                    xytext=(rec_k[arrow_start], cvar_k[arrow_start]),
                    arrowprops=dict(arrowstyle='-|>', color='#7f1d2d', lw=1.2,
                                    mutation_scale=11, shrinkA=8, shrinkB=8),
                    zorder=4)
        ax.text(0.60, 0.42, 'Increasing $\\lambda$', transform=ax.transAxes,
                fontsize=8, color='0.35', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='none', alpha=0.85))

    ax.set_xlabel('Expected Recourse Cost (\\$k)')
    ax.set_ylabel('CVaR$_{0.95}$ (\\$k)')
    ax.set_title('Pareto Frontier: Expected Cost vs. Tail Risk')
    ax.margins(x=0.10, y=0.10)

    fig.tight_layout()
    save_figure(fig, filename)


# Plot upper-tail exceedance probability of per-scenario recourse costs (Fig. 5b).
def plot_risk_tail_ecdf(lambda_vals, scenario_costs_by_lambda, filename):
    series = [(lam, costs) for lam, costs in zip(lambda_vals, scenario_costs_by_lambda)
              if costs]
    if not series:
        print(f"  [SKIP] Empty data for {filename}")
        return

    fig, ax = plt.subplots(figsize=(5.5, 4))

    colors = plt.cm.RdYlBu(np.linspace(0.05, 0.95, len(series)))

    for i, (lam, costs) in enumerate(series):
        costs_arr = np.sort(np.array(costs, dtype=float))
        n = len(costs_arr)
        exceedance_prob = 1.0 - np.arange(1, n + 1) / n
        costs_k = costs_arr / 1000

        ax.step(costs_k, exceedance_prob, where='post',
                color=colors[i], linewidth=1.8,
                label=f'$\\lambda$={fmt_num(lam)}')

    ax.axhline(y=0.05, color='gray', linestyle=':', linewidth=1.0, alpha=0.7)
    ax.text(ax.get_xlim()[0] + 0.28 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
            0.055, '$\\alpha = 0.95$ (CVaR threshold)',
            fontsize=8, color='gray', va='bottom', ha='left')

    ax.set_ylim(-0.005, 0.20)
    ax.set_xlabel('Recourse Cost (\\$k)')
    ax.set_ylabel('Exceedance Probability $P(X > x)$')
    ax.set_title('Tail Distribution of Recourse Costs')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    fig.tight_layout()
    save_figure(fig, filename)


# Plot capital vs. recourse cost and GPU tier composition across lambda (Fig. 6).
def plot_risk_decisions(lambda_vals, tier_means,
                        capital_means, expected_rec_means, filename):
    valid_idx = [i for i, (l, c, r) in enumerate(zip(lambda_vals, capital_means, expected_rec_means))
                 if l is not None and c is not None and r is not None]
    if not valid_idx:
        print(f"  [SKIP] Empty data for {filename}")
        return

    lambda_vals = [lambda_vals[i] for i in valid_idx]
    cap_k = [capital_means[i] / 1000 for i in valid_idx]
    rec_k = [expected_rec_means[i] / 1000 for i in valid_idx]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    ax1 = axes[0]
    color1, color2 = '#2166AC', '#B2182B'
    ax1.plot(lambda_vals, cap_k, 'o-', color=color1, linewidth=2, markersize=6,
             label='Capital Cost')
    ax1.set_xlabel('$\\lambda$')
    ax1.set_ylabel('Capital Cost (\\$k)', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    y0, y1 = ax1.get_ylim()
    ax1.set_ylim(y0, y1 + (y1 - y0) * 0.18)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(lambda_vals, rec_k, 's--', color=color2, linewidth=2, markersize=6,
                  label='E[Recourse]')
    ax1_twin.set_ylabel('Expected Recourse (\\$k)', color=color2)
    ax1_twin.tick_params(axis='y', labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center',
               fontsize=8, framealpha=0.9, ncol=2, columnspacing=1.0)
    ax1.set_title('(a) Cost Components')

    ax2 = axes[1]
    x = np.arange(len(lambda_vals))
    width = 0.6
    bottom = np.zeros(len(lambda_vals))
    for key, label, color in [('high', 'High-end', '#2166AC'),
                              ('mid', 'Mid-range', '#4DAF4A'),
                              ('standard', 'Standard', '#F1A340'),
                              ('ultra', 'Ultra-high', '#762A83')]:
        vals = tier_means.get(key, [])
        arr = np.array([0.0 if (i >= len(vals) or vals[i] is None) else vals[i]
                        for i in valid_idx], dtype=float)
        ax2.bar(x, arr, width, bottom=bottom, label=label,
                color=color, edgecolor='white', linewidth=0.5, zorder=3)
        bottom = bottom + arr

    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{fmt_num(l)}' for l in lambda_vals])
    ax2.set_xlabel('$\\lambda$')
    ax2.set_ylabel('GPUs Procured (by tier)')
    ax2.set_title('(b) GPU Procurement Composition')
    ax2.legend(loc='upper left', fontsize=7, framealpha=0.9)
    ax2.grid(axis='x')
    ax2.set_axisbelow(True)

    fig.tight_layout()
    save_figure(fig, filename)


# Plot total cost, CVaR_0.95, and fleet composition vs. correlation rho (Fig. 7).
def plot_rho_sweep(data, filename):
    rhos_raw = data.get('rhos', [])
    total_raw = data.get('total_mean', [])
    total_s_raw = data.get('total_std', [])
    cvar_raw = data.get('cvar_mean', [])
    cvar_s_raw = data.get('cvar_std', [])
    valid_idx = [i for i, (r, t, c) in enumerate(zip(rhos_raw, total_raw, cvar_raw))
                 if r is not None and t is not None and c is not None]
    if not valid_idx:
        print(f"  [SKIP] Empty data for {filename}")
        return

    rhos = [rhos_raw[i] for i in valid_idx]
    x = np.array(rhos, dtype=float)
    total_m = np.array([total_raw[i] for i in valid_idx], dtype=float) / 1000
    total_s = np.array([0.0 if total_s_raw[i] is None else total_s_raw[i]
                        for i in valid_idx], dtype=float) / 1000
    cvar_m = np.array([cvar_raw[i] for i in valid_idx], dtype=float) / 1000
    cvar_s = np.array([0.0 if cvar_s_raw[i] is None else cvar_s_raw[i]
                       for i in valid_idx], dtype=float) / 1000

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

    ax1 = axes[0]
    ax1.plot(x, total_m, 'o-', color='#2166AC', linewidth=2, markersize=6,
             markerfacecolor='white', markeredgewidth=1.8, markeredgecolor='#2166AC',
             label='Total Cost', zorder=3)
    ax1.fill_between(x, total_m - total_s, total_m + total_s, color='#2166AC', alpha=0.15)
    ax1.plot(x, cvar_m, 's--', color='#B2182B', linewidth=2, markersize=6,
             markerfacecolor='white', markeredgewidth=1.8, markeredgecolor='#B2182B',
             label='CVaR$_{0.95}$', zorder=3)
    ax1.fill_between(x, cvar_m - cvar_s, cvar_m + cvar_s, color='#B2182B', alpha=0.15)
    ax1.set_xlabel('Demand--Output Correlation $\\rho$')
    ax1.set_ylabel('Cost (\\$k)')
    ax1.set_title('(a) Cost and Tail Risk vs. $\\rho$')
    ax1.set_xticks(x)
    ax1.legend(loc='upper left', framealpha=0.9)

    ax2 = axes[1]
    xi = np.arange(len(rhos))
    width = 0.6
    bottom = np.zeros(len(rhos))
    tier_means = data.get('tier_means', {})
    for key, label, color in [('high', 'High-end', '#2166AC'),
                              ('mid', 'Mid-range', '#4DAF4A'),
                              ('standard', 'Standard', '#F1A340'),
                              ('ultra', 'Ultra-high', '#762A83')]:
        vals = tier_means.get(key, [])
        arr = np.array([0.0 if (i >= len(vals) or vals[i] is None) else vals[i]
                        for i in valid_idx], dtype=float)
        ax2.bar(xi, arr, width, bottom=bottom, label=label,
                color=color, edgecolor='white', linewidth=0.5, zorder=3)
        bottom = bottom + arr
    ax2.set_xticks(xi)
    ax2.set_xticklabels([f'{fmt_num(r)}' for r in rhos])
    ax2.set_xlabel('Demand--Output Correlation $\\rho$')
    ax2.set_ylabel('GPUs Procured (by tier)')
    ax2.set_title('(b) Fleet Composition vs. $\\rho$')
    ax2.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax2.grid(axis='x')
    ax2.set_axisbelow(True)

    fig.tight_layout()
    save_figure(fig, filename)


# Plot mean-CVaR frontiers across correlation levels rho (Fig. 8).
def plot_rho_frontier(data, filename):
    if not data:
        print(f"  [SKIP] Empty data for {filename}")
        return

    rhos = sorted(float(k) for k in data.keys())
    rho_colors = {rhos[0]: '#2166AC', rhos[1]: '#4DAF4A', rhos[2]: '#B2182B'} \
        if len(rhos) == 3 else {r: c for r, c in zip(rhos, plt.cm.viridis(np.linspace(0, 0.85, len(rhos))))}

    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'v', 'P']

    plotted = []
    for rho in rhos:
        cell = _lookup_float_key(data, rho)
        if not cell:
            continue
        points = sorted(
            [(l, er, cv)
             for l, er, cv in zip(cell.get('lambdas', []),
                                  cell.get('e_rec_mean', []),
                                  cell.get('cvar_mean', []))
             if l is not None and er is not None and cv is not None],
            key=lambda t: float(t[0]))
        if points:
            plotted.append((rho, points))

    if not plotted:
        print(f"  [SKIP] Empty data for {filename}")
        return

    fig, ax = plt.subplots(figsize=(6, 4.3))
    for idx, (rho, points) in enumerate(plotted):
        er = [v / 1000 for _, v, _ in points]
        cv = [v / 1000 for _, _, v in points]
        ax.plot(er, cv,
                linestyle=line_styles[idx % len(line_styles)],
                marker=markers[idx % len(markers)],
                color=rho_colors[rho], linewidth=1.8, markersize=6,
                markerfacecolor='white', markeredgewidth=1.6, markeredgecolor=rho_colors[rho],
                label=f'$\\rho$={fmt_num(rho)}', zorder=3)

    ax.annotate('Increasing $\\lambda$',
                xy=(0.18, 0.62), xytext=(0.36, 0.80),
                xycoords='axes fraction', textcoords='axes fraction',
                fontsize=8, color='0.35', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='none', alpha=0.85),
                arrowprops=dict(arrowstyle='->', color='0.35', lw=1.0,
                                shrinkA=2, shrinkB=2))

    ax.set_xlabel('Expected Recourse Cost (\\$k)')
    ax.set_ylabel('CVaR$_{0.95}$ (\\$k)')
    ax.set_title('Risk--Cost Frontier across Correlation Levels')
    ax.margins(x=0.08, y=0.08)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              framealpha=0.95, title='Correlation', borderaxespad=0.0)

    fig.tight_layout(rect=[0, 0, 0.82, 1])
    save_figure(fig, filename)


# Plot cross-distribution out-of-sample transfer parity scatter (Fig. 9).
def plot_dist_oos(rows, filename):
    rows = [r for r in (rows or [])
            if None not in (r.get('ga_opt'), r.get('lnplan_ga'),
                            r.get('ln_opt'), r.get('gaplan_ln'))]
    if not rows:
        print(f"  [SKIP] Empty rows for {filename}")
        return
    fig, ax = plt.subplots(figsize=(5.8, 5))

    x_ln = np.array([r['ga_opt'] for r in rows], dtype=float) / 1000
    y_ln = np.array([r['lnplan_ga'] for r in rows], dtype=float) / 1000
    x_ga = np.array([r['ln_opt'] for r in rows], dtype=float) / 1000
    y_ga = np.array([r['gaplan_ln'] for r in rows], dtype=float) / 1000

    allv = np.concatenate([x_ln, y_ln, x_ga, y_ga])
    lo, hi = allv.min() * 0.8, allv.max() * 1.25

    ax.plot([lo, hi], [lo, hi], color='gray', linestyle='--', linewidth=1.2,
            zorder=1, label='$y=x$ (zero regret)')

    ax.scatter(x_ln, y_ln, s=55, facecolor='#2166AC', edgecolor='white',
               linewidth=0.8, alpha=0.9, zorder=3, label='LN plan on Gamma')
    ax.scatter(x_ga, y_ga, s=55, facecolor='#B2182B', edgecolor='white',
               linewidth=0.8, marker='^', alpha=0.9, zorder=3, label='Gamma plan on LN')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('In-distribution Optimum (\\$k)')
    ax.set_ylabel('Out-of-sample Cost of Transferred Plan (\\$k)')
    ax.set_title('Cross-distribution Plan Transfer')

    reg = [r['reg_ln'] for r in rows if r.get('reg_ln') is not None] + \
          [r['reg_ga'] for r in rows if r.get('reg_ga') is not None]
    if reg:
        med = np.median(reg)
        mx = np.max(reg)
        txt = f'median regret = {med:.2f}%\nmax regret = {mx:.2f}%'
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, ha='left', va='top',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))

    ax.legend(loc='lower right', framealpha=0.9, fontsize=8)
    fig.tight_layout()
    save_figure(fig, filename)


# Load plot_data.json and render all figures to PDF.
def main():
    if not os.path.exists(PLOT_JSON):
        print(f"[ERROR] {PLOT_JSON} not found. Run sensitivity_analysis.py first.")
        sys.exit(1)

    with open(PLOT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with plt.rc_context(PLOT_STYLE):
        print("=" * 70)
        print("Part 1: Sensitivity analysis plots")
        print("=" * 70)

        for key, exp_label, xlabel, fig_name, logy in [
            ('sens_cloud_cost', 'Elastic Cloud Cost Multiplier',
             '$\\gamma^{cloud}$', 'fig_sens_cloud_cost.pdf', False),
            ('sens_mukv', 'KV Cache Memory Coefficient Multiplier',
             '$\\gamma^{mem}$', 'fig_sens_mukv.pdf', True),
            ('sens_demand_cv', 'Demand Uncertainty Level',
             '$\\gamma^{cv}$', 'fig_sens_demand_cv.pdf', False),
        ]:
            if key not in data:
                print(f"  [SKIP] No data for {key}")
                continue
            d = data[key]
            plot_sensitivity_line(
                d.get('gammas', []), d.get('total_mean', []), d.get('total_std', []),
                xlabel, f'Total Cost vs. {exp_label}',
                os.path.join(OUTPUT_DIR, fig_name), logy=logy)

        print("\n" + "=" * 70)
        print("Part 2: Risk aversion plots")
        print("=" * 70)

        if 'risk_pareto' in data:
            pareto = data['risk_pareto']
            lambda_vals = pareto.get('lambda', [])
            tail = data.get('risk_tail_ecdf', {}).get('by_lambda', {})
            dec = data.get('risk_decisions', {})
            scenario_costs_by_lambda = [
                _lookup_float_key(tail, l) or [] for l in lambda_vals
            ]

            plot_risk_pareto(lambda_vals, pareto.get('e_recourse_mean', []),
                             pareto.get('cvar_mean', []),
                             os.path.join(OUTPUT_DIR, 'fig_risk_pareto.pdf'))
            plot_risk_tail_ecdf(lambda_vals, scenario_costs_by_lambda,
                                os.path.join(OUTPUT_DIR, 'fig_risk_tail_ecdf.pdf'))
            plot_risk_decisions(dec.get('lambda', []), dec.get('tier_means', {}),
                                dec.get('capital_mean', []), dec.get('e_recourse_mean', []),
                                os.path.join(OUTPUT_DIR, 'fig_risk_decisions.pdf'))
        else:
            print("  [SKIP] risk_pareto")

        print("\n" + "=" * 70)
        print("Part 3: Robustness plots")
        print("=" * 70)

        if 'robust_rho' in data:
            plot_rho_sweep(data['robust_rho'],
                           os.path.join(OUTPUT_DIR, 'fig_robust_rho.pdf'))
        else:
            print("  [SKIP] robust_rho")

        if 'robust_rho_frontier' in data:
            plot_rho_frontier(data['robust_rho_frontier'].get('by_rho', {}),
                              os.path.join(OUTPUT_DIR, 'fig_robust_rho_frontier.pdf'))
        else:
            print("  [SKIP] robust_rho_frontier")

        if 'robust_dist_oos' in data:
            plot_dist_oos(data['robust_dist_oos'].get('rows', []),
                          os.path.join(OUTPUT_DIR, 'fig_robust_dist_oos.pdf'))
        else:
            print("  [SKIP] robust_dist_oos")

    print("\n" + "=" * 70)
    print("All plots done.")
    print("=" * 70)


if __name__ == "__main__":
    main()

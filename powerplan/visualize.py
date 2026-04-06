"""
Visualization and reporting for power system simulations.

Generates multi-panel plots of energy flows, storage state, efficiency,
cost analysis, deployment comparisons, system architecture diagrams,
and Sankey-style energy flow diagrams.

Uses a colorblind-safe palette from styles.py.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from typing import Optional

from .grid import EnergyDispatcher, DispatchResult
from .styles import (
    PALETTE, FONTS, apply_style, source_color, storage_color,
    categorical_colors, format_kw, format_kwh, kw_formatter, kwh_formatter,
    add_nighttime_shading, styled_legend, draw_rounded_box, draw_flow_arrow,
)


# ──────────────────────────────────────────────────────────────────────
# Main Simulation Dashboard
# ──────────────────────────────────────────────────────────────────────

def plot_simulation(dispatcher: EnergyDispatcher, save_path: str = "powerplan_results.png",
                    title: Optional[str] = None, days: Optional[int] = None):
    """Generate comprehensive multi-panel simulation dashboard."""
    results = dispatcher.results
    if not results:
        print("No results to plot. Run simulation first.")
        return

    metrics = dispatcher.compute_metrics()
    dt = results[1].hour - results[0].hour if len(results) > 1 else 1.0
    steps_per_day = int(24 / dt)
    if days is not None:
        results = results[:days * steps_per_day]

    hours = np.array([r.hour for r in results])
    days_axis = hours / 24.0
    demand = np.array([r.demand_kw for r in results])
    generation = np.array([r.total_generation_kw for r in results])
    discharge = np.array([r.total_storage_discharge_kw for r in results])
    charge = np.array([r.total_storage_charge_kw for r in results])
    grid_imp = np.array([r.grid_import_kw for r in results])
    grid_exp = np.array([r.grid_export_kw for r in results])
    curtail = np.array([r.curtailment_kw for r in results])
    sys_eff = np.array([r.system_efficiency for r in results])
    renew_frac = np.array([r.renewable_fraction for r in results])

    storage_names = list(results[0].storage_states.keys()) if results else []
    storage_socs = {name: np.array([r.storage_states[name].soc for r in results])
                    for name in storage_names}
    source_names = list(results[0].source_outputs.keys()) if results else []
    source_powers = {name: np.array([r.source_outputs[name] for r in results])
                     for name in source_names}

    if title is None:
        title = f"PowerPlan \u2014 {dispatcher.config.name} ({dispatcher.config.scale.name})"

    fig = plt.figure(figsize=(20, 16), facecolor="white")
    fig.suptitle(title, **FONTS["suptitle"], y=0.98)
    gs = gridspec.GridSpec(4, 3, hspace=0.38, wspace=0.32,
                           left=0.06, right=0.97, top=0.93, bottom=0.05)

    C = PALETTE  # shorthand

    # --- Panel 1: Power Balance ---
    ax1 = fig.add_subplot(gs[0, :2])
    apply_style(ax1)
    add_nighttime_shading(ax1, days_axis)
    ax1.fill_between(days_axis, 0, generation, alpha=0.25, color=C["generation"],
                     label="Generation", linewidth=0)
    ax1.plot(days_axis, generation, color=C["generation"], linewidth=0.5, alpha=0.6)
    ax1.plot(days_axis, demand, color=C["demand"], linewidth=1.2, label="Demand")
    ax1.fill_between(days_axis, 0, -charge, alpha=0.25, color=C["charge"],
                     label="Charging", linewidth=0)
    if np.any(discharge > 0):
        ax1.fill_between(days_axis, demand, demand + discharge, alpha=0.25,
                         color=C["discharge"], label="Discharge", linewidth=0)
    if np.any(grid_imp > 0):
        ax1.fill_between(days_axis, 0, grid_imp, alpha=0.15, color=C["grid_import"],
                         label="Grid Import", linewidth=0)
    ax1.set_ylabel("Power", fontsize=9)
    ax1.set_xlabel("Day", fontsize=9)
    ax1.set_title("Power Balance", **FONTS["title"])
    ax1.yaxis.set_major_formatter(kw_formatter())
    styled_legend(ax1, loc="upper right", ncol=3)

    # --- Panel 2: Visual Metrics Dashboard (replaces text wall) ---
    ax_info = fig.add_subplot(gs[0, 2])
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    ax_info.axis("off")
    ax_info.set_facecolor(C["bg_panel"])

    # Gauge bars
    gauge_metrics = [
        ("Self-Sufficiency", metrics["self_sufficiency"], C["charge"]),
        ("System Efficiency", metrics["avg_system_efficiency"], C["discharge"]),
        ("Renewable Fraction", metrics["avg_renewable_fraction"], C["solar"]),
        ("Gen/Demand Ratio", min(metrics["generation_to_demand_ratio"] / 2, 1.0), C["generation"]),
    ]
    for i, (label, val, color) in enumerate(gauge_metrics):
        y = 0.88 - i * 0.14
        ax_info.barh(y, val, height=0.06, color=color, alpha=0.8, left=0.32)
        ax_info.barh(y, 1.0, height=0.06, color="#E8E8E8", alpha=0.4, left=0.32, zorder=0)
        ax_info.text(0.30, y, label, ha="right", va="center", fontsize=8,
                    color=C["text_dark"])
        if "Ratio" in label:
            ax_info.text(0.34 + val, y, f"{metrics['generation_to_demand_ratio']:.2f}",
                        ha="left", va="center", fontsize=8, fontweight="bold")
        else:
            ax_info.text(0.34 + val, y, f"{val*100:.1f}%",
                        ha="left", va="center", fontsize=8, fontweight="bold")

    # Key numbers below gauges
    y_nums = 0.22
    ax_info.text(0.15, y_nums, f"${metrics['total_capex_usd']:,.0f}",
                fontsize=11, fontweight="bold", ha="center", color=C["text_dark"])
    ax_info.text(0.15, y_nums - 0.08, "Total CAPEX", fontsize=7, ha="center",
                color=C["text_mid"])
    ax_info.text(0.50, y_nums, f"${metrics['estimated_lcoe_usd_kwh']:.4f}",
                fontsize=11, fontweight="bold", ha="center", color=C["text_dark"])
    ax_info.text(0.50, y_nums - 0.08, "LCOE $/kWh", fontsize=7, ha="center",
                color=C["text_mid"])
    ax_info.text(0.85, y_nums, format_kw(metrics["peak_demand_kw"]),
                fontsize=11, fontweight="bold", ha="center", color=C["text_dark"])
    ax_info.text(0.85, y_nums - 0.08, "Peak Demand", fontsize=7, ha="center",
                color=C["text_mid"])

    # --- Panel 3: Source Breakdown ---
    ax2 = fig.add_subplot(gs[1, :2])
    apply_style(ax2)
    add_nighttime_shading(ax2, days_axis)
    bottom = np.zeros_like(days_axis)
    for name in source_names:
        color = source_color(name)
        ax2.fill_between(days_axis, bottom, bottom + source_powers[name],
                         alpha=0.55, color=color, label=name, linewidth=0)
        bottom += source_powers[name]
    ax2.plot(days_axis, demand, color=C["demand"], linewidth=1.0,
             linestyle="--", label="Demand", alpha=0.8)
    ax2.set_ylabel("Power", fontsize=9)
    ax2.set_xlabel("Day", fontsize=9)
    ax2.set_title("Generation Source Breakdown", **FONTS["title"])
    ax2.yaxis.set_major_formatter(kw_formatter())
    styled_legend(ax2, loc="upper right", ncol=2)

    # --- Panel 4: Storage SOC ---
    ax3 = fig.add_subplot(gs[1, 2])
    apply_style(ax3)
    for name in storage_names:
        color = storage_color(name)
        ax3.plot(days_axis, storage_socs[name] * 100, color=color,
                 linewidth=1.2, label=name)
    ax3.set_ylabel("State of Charge (%)", fontsize=9)
    ax3.set_xlabel("Day", fontsize=9)
    ax3.set_ylim(-5, 105)
    ax3.set_title("Storage SOC", **FONTS["title"])
    styled_legend(ax3, loc="upper right")

    # --- Panel 5: System Efficiency ---
    ax4 = fig.add_subplot(gs[2, 0])
    apply_style(ax4)
    window = max(1, steps_per_day)
    if len(sys_eff) >= window:
        eff_smooth = np.convolve(sys_eff, np.ones(window)/window, mode="valid")
        d_ax = days_axis[:len(eff_smooth)]
        ax4.fill_between(d_ax, 0, eff_smooth * 100, alpha=0.15, color=C["discharge"])
        ax4.plot(d_ax, eff_smooth * 100, color=C["discharge"], linewidth=1.5)
    else:
        ax4.plot(days_axis, sys_eff * 100, color=C["discharge"], linewidth=1.5)
    ax4.set_ylabel("Efficiency (%)", fontsize=9)
    ax4.set_xlabel("Day", fontsize=9)
    ax4.set_ylim(0, 105)
    ax4.set_title("System Efficiency (24h avg)", **FONTS["title"])

    # --- Panel 6: Renewable Penetration ---
    ax5 = fig.add_subplot(gs[2, 1])
    apply_style(ax5)
    if len(renew_frac) >= window:
        rf_smooth = np.convolve(renew_frac, np.ones(window)/window, mode="valid")
        d_ax = days_axis[:len(rf_smooth)]
        ax5.fill_between(d_ax, 0, rf_smooth * 100, alpha=0.20, color=C["charge"])
        ax5.plot(d_ax, rf_smooth * 100, color=C["charge"], linewidth=1.5)
    else:
        ax5.fill_between(days_axis, 0, renew_frac * 100, alpha=0.20, color=C["charge"])
    ax5.set_ylabel("Renewable (%)", fontsize=9)
    ax5.set_xlabel("Day", fontsize=9)
    ax5.set_ylim(0, 105)
    ax5.set_title("Renewable Penetration (24h avg)", **FONTS["title"])

    # --- Panel 7: Energy Flow Waterfall ---
    ax6 = fig.add_subplot(gs[2, 2])
    apply_style(ax6)
    gen_kwh = metrics["total_generation_kwh"]
    loss_kwh = metrics["total_controller_losses_kwh"]
    direct_kwh = metrics["total_demand_kwh"] - metrics["total_grid_import_kwh"]
    stored_kwh = np.sum(discharge) * dt
    curtail_kwh = metrics["total_curtailment_kwh"]
    import_kwh = metrics["total_grid_import_kwh"]

    wf_labels = ["Generated", "- Losses", "Direct Use", "Via Storage", "Curtailed", "Grid Import"]
    wf_values = [gen_kwh, -loss_kwh, -direct_kwh, -stored_kwh, -curtail_kwh, import_kwh]
    wf_colors = [C["generation"], C["loss"], C["demand"], C["discharge"],
                 C["curtail"], C["grid_import"]]

    cumulative = 0
    for i, (label, val, color) in enumerate(zip(wf_labels, wf_values, wf_colors)):
        if val >= 0:
            ax6.barh(i, val, left=cumulative, color=color, alpha=0.8, height=0.6)
            cumulative += val
        else:
            ax6.barh(i, -val, left=cumulative + val, color=color, alpha=0.8, height=0.6)
            cumulative += val
    ax6.set_yticks(range(len(wf_labels)))
    ax6.set_yticklabels(wf_labels, fontsize=8)
    ax6.invert_yaxis()
    ax6.set_title("Energy Flow", **FONTS["title"])
    ax6.xaxis.set_major_formatter(kwh_formatter())

    # --- Panel 8: Curtailment & Grid ---
    ax7 = fig.add_subplot(gs[3, :2])
    apply_style(ax7)
    add_nighttime_shading(ax7, days_axis)
    ax7.fill_between(days_axis, 0, curtail, alpha=0.35, color=C["curtail"],
                     label="Curtailment", linewidth=0)
    ax7.fill_between(days_axis, 0, -grid_exp, alpha=0.35, color=C["grid_export"],
                     label="Grid Export", linewidth=0)
    ax7.fill_between(days_axis, 0, grid_imp, alpha=0.35, color=C["grid_import"],
                     label="Grid Import", linewidth=0)
    ax7.set_ylabel("Power", fontsize=9)
    ax7.set_xlabel("Day", fontsize=9)
    ax7.set_title("Curtailment & Grid Interaction", **FONTS["title"])
    ax7.yaxis.set_major_formatter(kw_formatter())
    styled_legend(ax7, loc="upper right")

    # --- Panel 9: Storage Health ---
    ax8 = fig.add_subplot(gs[3, 2])
    apply_style(ax8)
    storage_info = metrics.get("storage_details", [])
    if storage_info:
        names = [s["name"] for s in storage_info]
        healths = [s["health"] * 100 for s in storage_info]
        cycles = [s["cycles"] for s in storage_info]
        x_pos = np.arange(len(names))
        colors_s = [storage_color(n) for n in names]
        ax8.bar(x_pos, healths, color=colors_s, alpha=0.8)
        ax8.set_xticks(x_pos)
        ax8.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
        ax8.set_ylabel("Health (%)", fontsize=9)
        ax8.set_ylim(0, 105)
        ax8.set_title("Storage Health", **FONTS["title"])
        for i, (h, c) in enumerate(zip(healths, cycles)):
            ax8.text(i, h + 1, f"{c:.0f}cy", ha="center", **FONTS["annotation"])
    else:
        ax8.text(0.5, 0.5, "No storage", ha="center", va="center",
                 transform=ax8.transAxes, color=C["text_light"], fontsize=12)
        ax8.set_title("Storage Health", **FONTS["title"])

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved plot to {save_path}")


# ──────────────────────────────────────────────────────────────────────
# Deployment Comparison with Radar Chart
# ──────────────────────────────────────────────────────────────────────

def plot_comparison(dispatchers: list[EnergyDispatcher],
                    save_path: str = "powerplan_comparison.png"):
    """Compare multiple deployment configurations with radar chart and bar charts."""
    n = len(dispatchers)
    if n == 0:
        return

    all_metrics = [d.compute_metrics() for d in dispatchers]
    names = [d.config.name for d in dispatchers]
    colors = categorical_colors(n)

    fig = plt.figure(figsize=(22, 10), facecolor="white")
    fig.suptitle("PowerPlan \u2014 Deployment Scale Comparison",
                 **FONTS["suptitle"])
    gs = gridspec.GridSpec(2, 4, width_ratios=[1.4, 1, 1, 1],
                           hspace=0.35, wspace=0.30,
                           left=0.05, right=0.97, top=0.90, bottom=0.08)

    # --- Radar Chart (tall left panel) ---
    ax_radar = fig.add_subplot(gs[:, 0], projection="polar")
    metric_keys = ["self_sufficiency", "avg_system_efficiency",
                   "avg_renewable_fraction", "generation_to_demand_ratio",
                   "curtailment_fraction"]
    metric_labels = ["Self-\nSufficiency", "System\nEfficiency",
                     "Renewable\nFraction", "Gen/Demand\nRatio", "Low\nCurtailment"]
    num_vars = len(metric_keys)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    for i, (m, name) in enumerate(zip(all_metrics, names)):
        values = []
        for k in metric_keys:
            v = m.get(k, 0)
            if k == "generation_to_demand_ratio":
                v = min(v / 2.0, 1.0)
            elif k == "curtailment_fraction":
                v = 1.0 - min(v, 1.0)
            values.append(v)
        values += values[:1]
        ax_radar.fill(angles, values, alpha=0.12, color=colors[i])
        ax_radar.plot(angles, values, linewidth=1.5, color=colors[i],
                      label=name, marker="o", markersize=3)

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(metric_labels, fontsize=7)
    ax_radar.set_ylim(0, 1.05)
    ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_radar.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=6,
                              color=PALETTE["text_light"])
    ax_radar.set_title("Multi-Metric Profile", **FONTS["title"], pad=15)
    ax_radar.legend(loc="lower left", bbox_to_anchor=(-0.1, -0.15),
                    fontsize=6, ncol=2)

    # --- Bar charts (2x3 grid on the right) ---
    bar_specs = [
        ("Self-Sufficiency (%)", "self_sufficiency", 100, "%"),
        ("Avg Efficiency (%)", "avg_system_efficiency", 100, "%"),
        ("Renewable Fraction (%)", "avg_renewable_fraction", 100, "%"),
        ("LCOE ($/kWh)", "estimated_lcoe_usd_kwh", 1, "$"),
        ("Total CAPEX", "total_capex_usd", 1, "capex"),
        ("Curtailment (%)", "curtailment_fraction", 100, "%"),
    ]
    positions = [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)]

    for (row, col), (title_str, key, mult, fmt) in zip(positions, bar_specs):
        ax = fig.add_subplot(gs[row, col])
        apply_style(ax)
        vals = [m.get(key, 0) * mult for m in all_metrics]
        ax.bar(range(n), vals, color=colors, alpha=0.85)
        ax.set_xticks(range(n))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=6)
        ax.set_title(title_str, **FONTS["title"])
        if "%" in fmt:
            ax.set_ylim(0, max(max(vals) * 1.15, 10))
        for j, v in enumerate(vals):
            if fmt == "$":
                label = f"${v:.4f}"
            elif fmt == "capex":
                label = f"${v/1e6:.1f}M" if v >= 1e6 else f"${v/1e3:.0f}K"
            else:
                label = f"{v:.1f}%"
            ax.text(j, v * 1.02 if v > 0 else 0.5, label,
                    ha="center", fontsize=7)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved comparison to {save_path}")


# ──────────────────────────────────────────────────────────────────────
# Growth Projection
# ──────────────────────────────────────────────────────────────────────

def plot_projection(projection, save_path: str = "powerplan_projection.png"):
    """Generate multi-year growth projection visualization."""
    results = projection.results
    if not results:
        print("No projection results to plot.")
        return

    profile = projection.profile
    years = [r.year for r in results]
    n = len(years)
    C = PALETTE

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor="white")
    fig.suptitle(f"Growth Projection \u2014 {profile.name}\n"
                 f"Pop: {profile.population:,} | Climate: {profile.climate.name} | "
                 f"Target: {profile.renewable_target_pct*100:.0f}% by "
                 f"{profile.renewable_target_year}, net-zero by {profile.net_zero_year}",
                 **FONTS["suptitle"])

    # 1. Capacity Evolution — stacked area
    ax = axes[0, 0]
    apply_style(ax)
    fossil = [r.fossil_capacity_kw / 1000 for r in results]
    renew = [r.renewable_capacity_kw / 1000 for r in results]
    storage = [r.storage_capacity_kwh / 1000 for r in results]

    ax.fill_between(years, 0, fossil, alpha=0.6, color=C["gas"],
                    label="Fossil (MW)")
    ax.fill_between(years, fossil, [f + r for f, r in zip(fossil, renew)],
                    alpha=0.6, color=C["charge"], label="Renewable (MW)")
    ax2t = ax.twinx()
    ax2t.plot(years, storage, "s-", color=C["discharge"], label="Storage (MWh)",
              markersize=5, linewidth=2)
    ax2t.set_ylabel("Storage (MWh)", fontsize=8, color=C["discharge"])
    ax2t.tick_params(labelsize=7, colors=C["text_mid"])
    ax.set_ylabel("Capacity (MW)", fontsize=9)
    ax.set_title("Capacity Evolution", **FONTS["title"])
    styled_legend(ax, loc="upper left")
    ax2t.legend(loc="upper right", fontsize=7)

    # Find and annotate crossover
    for i in range(len(years) - 1):
        if fossil[i] > renew[i] and fossil[i+1] <= renew[i+1]:
            frac = (fossil[i] - renew[i]) / ((renew[i+1] - renew[i]) - (fossil[i+1] - fossil[i]))
            cross_year = years[i] + frac * (years[i+1] - years[i])
            ax.axvline(cross_year, color=C["text_light"], linestyle=":", alpha=0.7)
            ax.text(cross_year, max(max(fossil), max(renew)) * 0.9,
                    f"Crossover\n{cross_year:.0f}", ha="center", fontsize=7,
                    color=C["text_mid"])
            break

    # 2. LCOE Trend
    ax = axes[0, 1]
    apply_style(ax)
    lcoes = [r.lcoe for r in results]
    ax.plot(years, lcoes, "o-", color=C["grid_import"], linewidth=2, markersize=6)
    ax.fill_between(years, 0, lcoes, alpha=0.10, color=C["grid_import"])
    ax.set_ylabel("LCOE ($/kWh)", fontsize=9)
    ax.set_title("Levelized Cost of Energy", **FONTS["title"])
    for i, v in enumerate(lcoes):
        ax.text(years[i], v * 1.04, f"${v:.3f}", ha="center", fontsize=7)

    # 3. Renewable Fraction vs Target
    ax = axes[0, 2]
    apply_style(ax)
    actual = [r.renewable_actual_pct * 100 for r in results]
    target = [r.renewable_target_pct * 100 for r in results]
    ax.plot(years, actual, "o-", color=C["charge"], linewidth=2, label="Actual",
            markersize=6)
    ax.plot(years, target, "s--", color=C["generation"], linewidth=1.5,
            label="Target", markersize=5)
    ax.fill_between(years, 0, actual, alpha=0.12, color=C["charge"])
    ax.set_ylabel("Renewable (%)", fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_title("Renewable Penetration", **FONTS["title"])
    styled_legend(ax)
    for milestone, color in [(profile.renewable_target_year, C["generation"]),
                              (profile.net_zero_year, C["charge"])]:
        if years[0] <= milestone <= years[-1]:
            ax.axvline(milestone, color=color, linestyle=":", alpha=0.5)
            ax.text(milestone, 102, str(milestone), ha="center", fontsize=7,
                    color=color, fontweight="bold")

    # 4. Peak Demand Growth
    ax = axes[1, 0]
    apply_style(ax)
    peaks = [r.peak_demand_kw / 1000 for r in results]
    ax.bar(years, peaks, color=C["discharge"], alpha=0.8, width=3)
    ax.set_ylabel("Peak Demand (MW)", fontsize=9)
    ax.set_title("Peak Demand Growth", **FONTS["title"])
    for i, v in enumerate(peaks):
        ax.text(years[i], v * 1.01, f"{v:,.0f}", ha="center", fontsize=7)

    # 5. Self-Sufficiency
    ax = axes[1, 1]
    apply_style(ax)
    selfsuff = [r.metrics.get("self_sufficiency", 0) * 100 for r in results]
    ax.plot(years, selfsuff, "o-", color=C["hydro"], linewidth=2, markersize=6)
    ax.fill_between(years, 0, selfsuff, alpha=0.12, color=C["hydro"])
    ax.set_ylabel("Self-Sufficiency (%)", fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_title("Grid Independence", **FONTS["title"])

    # 6. Emissions
    ax = axes[1, 2]
    apply_style(ax)
    emissions = [r.emissions_tonnes for r in results]
    ax.bar(years, emissions, color=C["gas"], alpha=0.7, width=3)
    ax.set_ylabel("CO2 Emissions (tonnes)", fontsize=9)
    ax.set_title("Annual Emissions Trajectory", **FONTS["title"])
    for i, v in enumerate(emissions):
        if v > 0:
            label = f"{v/1000:,.1f}k" if v >= 1000 else f"{v:,.0f}"
            ax.text(years[i], v * 1.04, label, ha="center", fontsize=7)
    if len(emissions) >= 2 and emissions[0] > 0:
        reduction = (1 - emissions[-1] / emissions[0]) * 100
        ax.text(0.95, 0.95, f"{reduction:.0f}% reduction",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, fontweight="bold", color=C["charge"])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved projection to {save_path}")


# ──────────────────────────────────────────────────────────────────────
# System Architecture Diagram
# ──────────────────────────────────────────────────────────────────────

def plot_system_diagram(dispatcher: EnergyDispatcher,
                        save_path: str = "powerplan_diagram.png",
                        title: Optional[str] = None):
    """Draw a schematic of the power system showing components and energy flows."""
    config = dispatcher.config
    metrics = dispatcher.compute_metrics()
    C = PALETTE

    fig, ax = plt.subplots(1, 1, figsize=(16, 10), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    if title is None:
        title = f"System Architecture \u2014 {config.name}"
    ax.set_title(title, **FONTS["suptitle"], pad=20)

    sources = config.sources
    storage = config.storage_units
    controllers = config.controllers

    n_src = len(sources)
    n_stor = len(storage)
    n_ctrl = min(len(controllers), 4)  # Cap display

    # --- Source boxes (left column) ---
    src_x, src_w = 0.03, 0.18
    for i, src in enumerate(sources[:8]):
        y = 0.85 - i * (0.75 / max(n_src, 1))
        h = min(0.08, 0.70 / max(n_src, 1))
        color = source_color(src.name)
        label = f"{src.name}\n{format_kw(src.rated_kw)}"
        draw_rounded_box(ax, src_x, y - h/2, src_w, h, color, label, fontsize=7)
        # Arrow to controller column
        draw_flow_arrow(ax, (src_x + src_w, y), (0.28, 0.50),
                        width=src.rated_kw / max(sum(s.rated_kw for s in sources), 1),
                        color=color, alpha=0.3)

    ax.text(src_x + src_w / 2, 0.92, "SOURCES", ha="center",
            fontsize=10, fontweight="bold", color=C["text_mid"],
            transform=ax.transAxes)

    # --- Controller boxes (center-left) ---
    ctrl_x, ctrl_w = 0.30, 0.14
    for i, ctrl in enumerate(controllers[:n_ctrl]):
        y = 0.75 - i * (0.55 / max(n_ctrl, 1))
        h = min(0.07, 0.50 / max(n_ctrl, 1))
        label = f"{ctrl.name}\n{format_kw(ctrl.rated_kw)}"
        draw_rounded_box(ax, ctrl_x, y - h/2, ctrl_w, h, "#7F8C8D", label, fontsize=6)

    ax.text(ctrl_x + ctrl_w / 2, 0.92, "CONTROLLERS", ha="center",
            fontsize=10, fontweight="bold", color=C["text_mid"],
            transform=ax.transAxes)

    # --- Central bus ---
    bus_x = 0.52
    ax.plot([bus_x, bus_x], [0.15, 0.85], linewidth=4, color=C["text_dark"],
            alpha=0.6, transform=ax.transAxes, solid_capstyle="round")
    ax.text(bus_x, 0.88, "DC/AC BUS", ha="center", fontsize=8,
            fontweight="bold", color=C["text_dark"], transform=ax.transAxes)

    # Arrow from controllers to bus
    draw_flow_arrow(ax, (ctrl_x + ctrl_w, 0.55), (bus_x - 0.02, 0.55),
                    width=0.5, color=C["text_mid"], alpha=0.3)

    # --- Load box (right) ---
    load_x, load_w, load_h = 0.72, 0.18, 0.12
    load_y = 0.55
    draw_rounded_box(ax, load_x, load_y - load_h/2, load_w, load_h,
                     C["demand"],
                     f"LOAD\n{format_kw(metrics['peak_demand_kw'])} peak",
                     fontsize=9)
    draw_flow_arrow(ax, (bus_x + 0.02, load_y), (load_x, load_y),
                    width=0.6, color=C["demand"], alpha=0.4)

    # --- Grid box (top right) ---
    if config.grid_interconnect_kw > 0:
        grid_x, grid_w, grid_h = 0.72, 0.18, 0.08
        grid_y = 0.82
        draw_rounded_box(ax, grid_x, grid_y - grid_h/2, grid_w, grid_h,
                         C["grid_import"],
                         f"GRID\n{format_kw(config.grid_interconnect_kw)}",
                         fontsize=8)
        draw_flow_arrow(ax, (bus_x + 0.02, 0.75), (grid_x, grid_y),
                        width=0.3, color=C["grid_import"], alpha=0.3)

    # --- Storage boxes (bottom) ---
    ax.text(0.50, 0.08, "STORAGE", ha="center", fontsize=10,
            fontweight="bold", color=C["text_mid"], transform=ax.transAxes)
    stor_w = min(0.14, 0.70 / max(n_stor, 1))
    stor_start = 0.50 - (n_stor * stor_w * 1.1) / 2
    for i, unit in enumerate(storage[:6]):
        x = stor_start + i * stor_w * 1.1
        y = 0.14
        h = 0.06
        color = storage_color(unit.name)
        label = f"{unit.name}\n{format_kwh(unit.nominal_capacity_kwh)}"
        draw_rounded_box(ax, x, y, stor_w, h, color, label, fontsize=6)
        draw_flow_arrow(ax, (bus_x, 0.18), (x + stor_w / 2, y + h),
                        width=0.2, color=color, alpha=0.25)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved system diagram to {save_path}")


# ──────────────────────────────────────────────────────────────────────
# Energy Flow Diagram (Sankey-style)
# ──────────────────────────────────────────────────────────────────────

def plot_energy_flow(dispatcher: EnergyDispatcher,
                     save_path: str = "powerplan_flow.png",
                     title: Optional[str] = None):
    """Draw a Sankey-style energy flow diagram showing kWh from sources to destinations."""
    metrics = dispatcher.compute_metrics()
    C = PALETTE

    fig, ax = plt.subplots(1, 1, figsize=(14, 8), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    if title is None:
        title = f"Energy Flow \u2014 {dispatcher.config.name}"
    ax.set_title(title, **FONTS["suptitle"], pad=15)

    # --- Source data (left side) ---
    source_details = metrics.get("source_details", [])
    sources = [(s["name"], s.get("cumulative_kwh", 0)) for s in source_details
               if s.get("cumulative_kwh", 0) > 0]
    if not sources:
        ax.text(0.5, 0.5, "No energy data available", ha="center", va="center")
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return

    total_gen = sum(v for _, v in sources)

    # --- Destination data (right side) ---
    direct_use = metrics["total_demand_kwh"] - metrics["total_grid_import_kwh"]
    stored = np.sum([r.total_storage_charge_kw for r in dispatcher.results]) * (
        dispatcher.results[1].hour - dispatcher.results[0].hour if len(dispatcher.results) > 1 else 1)
    curtailed = metrics["total_curtailment_kwh"]
    ctrl_loss = metrics["total_controller_losses_kwh"]
    grid_exp = metrics["total_grid_export_kwh"]

    destinations = [
        ("Direct Use", max(0, direct_use), C["demand"]),
        ("Storage Charge", max(0, stored), C["charge"]),
        ("Curtailed", max(0, curtailed), C["curtail"]),
        ("Ctrl Losses", max(0, ctrl_loss), C["loss"]),
    ]
    if grid_exp > 0:
        destinations.append(("Grid Export", grid_exp, C["grid_export"]))
    total_dest = sum(v for _, v, _ in destinations)

    # Normalize heights
    left_margin, right_margin = 0.08, 0.92
    band_top, band_bottom = 0.85, 0.10
    usable_height = band_top - band_bottom

    # --- Draw source labels and bands (left) ---
    y_cursor = band_top
    src_positions = []
    for name, kwh in sources:
        h = (kwh / max(total_gen, 1)) * usable_height * 0.90
        h = max(h, 0.02)
        color = source_color(name)
        mid_y = y_cursor - h / 2
        ax.barh(mid_y, 0.12, height=h, left=left_margin, color=color, alpha=0.8)
        ax.text(left_margin - 0.01, mid_y, f"{name}\n{format_kwh(kwh)}",
                ha="right", va="center", fontsize=7, color=C["text_dark"])
        src_positions.append((mid_y, h, color))
        y_cursor -= h + 0.01

    # --- Draw destination labels and bands (right) ---
    y_cursor = band_top
    dest_positions = []
    for name, kwh, color in destinations:
        if kwh <= 0:
            continue
        h = (kwh / max(total_dest, 1)) * usable_height * 0.90
        h = max(h, 0.02)
        mid_y = y_cursor - h / 2
        ax.barh(mid_y, 0.12, height=h, left=right_margin - 0.12, color=color, alpha=0.8)
        ax.text(right_margin + 0.01, mid_y, f"{name}\n{format_kwh(kwh)}",
                ha="left", va="center", fontsize=7, color=C["text_dark"])
        dest_positions.append((mid_y, h, color))
        y_cursor -= h + 0.01

    # --- Draw connecting flow bands ---
    x_left = left_margin + 0.12
    x_right = right_margin - 0.12
    x_mid = (x_left + x_right) / 2
    n_flows = min(len(src_positions), len(dest_positions))
    for i in range(n_flows):
        sy, sh, sc = src_positions[i]
        dy, dh, dc = dest_positions[i]
        # Bezier-like sigmoid curve using fill_between
        xs = np.linspace(x_left, x_right, 50)
        t = (xs - x_left) / (x_right - x_left)
        sigmoid = 1 / (1 + np.exp(-8 * (t - 0.5)))
        y_top = sy + sh/2 + (dy + dh/2 - sy - sh/2) * sigmoid
        y_bot = sy - sh/2 + (dy - dh/2 - sy + sh/2) * sigmoid
        ax.fill_between(xs, y_bot, y_top, alpha=0.15, color=sc, linewidth=0)

    # Center label
    ax.text(0.50, 0.04, f"Total: {format_kwh(total_gen)}", ha="center",
            fontsize=11, fontweight="bold", color=C["text_dark"])

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved energy flow to {save_path}")

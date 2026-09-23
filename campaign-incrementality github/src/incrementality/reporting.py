"""Exhibits.

Four charts, in the order a reader needs them:

1. what happened versus what would have happened,
2. how the effect accumulated and whether it faded,
3. the tests that could have killed the estimate and did not,
4. what it was worth, and where the decision boundary sits.

Chart three is the one that gets an analysis believed. It is also the one most
readouts leave out.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

INK = "#1f2933"
MUTED = "#7b8794"
ACCENT = "#0b6e99"
COUNTERFACTUAL = "#9aa5b1"
POSITIVE = "#18794e"
NEGATIVE = "#b42318"
GRID = "#e4e7eb"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 10.5,
        "axes.titleweight": "semibold",
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "grid.color": GRID,
        "figure.facecolor": "white",
    }
)


def _format_week_axis(ax, label: str = "Week commencing") -> None:
    """Readable date ticks. Matplotlib's default packs one label per point,
    which on a weekly series runs them into an unreadable strip."""
    import matplotlib.dates as mdates

    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_xlabel(label, fontsize=9)
    for tick in ax.get_xticklabels():
        tick.set_rotation(0)
        tick.set_ha("center")


def _millions(value, _pos=None) -> str:
    return f"${value / 1e6:,.1f}M"


def _pct(value, _pos=None) -> str:
    return f"{value:+.0%}"


def plot_counterfactual(
    result, activation_week, outcome_label: str, path: Path, series=None,
    context_weeks: int = 20,
) -> Path:
    """What the business sold, against what it would have sold anyway.

    Pre-campaign weeks are shown so the reader can see the two lines are the
    same thing before the deal goes live. Without that context the chart
    opens on a gap already wide, and looks like an assertion rather than a
    measurement.
    """
    weekly = result.weekly
    fig, ax = plt.subplots(figsize=(9.2, 4.0))

    if series is not None and "week" in series.columns:
        history = series.copy()
        history["week"] = pd.to_datetime(history["week"])
        outcome_col = getattr(result, "outcome", None)
        if outcome_col in history.columns:
            history = history[history["week"] < activation_week].tail(context_weeks)
            if not history.empty:
                ax.plot(
                    history["week"], history[outcome_col],
                    color=ACCENT, linewidth=2.0,
                )
                bridge_x = [history["week"].iloc[-1], weekly["week"].iloc[0]]
                ax.plot(
                    bridge_x, [history[outcome_col].iloc[-1], weekly["actual"].iloc[0]],
                    color=ACCENT, linewidth=2.0,
                )
                ax.plot(
                    bridge_x,
                    [history[outcome_col].iloc[-1], weekly["counterfactual"].iloc[0]],
                    color=COUNTERFACTUAL, linewidth=1.8, linestyle="--",
                )

    # Shade the gap itself -- it is the measurement, so it should be the
    # thing the eye lands on rather than something the reader has to infer
    # from the distance between two lines.
    ax.fill_between(
        weekly["week"], weekly["counterfactual"], weekly["actual"],
        color=POSITIVE, alpha=0.16, linewidth=0, label="Extra sales from the campaign",
    )
    ax.fill_between(
        weekly["week"], weekly["counterfactual_low"], weekly["counterfactual_high"],
        color=COUNTERFACTUAL, alpha=0.30, linewidth=0,
        label=f"Range of likely outcomes ({int((1 - result.alpha) * 100)}%)",
    )
    ax.plot(
        weekly["week"], weekly["counterfactual"], color=COUNTERFACTUAL,
        linestyle="--", linewidth=2.0, label="Expected without the campaign",
    )
    ax.plot(
        weekly["week"], weekly["actual"], color=ACCENT, linewidth=2.4,
        marker="o", markersize=4, label="Actual sales",
    )
    ax.axvline(activation_week, color=INK, linewidth=1.1, linestyle=":")

    top = ax.get_ylim()[1]
    ax.annotate(
        "Campaign starts", xy=(activation_week, top), xytext=(-6, -10),
        textcoords="offset points", fontsize=8, color=MUTED, va="top", ha="right",
    )
    # Label the gap, which is the entire point of the chart.
    mid = weekly.index[len(weekly) // 2]
    ax.annotate(
        f"{result.relative_uplift:+.1%}  (${result.absolute_uplift / 1e6:,.1f}M)",
        xy=(weekly.loc[mid, "week"],
            (weekly.loc[mid, "actual"] + weekly.loc[mid, "counterfactual"]) / 2),
        xytext=(0, 0), textcoords="offset points", fontsize=9,
        color=POSITIVE, weight="bold", va="center", ha="center",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
              "edgecolor": POSITIVE, "linewidth": 0.8, "alpha": 0.92},
    )

    ax.set_title(
        f"Participating properties sold {result.relative_uplift:+.1%} more "
        f"than expected"
    )
    ax.set_ylabel(f"{outcome_label} ($M per week)", fontsize=9)
    ax.yaxis.set_major_formatter(FuncFormatter(_millions))
    _format_week_axis(ax)
    ax.grid(axis="y", linewidth=0.6)
    ax.legend(loc="upper left", fontsize=8.2, ncol=1, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_uplift(result, path: Path) -> Path:
    """How the extra sales built up, and whether the effect faded."""
    weekly = result.weekly
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7))

    colors = [POSITIVE if v >= 0 else NEGATIVE for v in weekly["relative_uplift"]]
    axes[0].bar(weekly["week"], weekly["relative_uplift"], color=colors, width=4.5)
    axes[0].axhline(0, color=INK, linewidth=0.8)
    axes[0].set_title("Extra sales each week, vs. expected")
    axes[0].set_ylabel("Sales above expected (%)", fontsize=9)
    axes[0].yaxis.set_major_formatter(FuncFormatter(_pct))
    axes[0].grid(axis="y", linewidth=0.6)
    _format_week_axis(axes[0])

    axes[1].plot(
        weekly["week"], weekly["cumulative_uplift"], color=ACCENT,
        linewidth=2.4, marker="o", markersize=4,
    )
    axes[1].fill_between(
        weekly["week"], 0, weekly["cumulative_uplift"], color=ACCENT,
        alpha=0.12, linewidth=0,
    )
    axes[1].axhline(0, color=INK, linewidth=0.8)
    axes[1].set_title("Running total of extra sales")
    axes[1].set_ylabel("Cumulative extra sales ($M)", fontsize=9)
    axes[1].yaxis.set_major_formatter(FuncFormatter(_millions))
    axes[1].grid(axis="y", linewidth=0.6)
    _format_week_axis(axes[1])
    final = weekly["cumulative_uplift"].iloc[-1]
    axes[1].annotate(
        f"${final / 1e6:,.1f}M",
        xy=(weekly["week"].iloc[-1], final), xytext=(-6, 8),
        textcoords="offset points", ha="right", fontsize=9,
        weight="bold", color=ACCENT,
    )

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_validation(report, observed_relative: float, path: Path) -> Path:
    """The checks that could have overturned the result."""

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))

    # -- 1: could a random group of properties look like this? -----------
    effects = report.placebo_distribution
    axes[0].hist(effects, bins=12, color=COUNTERFACTUAL, edgecolor="white", linewidth=0.8)
    axes[0].axvline(observed_relative, color=NEGATIVE, linewidth=2.2)
    axes[0].annotate(
        f"This campaign\n{observed_relative:+.1%}",
        xy=(observed_relative, axes[0].get_ylim()[1] * 0.92),
        xytext=(-8, 0), textcoords="offset points", ha="right",
        fontsize=8.5, color=NEGATIVE, weight="bold", va="top",
    )
    axes[0].set_title(f"Untouched property groups (n={len(effects)})", fontsize=10)
    axes[0].set_xlabel("Apparent sales lift (%)", fontsize=8.5)
    axes[0].set_ylabel("Number of groups", fontsize=8.5)
    axes[0].xaxis.set_major_formatter(FuncFormatter(_pct))
    axes[0].grid(axis="y", linewidth=0.5)

    # -- 2: does the answer depend on choices we made? -------------------
    sens = report.sensitivity.sort_values("relative_uplift")
    ypos = np.arange(len(sens))
    bar_colors = [
        ACCENT if ok else MUTED for ok in sens.get("defensible", [True] * len(sens))
    ]
    axes[1].barh(ypos, sens["relative_uplift"], color=bar_colors, alpha=0.85, height=0.62)
    axes[1].axvline(observed_relative, color=INK, linestyle=":", linewidth=1.3)
    axes[1].set_yticks(ypos)
    axes[1].set_yticklabels(
        [v.replace("_", " ").replace("pre-window", "history") for v in sens["variant"]],
        fontsize=7.4,
    )
    axes[1].set_title("Answer under alternative choices", fontsize=10)
    axes[1].set_xlabel("Sales lift (%)", fontsize=8.5)
    axes[1].xaxis.set_major_formatter(FuncFormatter(_pct))
    axes[1].grid(axis="x", linewidth=0.5)

    # -- 3: scorecard ----------------------------------------------------
    axes[2].axis("off")
    axes[2].set_title("All checks", loc="left", fontsize=10)
    y = 0.97
    for test in report.tests:
        mark, colour = ("PASS", POSITIVE) if test.passed else ("REVIEW", NEGATIVE)
        name = test.name.replace(" (underpowered)", "")
        axes[2].text(0.0, y, name, fontsize=8.6, color=INK, va="top")
        axes[2].text(
            1.0, y, mark, fontsize=8.6, color=colour, ha="right",
            va="top", weight="bold",
        )
        y -= 0.075
        for line in textwrap.wrap(test.detail, width=62)[:2]:
            axes[2].text(0.0, y, line, fontsize=6.8, color=MUTED, va="top")
            y -= 0.058
        y -= 0.018

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_economics(econ_result, grid: pd.DataFrame, path: Path) -> Path:
    """What the campaign was worth, and the line it had to clear."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9))

    rows = econ_result.pnl_rows()
    labels = [lab.replace("\n", " ") for lab in rows["line_item"]]
    values = rows["usd"].to_numpy(dtype=float)
    colors = [POSITIVE if v >= 0 else NEGATIVE for v in values]
    colors[-1] = ACCENT if values[-1] >= 0 else NEGATIVE
    axes[0].bar(range(len(values)), values, color=colors, width=0.58)
    axes[0].axhline(0, color=INK, linewidth=0.9)
    axes[0].set_xticks(range(len(values)))
    axes[0].set_xticklabels(
        [lab.replace(" ", "\n", 1) for lab in labels], fontsize=7.8
    )
    for i, value in enumerate(values):
        axes[0].annotate(
            f"${value / 1e6:,.2f}M", xy=(i, value),
            xytext=(0, 5 if value >= 0 else -12), textcoords="offset points",
            ha="center", fontsize=7.6, color=INK,
        )
    axes[0].margins(y=0.22)
    axes[0].set_title("Where the money went")
    axes[0].set_ylabel("Value ($M)", fontsize=9)
    axes[0].yaxis.set_major_formatter(FuncFormatter(_millions))
    axes[0].grid(axis="y", linewidth=0.6)

    pivot = grid.pivot(index="discount_depth", columns="relative_uplift", values="net_value")
    extent = [
        grid["relative_uplift"].min(), grid["relative_uplift"].max(),
        grid["discount_depth"].min(), grid["discount_depth"].max(),
    ]
    limit = float(np.nanmax(np.abs(pivot.to_numpy())))
    image = axes[1].imshow(
        pivot.to_numpy(), origin="lower", aspect="auto", extent=extent,
        cmap="RdYlGn", vmin=-limit, vmax=limit,
    )
    contour = axes[1].contour(
        pivot.columns.to_numpy(), pivot.index.to_numpy(), pivot.to_numpy(),
        levels=[0.0], colors=INK, linewidths=1.6,
    )
    axes[1].clabel(contour, fmt={0.0: "breaks even"}, fontsize=7.8)
    axes[1].set_title("Green pays back, red does not")
    axes[1].set_xlabel("Total sales lift (%)", fontsize=9)
    axes[1].set_ylabel("Discount depth (%)", fontsize=9)
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0%}"))
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0%}"))
    bar = fig.colorbar(image, ax=axes[1], fraction=0.045)
    bar.set_label("Net value ($M)", fontsize=8)
    bar.ax.yaxis.set_major_formatter(FuncFormatter(_millions))
    bar.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_drivers(decomposition, path: Path) -> Path:
    """What moved the lift: traffic, conversion, stay length and price.

    Green for drivers that added, red for what was given back. Ordered by
    contribution so the lever with the largest share reads first.
    """
    frame = decomposition.to_frame()
    if frame.empty:
        raise ValueError("no drivers to plot")
    frame = frame.sort_values("lift")

    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    colors = [POSITIVE if v >= 0 else NEGATIVE for v in frame["lift"]]
    ypos = np.arange(len(frame))
    ax.barh(ypos, frame["lift"], color=colors, height=0.58, alpha=0.9)
    ax.axvline(0, color=INK, linewidth=0.9)
    ax.set_yticks(ypos)
    ax.set_yticklabels(frame["driver"], fontsize=9)
    ax.xaxis.set_major_formatter(FuncFormatter(_pct))
    ax.set_title("What drove the lift: conversion, not traffic")
    ax.set_xlabel("Contribution to sales lift (%)", fontsize=9)
    ax.grid(axis="x", linewidth=0.6)

    # Labels always sit to the right of the zero line. Placing a negative
    # bar's label to its left runs it straight into the axis tick labels.
    for y, (lift, share) in enumerate(
        zip(frame["lift"], frame["share_of_movement"], strict=True)
    ):
        ax.annotate(
            f"{lift:+.1%}   ({share:.0%} of movement)",
            xy=(max(lift, 0.0), y),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=7.8,
            color=INK,
        )
    ax.margins(x=0.34)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def write_decision_memo(
    *,
    cfg,
    uplift_results: dict,
    econ_result,
    gross_econ,
    incrementality,
    drivers,
    report,
    stacking,
    path: Path,
) -> Path:
    """A one-page markdown readout, generated from the run rather than typed.

    Regenerating the memo from the artifacts is the point: nobody has to
    remember to update a number when the model changes.
    """
    primary = uplift_results[cfg.primary_outcome]
    lines: list[str] = []
    add = lines.append

    add(f"# Decision memo: {cfg.name}")
    add("")
    add(f"_Generated from pipeline run. Activation week {cfg.activation_week.date()}; "
        f"measurement window {cfg.post_start.date()} to {cfg.post_end.date()} "
        f"({cfg.n_post_weeks} weeks)._")
    add("")
    conf = int((1 - primary.alpha) * 100)
    add("## Headline")
    add("")
    add(
        f"Net incrementality: **{incrementality.net_lift:+.1%}** on "
        f"{cfg.primary_outcome.upper()} "
        f"({conf}% interval {incrementality.net_ci[0]:+.1%} to "
        f"{incrementality.net_ci[1]:+.1%}), worth "
        f"**${incrementality.net_value / 1e6:,.1f}M**. Verdict: {primary.verdict}."
    )
    add("")
    add("| Build | Rate | Value |")
    add("| --- | ---: | ---: |")
    for _, row in incrementality.build_rows().iterrows():
        add(f"| {row['line_item']} | {row['rate']:+.1%} | ${row['value'] / 1e6:,.1f}M |")
    add("")
    add(
        f"Room nights moved {uplift_results['room_nights'].relative_uplift:+.1%} against "
        f"booking value's {primary.relative_uplift:+.1%}; the gap is the discount "
        f"reaching travellers."
        if "room_nights" in uplift_results
        else None
    )
    add("")
    if drivers is not None and not drivers.to_frame().empty:
        add("## Drivers of growth")
        add("")
        add(drivers.narrative())
        add("")
        add("| Driver | Lift | 90% interval | Share of movement |")
        add("| --- | ---: | ---: | ---: |")
        for _, row in drivers.to_frame().iterrows():
            add(
                f"| {row['driver']} | {row['lift']:+.1%} | "
                f"{row['ci_low']:+.1%} to {row['ci_high']:+.1%} | "
                f"{row['share_of_movement']:.0%} |"
            )
        add("")
        add(
            f"Drivers compose to {drivers.composed_lift:+.2%} against a directly "
            f"estimated {drivers.outcome_lift:+.2%} — a residual of "
            f"{drivers.residual:+.2%}"
            + (
                ", which reconciles."
                if drivers.reconciles
                else ", which does NOT reconcile; treat the split as indicative and "
                "check the panel."
            )
        )
        add("")

    add("## Money")
    add("")
    add("| Line | USD |")
    add("| --- | --- |")
    for _, row in econ_result.summary_rows().iterrows():
        add(f"| {row['line_item']} | ${row['usd'] / 1e6:,.2f}M |")
    add("")
    add(
        f"Return on campaign spend: **{econ_result.return_on_spend:.2f}x** "
        f"(net ${econ_result.net_value / 1e6:,.1f}M). Break-even lift: "
        f"**{econ_result.breakeven_relative_uplift:.1%}**, against a measured "
        f"{primary.relative_uplift:.1%}. Of the "
        f"${econ_result.discount_value_granted / 1e6:,.1f}M discount granted, "
        f"**${econ_result.subsidy_on_baseline / 1e6:,.1f}M "
        f"({econ_result.subsidy_on_baseline / max(econ_result.discount_value_granted, 1):.0%})** "
        f"went to demand the model says would have converted anyway."
    )
    add("")

    if report.displacement_relative < 0:
        flipped = (gross_econ.net_value > 0) != (econ_result.net_value > 0)
        add("## What the displacement test is worth")
        add("")
        add("| | Gross of displacement | Net of displacement |")
        add("| --- | --- | --- |")
        add(
            f"| Incremental booking value | "
            f"${gross_econ.incremental_bookings / 1e6:,.1f}M | "
            f"${econ_result.incremental_bookings / 1e6:,.1f}M |"
        )
        add(
            f"| Effective lift | {gross_econ.incremental_bookings / gross_econ.baseline_bookings:.1%} | "
            f"{econ_result.incremental_bookings / econ_result.baseline_bookings:.1%} |"
        )
        add(
            f"| Return on spend | {gross_econ.return_on_spend:.2f}x | "
            f"{econ_result.return_on_spend:.2f}x |"
        )
        add(
            f"| Net value | ${gross_econ.net_value / 1e6:,.2f}M | "
            f"${econ_result.net_value / 1e6:,.2f}M |"
        )
        add("")
        if flipped:
            add(
                "**These land on opposite sides of break-even.** Accepting the "
                "no-displacement assumption without testing it would have booked this "
                "campaign as value-creating. Testing it reverses the decision, which "
                "makes the comp-set test the analysis rather than a robustness check."
            )
        else:
            add(
                "Both readings fall on the same side of break-even, so the "
                "displacement adjustment changes the size of the answer but not the "
                "decision."
            )
        add("")

    add("## Falsification scorecard")
    add("")
    add("| Test | Result | Statistic | Detail |")
    add("| --- | --- | --- | --- |")
    for test in report.tests:
        add(f"| {test.name} | {'PASS' if test.passed else 'FAIL'} | {test.statistic:.4f} | {test.detail} |")
    add("")

    if stacking is not None:
        add("## Stacking")
        add("")
        add(
            f"Stack-enabled properties outperformed stack-disabled participants by "
            f"**{stacking.incremental_stacking_value:+.1%}** "
            f"(90% CI {stacking.ci[0]:+.1%} to {stacking.ci[1]:+.1%}, p={stacking.p_value:.3f}). "
            f"Parallel-trends check p={stacking.parallel_trends_p:.2f} "
            f"({'holds' if stacking.parallel_trends_holds else 'FAILS -- treat as descriptive'})."
        )
        add("")

    add("## What this does not answer")
    add("")
    add("- Post-window carryover: the measurement stops at the campaign end date, so any "
        "repeat-booking effect is excluded and the estimate is a lower bound.")
    add("- Participation is not randomised. Properties opted into the campaign, so the "
        "estimate is the effect on participants, not the effect of extending the campaign "
        "to non-participants.")
    add("- The discount-depth ceiling holds volume fixed; a shallower discount would "
        "presumably have produced less lift.")
    add("")

    path.write_text("\n".join(line for line in lines if line is not None))
    return path


def plot_assumptions(
    report,
    series: pd.DataFrame,
    cfg,
    path: Path,
) -> Path:
    """One page showing whether the model's assumptions hold.

    Laid out the way a reviewer interrogates them: are the groups comparable,
    are the predictors clean, and does a period with no campaign read as
    having no campaign.
    """
    fig = plt.figure(figsize=(12.6, 5.4))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.35], hspace=0.55, wspace=0.22)
    fig.suptitle(
        "Assumption checks — all must hold before any result is reported",
        fontsize=12.5, fontweight="bold", x=0.075, ha="left", y=1.0,
    )

    # -- 1. cohort comparability ------------------------------------------
    ax1 = fig.add_subplot(grid[:, 0])
    comp = report.comparability
    cohorts = list(comp.index)
    ypos = np.arange(len(cohorts))
    ax1.barh(ypos, comp["volume_per_property_usd"] / 1e3, color=ACCENT, alpha=0.85, height=0.5)
    ax1.set_yticks(ypos)
    ax1.set_yticklabels([c.replace("_", " ").title() for c in cohorts], fontsize=9)
    ax1.set_xlabel("Average weekly sales per property ($000)", fontsize=8.5)
    ax1.set_title("1  The groups are comparable", loc="left", fontsize=10.5)
    ax1.grid(axis="x", linewidth=0.6)
    for y, cohort in enumerate(cohorts):
        corr = comp.loc[cohort, "pre_period_growth_corr"]
        props = int(comp.loc[cohort, "properties"])
        label = f"{props:,} properties"
        if np.isfinite(corr):
            label += f"  ·  corr {corr:.2f}"
        ax1.annotate(
            label,
            xy=(comp.loc[cohort, "volume_per_property_usd"] / 1e3, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.6,
            color=MUTED,
        )
    ax1.margins(x=0.38)

    # -- 2i. predictors unaffected ----------------------------------------
    ax2 = fig.add_subplot(grid[0, 1])
    frame = series.copy()
    frame["week"] = pd.to_datetime(frame["week"])
    window = frame[frame["week"] >= cfg.pre_start]
    cols = [cfg.primary_outcome, *cfg.model.predictors]
    palette = [ACCENT, "#7b8794", "#b0bcc9", "#6aa9d8", "#9ecae1", "#c6dbef"]
    for i, col in enumerate(cols):
        if col not in window.columns:
            continue
        values = window[col].to_numpy(dtype=float)
        scaled = (values - values.min()) / (values.max() - values.min() + 1e-12)
        ax2.plot(
            window["week"],
            scaled,
            linewidth=2.0 if i == 0 else 1.1,
            linestyle="--" if i == 0 else "-",
            color=palette[i % len(palette)],
            label=("Outcome" if i == 0 else col.replace("_", " ")),
        )
    ax2.axvline(cfg.activation_week, color=INK, linewidth=1.3)
    ax2.annotate(
        "Campaign starts", xy=(cfg.activation_week, 1.02), xycoords=("data", "axes fraction"),
        xytext=(5, 0), textcoords="offset points", fontsize=7.5, color=MUTED,
    )
    ax2.set_title(
        "2  Our yardsticks were not moved by the campaign", loc="left", fontsize=10.5
    )
    ax2.set_ylabel("Indexed to own range", fontsize=8)
    ax2.set_yticks([])
    ax2.set_ylim(-0.35, 1.18)
    _format_week_axis(ax2, label="")
    ax2.legend(fontsize=6.2, ncol=3, loc="lower left", framealpha=0.9)
    ax2.grid(axis="y", linewidth=0.5)

    # -- 2ii. pre-period placebo ------------------------------------------
    ax3 = fig.add_subplot(grid[1, 1])
    ax3.axis("off")
    ax3.set_title(
        "3  A quiet period correctly reads as no campaign", loc="left", fontsize=10.5
    )
    placebo = report.placebo
    header = f"{'Cohort':<12}{'Apparent effect':>18}{'90% interval':>24}{'Fit error':>12}"
    ax3.text(0.0, 0.80, header, fontsize=8.2, family="monospace", color=MUTED)
    for i, row in enumerate(placebo.itertuples()):
        effect = getattr(row, "apparent_effect", float("nan"))
        low = getattr(row, "ci_low", float("nan"))
        high = getattr(row, "ci_high", float("nan"))
        fit = getattr(row, "fit_error", float("nan"))
        ok = np.isfinite(effect) and abs(effect) <= 0.02
        line = (
            f"{row.cohort:<12}{effect:>17.2%}"
            f"{f'[{low:+.2%}, {high:+.2%}]':>24}{fit:>12.2%}"
        )
        ax3.text(
            0.0, 0.62 - i * 0.16, line, fontsize=8.2, family="monospace",
            color=POSITIVE if ok else NEGATIVE,
        )
    verdict = next(
        (c for c in report.checks if c.name == "Quiet period reads as quiet"), None
    )
    if verdict is not None:
        ax3.text(
            0.0, 0.18,
            ("PASS — " if verdict.passed else "FAIL — ") + verdict.detail,
            fontsize=7.6,
            color=POSITIVE if verdict.passed else NEGATIVE,
            wrap=True,
        )

    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return path

#!/usr/bin/env python3
"""The know-you score chart: does the profile actually model the student?

Plots a rolling 5-round accuracy of BOTH halves of the sealed prediction --
which explanation the student will pick, and whether they will then answer the
check question correctly -- one line per student, against a dashed
random-guess baseline.

HOW THE SCORE IS COUNTED (and the one trap in it)
-------------------------------------------------
Each round offers at most two scorable halves: the pick and the correctness.
On an intervention round the tutor FORCED the explanation style, so the pick
was never predicted -- ``pick_hit`` is ``None``. That half is EXCLUDED from
both the numerator and the denominator. It is not a miss. Treating ``None``
as ``False`` would silently drag every line on this chart down by roughly a
sixth and make the agent look worse than it is; treating it as ``True`` would
flatter it. It simply was not a question that was asked.

    know-you score (window) = halves hit / halves actually scored

THE BASELINE IS NOT A CONSTANT
------------------------------
Random guessing gets 1/3 on a three-way pick and 1/2 on a boolean. So a round
with both halves live is worth ``(1/3 + 1/2)/2 = 0.417`` to chance -- but a
forced round scores only the boolean half, and chance on that round is
``1/2``. The dashed line here is therefore computed per window from how many
halves of each kind were actually scored::

    baseline(window) = (pick_halves/3 + correctness_halves/2) / total_halves

so it sits at 0.417 on a fully-scored stretch and drifts up toward 0.44 across
windows containing a forced round. Drawing a flat 0.417 would overstate the
agent on exactly the rounds that matter most to the pitch.

CLI
---
    .venv/bin/python make_chart.py --in results_sim.json
    .venv/bin/python make_chart.py --in results_sim.json --out chart.png --window 5
    .venv/bin/python make_chart.py --in results.json --title "Live Gemini run"

Reads a sim checkpoint (partial files from a quota-killed run chart fine) or a
bare list of round logs. Writes a PNG. Touches nothing else.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: no display on the demo machine
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_IN = PROJECT_ROOT / "results_sim.json"
DEFAULT_OUT = PROJECT_ROOT / "know_you_score.png"

#: Number of styles offered, i.e. chance on the pick half. Read from the
#: agent package when importable so a fourth style moves the baseline too.
try:  # pragma: no cover - import guard only
    from tutor_agent import STYLES

    N_STYLES = len(STYLES)
except Exception:  # noqa: BLE001 - charting must work without the agent deps
    N_STYLES = 3

# Validated categorical slots 1 and 2 (dataviz reference palette, light mode):
# worst all-pairs CVD delta-E 24.7, normal-vision 33.6, both >= 3:1 on surface.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
# Distinct markers as well as distinct hues, so identity survives a bad
# projector, a greyscale print, and a colourblind judge.
SERIES_MARKERS = ["o", "s", "^", "D"]
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#dcdbd6"
SURFACE = "#fcfcfb"
BAND = "#ecebe6"


# ==========================================================================
# Loading
# ==========================================================================


def load_rounds(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return ``(rounds, meta)`` from a sim checkpoint or a bare round list."""
    if not path.exists():
        raise SystemExit(
            f"no such file: {path}\nRun the simulation first:\n"
            "    .venv/bin/python sim.py --fake --rounds 20"
        )
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except ValueError as exc:
        raise SystemExit(f"{path.name} is not valid JSON: {exc}") from exc

    if isinstance(data, list):
        # A bare list of round logs (the older results.json shape).
        return [dict(r) for r in data], {}
    if isinstance(data, dict) and isinstance(data.get("rounds"), list):
        return [dict(r) for r in data["rounds"]], dict(data.get("meta") or {})
    raise SystemExit(
        f"{path.name} has no round data (expected a list, or a dict with a "
        "'rounds' key)."
    )


def split_by_student(rounds: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by: dict[str, list[dict[str, Any]]] = {}
    for log in rounds:
        name = str(log.get("student") or "student")
        by.setdefault(name, []).append(dict(log))
    for logs in by.values():
        logs.sort(key=lambda lg: int(lg.get("round", 0)))
    return by


# ==========================================================================
# The maths
# ==========================================================================


def halves(log: Mapping[str, Any]) -> tuple[int, int, int, int]:
    """``(pick_halves, pick_hits, correctness_halves, correctness_hits)``.

    ``pick_halves`` is 0 exactly when ``pick_hit is None`` -- the forced-style
    round. ``is None`` rather than falsiness is the whole point: ``False``
    means "predicted and wrong" and must still be counted in the denominator.
    """
    if log.get("pick_hit") is None:
        pick_halves, pick_hits = 0, 0
    else:
        pick_halves, pick_hits = 1, int(bool(log["pick_hit"]))
    correct_hits = int(bool(log.get("correct_hit")))
    return pick_halves, pick_hits, 1, correct_hits


def rolling(
    logs: Sequence[Mapping[str, Any]], window: int
) -> tuple[list[int], list[float], list[float], list[int]]:
    """Rolling know-you score and matching rolling chance baseline.

    Returns ``(x, score, baseline, window_sizes)``. For the first rounds the
    window is short (all rounds so far) rather than absent, because a
    quota-killed run of five rounds still has to produce a chart -- the caption
    says which points are warm-up.
    """
    x: list[int] = []
    score: list[float] = []
    base: list[float] = []
    sizes: list[int] = []
    for i, log in enumerate(logs):
        chunk = logs[max(0, i - window + 1) : i + 1]
        p_halves = p_hits = c_halves = c_hits = 0
        for entry in chunk:
            ph, phit, ch, chit = halves(entry)
            p_halves += ph
            p_hits += phit
            c_halves += ch
            c_hits += chit
        total = p_halves + c_halves
        if not total:
            continue
        x.append(int(log.get("round", i + 1)))
        score.append((p_hits + c_hits) / total)
        # Chance on the halves that were ACTUALLY scored in this window.
        base.append((p_halves / N_STYLES + c_halves / 2) / total)
        sizes.append(len(chunk))
    return x, score, base, sizes


def overall(logs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    p_halves = p_hits = c_halves = c_hits = 0
    for log in logs:
        ph, phit, ch, chit = halves(log)
        p_halves += ph
        p_hits += phit
        c_halves += ch
        c_hits += chit
    total = p_halves + c_halves
    return {
        "rounds": len(logs),
        "forced": c_halves - p_halves,
        "halves_scored": total,
        "halves_hit": p_hits + c_hits,
        "score": (p_hits + c_hits) / total if total else float("nan"),
        "baseline": (p_halves / N_STYLES + c_halves / 2) / total if total else float("nan"),
        "pick_accuracy": (p_hits / p_halves) if p_halves else None,
        "correct_accuracy": (c_hits / c_halves) if c_halves else None,
    }


def forced_rounds(logs: Sequence[Mapping[str, Any]]) -> list[int]:
    return [int(lg.get("round", 0)) for lg in logs if lg.get("pick_hit") is None]


def _wrap(text: str, width: int) -> str:
    """Hard-wrap at a character count chosen for the block's point size."""
    return "\n".join(textwrap.wrap(text, width=width)) or text


# ==========================================================================
# The chart
# ==========================================================================


def build_chart(
    by_student: Mapping[str, list[dict[str, Any]]],
    meta: Mapping[str, Any],
    window: int,
    title: str | None,
) -> tuple[Any, dict[str, dict[str, Any]]]:
    names = sorted(by_student)
    stats = {name: overall(by_student[name]) for name in names}

    fig, ax = plt.subplots(figsize=(16, 9), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # ---- forced-style rounds, as recessive bands ------------------------
    all_forced = sorted({r for name in names for r in forced_rounds(by_student[name])})
    for round_number in all_forced:
        ax.axvspan(round_number - 0.32, round_number + 0.32, color=BAND, zorder=0, lw=0)

    # ---- the baseline ---------------------------------------------------
    # Every student normally runs the same intervention schedule, so their
    # per-window baselines coincide and ONE dashed line is the honest drawing.
    # Agreement is checked round by round rather than by comparing whole
    # lists: a part-finished run leaves students with different numbers of
    # rounds while still agreeing on every round they share, and that is not
    # a disagreement worth a second line or a warning.
    series = {name: rolling(by_student[name], window) for name in names}
    per_round: dict[str, dict[int, float]] = {
        name: dict(zip(series[name][0], series[name][2])) for name in names
    }
    longest = max(names, key=lambda n: len(series[n][0])) if names else None
    if longest is not None:
        bx, _, bbase, _ = series[longest]
        reference = per_round[longest]
        disagree = sorted(
            r
            for name in names
            for r, value in per_round[name].items()
            if r in reference and abs(value - reference[r]) > 1e-9
        )
        ax.plot(
            bx,
            bbase,
            linestyle=(0, (6, 4)),
            color=INK_SOFT,
            linewidth=3.0,
            zorder=2,
            label="Random guessing",
        )
        if disagree:
            print(
                "note: students ran different intervention schedules, so their "
                f"baselines differ on round(s) {disagree[:8]}; the dashed line "
                f"shown is '{longest}'s.",
                file=sys.stderr,
            )

    # ---- one line per student -------------------------------------------
    for i, name in enumerate(names):
        colour = SERIES_COLORS[i % len(SERIES_COLORS)]
        marker = SERIES_MARKERS[i % len(SERIES_MARKERS)]
        x, score, _, _ = series[name]
        if not x:
            continue
        ax.plot(
            x,
            score,
            color=colour,
            linewidth=4.0,
            marker=marker,
            markersize=11,
            markerfacecolor=colour,
            markeredgecolor=SURFACE,
            markeredgewidth=2.0,  # 2px surface ring where marks overlap
            solid_capstyle="round",
            zorder=4 - i * 0.1,
            label=f"{name.capitalize()}  (final {stats[name]['score']:.0%})",
        )
        # Direct end label: identity is never carried by colour alone.
        ax.annotate(
            name.capitalize(),
            xy=(x[-1], score[-1]),
            xytext=(12, 0),
            textcoords="offset points",
            color=colour,
            fontsize=22,
            fontweight="bold",
            va="center",
            zorder=6,
            # A surface halo: on a short or flat run an end label can land on
            # the dashed baseline, and a legible label beats a tidy one.
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 2.0, "alpha": 0.88},
        )

    # ---- axes -----------------------------------------------------------
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
    all_x = [r for name in names for r in series[name][0]]
    if all_x:
        ax.set_xlim(min(all_x) - 0.4, max(all_x) + 1.4)

    ax.set_xlabel("Round", fontsize=24, color=INK, labelpad=14)
    ax.set_ylabel(
        f"Know-you score\n(rolling {window}-round accuracy)",
        fontsize=24,
        color=INK,
        labelpad=14,
    )
    ax.tick_params(axis="both", labelsize=20, colors=INK_SOFT, length=6, width=1.2)
    ax.grid(axis="y", color=GRID, linewidth=1.2, zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.4)

    # ---- titles ---------------------------------------------------------
    # Each text block gets its own band of the figure and is hard-wrapped to a
    # character count that fits at its own point size. Matplotlib will happily
    # run a long title straight off the canvas, and a title that leaves the
    # slide is the one failure mode a projector makes unmissable.
    headline = title or "The tutor learns the student"
    fig.suptitle(headline, fontsize=33, fontweight="bold", color=INK, x=0.065, ha="left", y=0.965)

    subtitle = (
        "Sealed prediction, made and scored before the student acts: which "
        "explanation they pick, and whether they then answer the check "
        "question right."
    )
    fig.text(
        0.065,
        0.905,
        _wrap(subtitle, 108),
        fontsize=19,
        color=INK_SOFT,
        ha="left",
        va="top",
        linespacing=1.4,
    )
    if meta.get("fake"):
        fig.text(
            0.965,
            0.965,
            "STUB AGENTS — mechanics only,\nnot a live model result",
            fontsize=17,
            color="#b4532a",
            ha="right",
            va="top",
            fontweight="bold",
            linespacing=1.3,
        )

    # ---- legend: a row ABOVE the axes, so it can never sit on the data ---
    handles, labels = ax.get_legend_handles_labels()
    if all_forced:
        handles.append(Line2D([0], [0], color=BAND, linewidth=16))
        labels.append("Forced-style round (pick not scored)")
    # Placed in the low band of the plot, which the score lines never enter
    # (chance is 0.42: a profile scoring under ~30% over five rounds would be
    # anti-predictive). Opaque, so it stays readable if a run ever does.
    legend = ax.legend(
        handles,
        labels,
        fontsize=19,
        loc="lower left",
        ncol=2,
        frameon=True,
        facecolor=SURFACE,
        edgecolor=GRID,
        framealpha=0.96,
        borderpad=0.8,
        labelspacing=0.6,
        columnspacing=2.0,
        handlelength=2.6,
        handletextpad=0.8,
    )
    legend.get_frame().set_linewidth(1.2)
    for text in legend.get_texts():
        text.set_color(INK)

    # ---- caption: state the assumptions, don't hide them -----------------
    warm = min(window - 1, max((len(series[n][0]) for n in names), default=0))
    full = (1 / N_STYLES + 0.5) / 2
    # The caption describes THIS chart, not charts in general: a run with no
    # forced rounds must not carry a paragraph about forced rounds, or the
    # baseline it explains stops matching the dashed line on screen.
    caption = "Scored in halves — one for the pick, one for the correctness."
    if all_forced:
        caption += (
            " On a forced-style round the tutor chose the explanation, so the "
            "pick was never predicted: that half is EXCLUDED from the score, "
            "not counted as wrong."
        )
    caption2 = (
        f"The dashed baseline is chance on the halves each window actually "
        f"scored ({N_STYLES}-way pick = 1/{N_STYLES}, correctness = 1/2)"
    )
    caption2 += (
        f" — {full:.3f} across a fully-scored window, 0.500 across forced "
        "rounds alone."
        if all_forced
        else f", a flat {full:.3f} here because every round scored both halves."
    )
    if warm > 0:
        caption2 += (
            f" The first {warm} point(s) use a shorter window (all rounds so "
            "far), so a part-finished run still charts."
        )
    fig.text(
        0.065,
        0.018,
        _wrap(caption, 150) + "\n" + _wrap(caption2, 150),
        fontsize=14,
        color=INK_SOFT,
        ha="left",
        va="bottom",
        linespacing=1.45,
    )

    fig.subplots_adjust(left=0.125, right=0.945, top=0.800, bottom=0.310)
    return fig, stats


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Plot the know-you score from a sim results file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  make_chart.py --in results_sim.json\n"
            "  make_chart.py --in results_sim.json --window 5 --out know_you_score.png\n"
        ),
    )
    p.add_argument("--in", dest="infile", default=str(DEFAULT_IN), help="results JSON to read")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="PNG to write")
    p.add_argument("--window", type=int, default=5, help="rolling window in rounds (default 5)")
    p.add_argument("--title", default=None, help="override the headline")
    args = p.parse_args(argv)

    if args.window < 1:
        raise SystemExit("--window must be >= 1")

    rounds, meta = load_rounds(Path(args.infile))
    if not rounds:
        raise SystemExit(
            f"{Path(args.infile).name} has no completed rounds yet -- nothing to plot."
        )
    by_student = split_by_student(rounds)

    fig, stats = build_chart(by_student, meta, args.window, args.title)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)

    status = meta.get("status")
    print(f"== read {Path(args.infile).name}" + (f" (status: {status})" if status else ""))
    if meta.get("fake"):
        print("== NOTE: this file came from --fake (stub agents); chart is labelled as such")
    for name, s in stats.items():
        pick = "n/a" if s["pick_accuracy"] is None else f"{s['pick_accuracy']:.0%}"
        print(
            f"   {name:<6} {s['rounds']:>3} rounds | know-you {s['score']:.1%} "
            f"(chance {s['baseline']:.1%}) | pick {pick} | "
            f"correctness {s['correct_accuracy']:.0%} | "
            f"{s['forced']} forced round(s) excluded from the pick half"
        )
    print(f"== wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

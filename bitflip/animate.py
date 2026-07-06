"""Two polished animations (MP4 primary, looping GIF secondary).

Animation A (``anim_round.mp4``): a walkthrough of one correction round in
three verified scenarios - a clean correction, a misleading report that
causes a logical failure, and a false report that injects a residual error
while the logical bit survives. Every number shown is derived through
``bitflip.scenarios`` (tested against the 32-event oracle).

Animation B (``anim_sweep.mp4``): the p-sweep. Curves are drawn from the
exact oracle, Monte Carlo estimates with Wilson 95% intervals come from the
same CSV records that produce the comparison figure. The animation pauses at
p = 0.1 on verified numbers and ends on the p = 0.5 boundary check.

Both are 12 fps, 960x540. Colors are the shared Okabe-Ito palette from
``bitflip.plotting`` with text labels alongside color throughout. Frame
timing and content are verified in ``results/animation_verification.json``.
"""

from __future__ import annotations

import hashlib
import json
import textwrap
from pathlib import Path

import imageio
import imageio.v2 as iio_v2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from .enumeration import exact_rates
from .plotting import COLOR_CLEAN, COLOR_NOISY, COLOR_UNENCODED
from .scenarios import scenarios

FPS_DEFAULT = 12
SIZE = (9.6, 5.4)  # inches at dpi=100 -> 960x540 px
COLOR_SUCCESS = "#009E73"
COLOR_FAIL = "#D55E00"
COLOR_HIGHLIGHT = "#E69F00"
COLOR_CORRECTION = "#0072B2"
COLOR_TEXT = "#222222"

_LOGICAL = "logical_failure"


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

def _new_fig():
    fig = plt.figure(figsize=SIZE, dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 9.6)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    return fig, ax


def _grab(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_frames(frames: list[np.ndarray], mp4_path: Path, gif_path: Path, fps: int):
    """Write an MP4 (ffmpeg via imageio-ffmpeg) and a looping GIF.

    The GIF is written with Pillow directly: imageio's GIF writer
    deduplicates identical consecutive frames and drops per-frame delays,
    which corrupts held frames (e.g. pauses for prediction). A global
    256-colour palette (computed from a sample of frames, no dithering)
    keeps text crisp and colors stable.
    """
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    with iio_v2.get_writer(
        mp4_path, fps=fps, codec="libx264", quality=8,
        pixelformat="yuv420p", macro_block_size=2,
    ) as writer:
        for frame in frames:
            writer.append_data(frame)
    _write_gif(frames, gif_path, fps)


def _write_gif(frames: list[np.ndarray], gif_path: Path, fps: int):
    sample = frames[:: max(1, len(frames) // 8)]
    pixels = np.concatenate([f.reshape(-1, 3) for f in sample])
    palette = Image.fromarray(pixels.reshape(1, -1, 3)).quantize(colors=256)
    pil_frames = [
        Image.fromarray(frame).quantize(palette=palette, dither=Image.Dither.NONE)
        for frame in frames
    ]
    pil_frames[0].save(
        gif_path, save_all=True, append_images=pil_frames[1:],
        duration=int(round(1000 / fps)), loop=0, disposal=2, optimize=False,
    )


def _probe(mp4_path: Path, gif_path: Path, fps: int) -> dict:
    """Verify frame counts and real durations; used for the audit record."""
    reader = iio_v2.get_reader(mp4_path)
    n_mp4 = reader.count_frames()
    reader.close()
    with Image.open(gif_path) as gif:
        n_gif = gif.n_frames
        gif_ms = 0
        for i in range(n_gif):
            gif.seek(i)
            gif_ms += int(gif.info.get("duration") or 0)
    return {
        "mp4_frames": int(n_mp4),
        "mp4_duration_s": round(n_mp4 / fps, 3),
        "gif_frames": int(n_gif),
        "gif_duration_s": round(gif_ms / 1000.0, 3),
    }


def _export_previews(mp4_path: Path, frames: list[np.ndarray], picks: dict[str, int],
                     stem: str, outdir: Path):
    """Static previews (opening, transition, final) as PNG files."""
    out = []
    for name, idx in picks.items():
        idx = min(idx, len(frames) - 1)
        path = outdir / f"preview_{stem}_{name}.png"
        Image.fromarray(frames[idx]).save(path)
        out.append(path)
    return out


def _merge_record(outdir: Path, info: dict):
    """Add one animation's verification entry, preserving the others."""
    record_path = outdir / "animation_verification.json"
    entries = []
    if record_path.exists():
        old = json.loads(record_path.read_text())
        entries = [e for e in old.get("animations", [])
                   if not e["name"].startswith(info["name"].split(" ")[0])]
    entries.append(info)
    _record(outdir, entries)


def _record(outdir: Path, entries: list[dict]):
    record = {
        "fps": FPS_DEFAULT,
        "size_px": [960, 540],
        "note": "Frame counts and durations probed from the written files; "
                "sha256 identifies the exact delivered artifacts.",
        "animations": entries,
    }
    (outdir / "animation_verification.json").write_text(json.dumps(record, indent=2))


# --------------------------------------------------------------------------
# animation A: one-round walkthrough
# --------------------------------------------------------------------------

def _draw_qubits(ax, bits, error_idx, corrected_idx, show_correction):
    for i in range(3):
        x = 1.1 + 1.15 * i
        circ = plt.Circle((x, 3.55), 0.42, facecolor="white",
                          edgecolor=COLOR_TEXT, linewidth=1.6)
        ax.add_patch(circ)
        ax.text(x, 3.55, str(bits[i]), ha="center", va="center", fontsize=20)
        ax.text(x, 4.25, f"q{i + 1}", ha="center", fontsize=11, color="#555555")
        if error_idx == i:
            ax.add_patch(plt.Circle((x, 3.55), 0.52, facecolor="none",
                                    edgecolor=COLOR_FAIL, linewidth=2.6))
            ax.text(x - 0.55, 4.62, "X error", ha="center", fontsize=10.5,
                    color=COLOR_FAIL, fontweight="bold")
        if show_correction and corrected_idx == i:
            ax.annotate("", xy=(x + 0.30, 4.02), xytext=(x + 0.30, 4.85),
                        arrowprops=dict(arrowstyle="-|>", color=COLOR_CORRECTION, lw=2.4))
            ax.text(x + 0.52, 4.72, "correction X", fontsize=9.5, color=COLOR_CORRECTION,
                    ha="left")


def _draw_parities(ax, state):
    labels = ["s1 = q1⊕q2", "s2 = q2⊕q3"]
    for j in range(2):
        x, y = 0.95 + 2.35 * j, 1.85
        flipped = state.get("show_reported") and state["noise_bit"] == j
        face = "#FDEBD0" if flipped else "#Eef2f7"
        box = plt.Rectangle((x - 0.55, y - 0.42), 1.55, 1.25,
                            facecolor=face, edgecolor="#888888")
        ax.add_patch(box)
        ax.text(x + 0.22, y + 0.95, labels[j], ha="center", fontsize=10, color="#555555")
        if state.get("show_true"):
            ax.text(x - 0.22, y - 0.08, str(state["true_syndrome"][j]),
                    ha="center", fontsize=15, color=COLOR_TEXT)
            ax.text(x - 0.22, y - 0.34, "true", ha="center", fontsize=8, color="#777777")
        if state.get("show_reported"):
            ax.text(x + 0.62, y - 0.08, str(state["reported_syndrome"][j]),
                    ha="center", fontsize=15,
                    color=COLOR_HIGHLIGHT if flipped else COLOR_TEXT,
                    fontweight="bold" if flipped else "normal")
            ax.text(x + 0.62, y - 0.34, "reported", ha="center", fontsize=8,
                    color=COLOR_HIGHLIGHT if flipped else "#777777")
        if flipped:
            ax.text(x + 0.40, y + 0.45, "readout flip!", ha="center", fontsize=8,
                    color=COLOR_HIGHLIGHT, fontweight="bold")


def _status_lines(state):
    def fmt_mask(m):
        return "".join(str(b) for b in m) if m is not None else "-"

    lines = [
        ("Physical error mask", fmt_mask(state.get("error_mask")) if state.get("show_error") else None),
        ("True syndrome", fmt_mask(state.get("true_syndrome")) if state.get("show_true") else None),
        ("Reported syndrome", fmt_mask(state.get("reported_syndrome")) if state.get("show_reported") else None),
        ("Correction chosen",
         (f"X on q{state['correction_qubit'] + 1}" if state.get("correction_qubit") is not None else "none")
         if state.get("show_correction") else None),
        ("Residual mask", fmt_mask(state.get("final_block")) if state.get("show_residual") else None),
    ]
    return lines


def _draw_status(ax, state):
    x0 = 4.85
    ax.text(x0, 4.62, "Round status", fontsize=12, fontweight="bold", color=COLOR_TEXT)
    y = 4.2
    for label, value in _status_lines(state):
        ax.text(x0, y, label, fontsize=10.5, color="#555555", va="center")
        if value is not None:
            ax.text(x0 + 2.6, y, value, fontsize=11.5, color=COLOR_TEXT,
                    va="center", family="monospace")
        y -= 0.42

    if state.get("show_outcome"):
        outcome = state["outcome"]
        if outcome == "clean_success":
            msg, color = "LOGICAL 0 - corrected, SUCCESS ✓", COLOR_SUCCESS
        elif outcome == "residual_only":
            msg, color = "LOGICAL 0 ✓ but residual X\n(outside code space)", COLOR_HIGHLIGHT
        else:
            msg, color = "LOGICAL 1 - LOGICAL FAILURE ✗", COLOR_FAIL
        ax.text(4.85, 1.62, msg, fontsize=11.5, color=color, fontweight="bold",
                va="center", linespacing=1.3)

    if state.get("predict_prompt"):
        ax.text(4.85, 1.02, "▶ Predict before it happens:", fontsize=11,
                color="#8a6d00", fontweight="bold")
        ax.text(4.85, 0.62, state["predict_prompt"], fontsize=10.5,
                color="#8a6d00", style="italic")


def _draw_caption(ax, state):
    ax.text(0.25, 5.12, state["title"], fontsize=14, fontweight="bold", color=COLOR_TEXT)
    wrapped = "\n".join(textwrap.wrap(state["caption"], width=88))
    ax.text(0.25, 0.24, wrapped, fontsize=11, color="#444444", va="bottom")


def _walkthrough_state(frame_idx: int, data: dict):
    """Map a frame index to the walkthrough scene state (12 fps timeline)."""
    S = data
    t = frame_idx  # frames
    st = {
        "title": "Three-qubit bit-flip code - one correction round",
        "caption": "", "show_error": False, "show_true": False,
        "show_reported": False, "show_correction": False, "show_residual": False,
        "show_outcome": False, "predict_prompt": None, "scenario": None,
        "error_mask": None, "true_syndrome": None, "reported_syndrome": None,
        "noise_bit": None, "correction_qubit": None, "final_block": None,
        "outcome": None,
    }
    if t < 27:  # intro 0-2.25 s
        st["caption"] = ("Three data qubits encode a logical bit as 000 (logical |0⟩) "
                         "or 111. Today: X (bit-flip) errors only.")
        st["error_mask"], st["true_syndrome"], st["reported_syndrome"] = (0, 0, 0), (0, 0), (0, 0)
        return st

    # scenario boundaries in frames
    bounds = [(27, 111, "clean"), (111, 195, "misleading"), (195, 267, "false_report")]
    for start, end, name in bounds:
        if start <= t < end:
            s = S[name]
            st["scenario"] = name
            st["error_mask"] = s["error_mask"]
            st["true_syndrome"] = s["true_syndrome"]
            st["reported_syndrome"] = s["reported_syndrome"]
            st["noise_bit"] = s["noise_bit"]
            st["correction_qubit"] = s["correction_qubit"]
            st["final_block"] = s["final_block"]
            st["outcome"] = s["outcome"]
            k = t - start
            break
    else:  # outro 267-311
        st["caption"] = ("Logical failure = ideal majority returns the wrong bit. "
                         "A residual X is physical damage the majority can hide.")
        st["error_mask"], st["true_syndrome"], st["reported_syndrome"] = (0, 0, 0), (0, 0), (0, 0)
        st["show_outcome"] = False
        return st

    st["show_error"] = k >= 0
    if name == "clean":
        st["caption"] = "Scenario 1 - a single physical error, perfect parity readout (q = 0)."
        st["show_true"] = k >= 12
        st["show_reported"] = k >= 24
        st["predict_prompt"] = ("True parity is 11. Which qubit will the decoder correct?"
                                if 36 <= k < 48 else None)
        st["show_correction"] = k >= 48
        st["show_residual"] = k >= 48
        st["show_outcome"] = k >= 60
    elif name == "misleading":
        st["caption"] = ("Scenario 2 - a single physical error on qubit 1 (the mirror "
                         "image of scenario 1), but readout noise (q = 0.02) flips the "
                         "second reported parity.")
        st["show_true"] = k >= 12
        st["show_reported"] = k >= 24
        st["predict_prompt"] = ("The decoder trusts the report. Where does its correction land?"
                                if 36 <= k < 48 else None)
        st["show_correction"] = k >= 48
        st["show_residual"] = k >= 48
        st["show_outcome"] = k >= 60
    else:
        st["caption"] = "Scenario 3 - no physical error, but a false report."
        st["show_true"] = k >= 12
        st["show_reported"] = k >= 24
        st["predict_prompt"] = ("The block is clean. Can the correction itself damage it?"
                                if 36 <= k < 48 else None)
        st["show_correction"] = k >= 48
        st["show_residual"] = k >= 48
        st["show_outcome"] = k >= 60
    return st


def _draw_outro_numbers(ax):
    rates = exact_rates(0.1, 0.02)
    ax.text(0.25, 4.7, "Verified numbers for this model at p = 0.1, q = 0.02",
            fontsize=12, fontweight="bold")
    rows = [
        ("Logical failure (wrong majority)", f"{rates['logical_failure']:.5%}"),
        ("Any residual physical X", f"{rates['any_residual_x']:.5%}"),
        ("Corrected block outside code space", f"{rates['outside_code_space']:.5%}"),
        ("Unencoded single bit", f"{0.1:.5%}"),
    ]
    y = 4.25
    for label, val in rows:
        ax.text(0.55, y, label, fontsize=11, color="#444444")
        ax.text(4.1, y, val, fontsize=11, family="monospace", color=COLOR_TEXT)
        y -= 0.38


def render_animation_a(outdir: Path, fps: int = FPS_DEFAULT,
                       frame_indices=None) -> list[Path]:
    data = scenarios()
    frames = []
    n_frames = 312  # 26.0 s at 12 fps
    indices = list(frame_indices) if frame_indices is not None else range(n_frames)
    for idx in indices:
        fig, ax = _new_fig()
        if idx >= 267:
            # Outro: clean summary frame, no qubit layout underneath.
            st = _walkthrough_state(idx, data)
            _draw_outro_numbers(ax)
            _draw_caption(ax, st)
        else:
            st = _walkthrough_state(idx, data)
            _draw_qubits(ax, st["error_mask"] or (0, 0, 0),
                         _error_idx(st), st["correction_qubit"],
                         st["show_correction"])
            _draw_parities(ax, st)
            _draw_status(ax, st)
            _draw_caption(ax, st)
        frames.append(_grab(fig))
        plt.close(fig)

    mp4 = outdir / "anim_round.mp4"
    gif = outdir / "anim_round.gif"
    _save_frames(frames, mp4, gif, fps)
    info = _probe(mp4, gif, fps)
    info.update({
        "name": "anim_round - one-round walkthrough",
        "mp4": str(mp4), "gif": str(gif),
        "mp4_sha256": _sha256(mp4), "gif_sha256": _sha256(gif),
        "content": "three verified scenarios: clean correction (mask 010, reported 11), "
                   "misleading report (mask 100, reported 11 -> residual 110, logical failure), "
                   "false report (mask 000, reported 01 -> residual 001, logical survives)",
        "previews": [str(p) for p in _export_previews(
            mp4, frames,
            {"opening": 0, "clean_correction": 84, "misleading_report": 147,
             "misleading_outcome": 190, "false_report_outcome": 260, "final": 311},
            "round", outdir)],
    })
    _merge_record(outdir, info)
    return [mp4, gif]


def _error_idx(state) -> int | None:
    if not state.get("show_error") or state.get("error_mask") is None:
        return None
    m = state["error_mask"]
    return m.index(1) if 1 in m else None


# --------------------------------------------------------------------------
# animation B: the p-sweep, driven from the CSV records
# --------------------------------------------------------------------------

def _load_sweep(csv_path: Path | None, outdir: Path) -> pd.DataFrame:
    """Load the sweep records; compute and save them if the CSV is missing."""
    if csv_path is not None and Path(csv_path).exists():
        return pd.read_csv(csv_path)
    from .cli import DEFAULT_SEED, P_CHECK, P_SWEEP, Q_VALUES
    from .simulation import run_sweep, sweep_frame

    frame = sweep_frame(run_sweep(P_SWEEP + P_CHECK, Q_VALUES, shots=None, seed=DEFAULT_SEED))
    outdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(outdir / "sweep.csv", index=False)
    return frame


def _sweep_frame_state(idx: int):
    """Timeline for animation B: intro 24 | rise 66 | pause 36 | rise 138 | tail 72.

    Total 336 frames (28 s at 12 fps). The sweep rises 0 -> 0.1, pauses at
    the verified p = 0.1 example, continues to 0.31, then ends on the
    p = 0.5 boundary check.
    """
    intro, pause, tail = 24, 36, 72
    motion = 336 - intro - pause - tail  # 204 frames across p in [0, 0.31]
    rise1 = round(motion * 0.1 / 0.31)   # frames for 0 -> 0.1

    if idx < intro:
        return {"p_hi": 0.0, "paused": False, "final": False}
    pos = idx - intro
    if pos < rise1:
        return {"p_hi": 0.1 * pos / rise1, "paused": False, "final": False}
    if pos < rise1 + pause:
        return {"p_hi": 0.1, "paused": True, "final": False}
    if pos < motion + pause:
        done = (pos - rise1 - pause) / (motion - rise1)
        return {"p_hi": 0.1 + 0.21 * done, "paused": False, "final": False}
    return {"p_hi": 0.31, "paused": False, "final": True}


def _partial_segment(data: pd.DataFrame, p_hi: float, column: str):
    """Records up to p_hi plus an interpolated leading edge, so every frame
    advances. Driven by the CSV records that generate the figure."""
    sub = data[data["p"] <= p_hi]
    xs = sub["p"].to_numpy()
    ys = sub[column].to_numpy()
    if len(xs) and xs[-1] < p_hi:
        xs = np.append(xs, p_hi)
        ys = np.append(ys, np.interp(p_hi, data["p"], data[column]))
    return xs, ys


def _draw_sweep_axes(ax, frame, p_hi, paused, final):
    ax.set_xlim(-0.01, 0.34)
    ax.set_ylim(-0.012, 0.36)
    ax.set_xlabel("Physical bit-flip probability p", fontsize=12)
    ax.set_ylabel("Logical failure probability", fontsize=12)
    ax.grid(alpha=0.25)

    clean = frame[np.isclose(frame["q"], 0.0) & (frame["p"] <= 0.31)].sort_values("p")
    noisy = frame[np.isclose(frame["q"], 0.02) & (frame["p"] <= 0.31)].sort_values("p")

    xs, ys = _partial_segment(clean, p_hi, f"{_LOGICAL}_exact")
    ax.plot(xs, ys, color=COLOR_CLEAN, lw=2.2, label="encoded, exact, q = 0")
    xs, ys = _partial_segment(noisy, p_hi, f"{_LOGICAL}_exact")
    ax.plot(xs, ys, color=COLOR_NOISY, lw=2.2, label="encoded, exact, q = 0.02")
    xs, ys = _partial_segment(clean, p_hi, "unencoded_failure_exact")
    ax.plot(xs, ys, "--", color=COLOR_UNENCODED, lw=2.0, label="unencoded: fails with prob. p")

    # Monte Carlo estimates with Wilson intervals, revealed as p passes them.
    for data, color, marker in ((clean, COLOR_CLEAN, "o"), (noisy, COLOR_NOISY, "s")):
        sub = data[(data["p"] > 0) & (data["p"] <= p_hi)]
        if len(sub):
            y = sub[_LOGICAL].to_numpy()
            yerr = [y - sub[f"{_LOGICAL}_ci_low"].to_numpy(),
                    sub[f"{_LOGICAL}_ci_high"].to_numpy() - y]
            ax.errorbar(sub["p"].to_numpy(), y, yerr=yerr,
                        fmt=marker, color=color, markersize=3.4, elinewidth=0.9,
                        capsize=1.6, linestyle="none", alpha=0.85,
                        label=f"Monte Carlo {marker}, Wilson 95% CI")

    if paused:
        row = clean[np.isclose(clean["p"], 0.1)].iloc[0]
        nrow = noisy[np.isclose(noisy["p"], 0.1)].iloc[0]
        ax.axvline(0.1, color="#bbbbbb", lw=1.0, linestyle=":")
        txt = (
            "Verified pause at p = 0.1\n"
            f"unencoded:            10.00%\n"
            f"exact, q = 0:         {row[f'{_LOGICAL}_exact']:.2%}\n"
            f"MC,      q = 0:       {row[_LOGICAL]:.2%} "
            f"[{row[f'{_LOGICAL}_ci_low']:.2%}, {row[f'{_LOGICAL}_ci_high']:.2%}]\n"
            f"exact, q = 0.02:      {nrow[f'{_LOGICAL}_exact']:.2%}\n"
            f"MC,      q = 0.02:    {nrow[_LOGICAL]:.2%} "
            f"[{nrow[f'{_LOGICAL}_ci_low']:.2%}, {nrow[f'{_LOGICAL}_ci_high']:.2%}]"
        )
        ax.text(0.135, 0.02, txt, fontsize=9.5, family="monospace",
                bbox=dict(boxstyle="round", fc="white", ec="#999999"))

    if final:
        check = frame[np.isclose(frame["p"], 0.5)]
        if len(check) >= 2:
            c05 = check[check["q"] == 0.0].iloc[0]
            n05 = check[check["q"] == 0.02].iloc[0]
            ax.text(0.01, 0.02,
                    f"p = 0.5 boundary check (from the same records):\n"
                    f"unencoded {float(c05['unencoded_failure']):.1%}, "
                    f"q = 0: {float(c05[_LOGICAL]):.1%}, q = 0.02: {float(n05[_LOGICAL]):.1%} "
                    "- code and unencoded both sit at 50%",
                    transform=ax.transAxes, fontsize=9.5, va="bottom",
                    bbox=dict(boxstyle="round", fc="#FFF8E7", ec=COLOR_HIGHLIGHT))
        else:
            ax.text(0.01, 0.02,
                    "p = 0.5 boundary check not in this data\n"
                    "(simulate without --no-check to include it)",
                    transform=ax.transAxes, fontsize=9, va="bottom",
                    color="#777777")


def render_animation_b(outdir: Path, fps: int = FPS_DEFAULT,
                       csv_path: Path | None = None,
                       frame_indices=None) -> list[Path]:
    """Render the p-sweep animation.

    ``frame_indices`` optionally restricts which timeline frames are
    rendered (used by the test suite to exercise the final frame cheaply);
    by default all 336 frames (28 s at 12 fps) are produced.
    """
    from matplotlib.lines import Line2D

    frame = _load_sweep(csv_path, outdir)
    frames = []
    n_frames = 336  # 28.0 s at 12 fps
    indices = list(frame_indices) if frame_indices is not None else range(n_frames)
    for idx in indices:
        state = _sweep_frame_state(idx)
        fig = plt.figure(figsize=SIZE, dpi=100)
        ax = fig.add_axes((0.09, 0.155, 0.88, 0.745))
        _draw_sweep_axes(ax, frame, state["p_hi"], state["paused"], state["final"])
        fig.suptitle("Three-qubit repetition code: logical failure vs physical error rate",
                     fontsize=13, fontweight="bold", y=0.97)
        proxies = [
            Line2D([], [], color=COLOR_CLEAN, lw=2.2, label="encoded, exact, q = 0"),
            Line2D([], [], color=COLOR_NOISY, lw=2.2, label="encoded, exact, q = 0.02"),
            Line2D([], [], color=COLOR_UNENCODED, ls="--", lw=2.0,
                   label="unencoded: fails with prob. p"),
            Line2D([], [], marker="o", color=COLOR_CLEAN, markersize=4.5,
                   linestyle="none", label="Monte Carlo q = 0, Wilson 95% CI"),
            Line2D([], [], marker="s", color=COLOR_NOISY, markersize=4.5,
                   linestyle="none", label="Monte Carlo q = 0.02, Wilson 95% CI"),
        ]
        ax.legend(handles=proxies, loc="upper left", fontsize=8.5, framealpha=0.95)
        if state["paused"]:
            caption = ("Pause at p = 0.1: encoding cuts logical failures from 10% (unencoded) "
                       "to 2.8%; a 2% readout flip rate raises it to 3.37%.")
        elif state["final"]:
            caption = ("Full sweep: Monte Carlo tracks the exact curves within statistical "
                       "error. Adaptive shot budget, at least 100,000 shots per point.")
        else:
            caption = ("One correction round: independent X errors (p) and noisy parity "
                       "reports (q). Curves: exact 32-event enumeration; markers: seeded "
                       "Monte Carlo, appearing as p passes each estimate.")
        wrapped = "\n".join(textwrap.wrap(caption, width=118))
        fig.text(0.02, 0.005, wrapped, fontsize=10.5, color="#444444", va="bottom")
        frames.append(_grab(fig))
        plt.close(fig)

    mp4 = outdir / "anim_sweep.mp4"
    gif = outdir / "anim_sweep.gif"
    _save_frames(frames, mp4, gif, fps)
    info = _probe(mp4, gif, fps)
    info.update({
        "name": "anim_sweep - p-sweep with Monte Carlo estimates",
        "mp4": str(mp4), "gif": str(gif),
        "mp4_sha256": _sha256(mp4), "gif_sha256": _sha256(gif),
        "content": "driven by results/sweep.csv (same records as the comparison figure); "
                   "pause at p=0.1 with verified numbers; final frame includes the p=0.5 check",
        "previews": [str(p) for p in _export_previews(
            mp4, frames,
            {"opening": 0, "pause_p01": 110, "final": 335},
            "sweep", outdir)],
    })
    # merge with animation A's record if present
    _merge_record(outdir, info)
    return [mp4, gif]

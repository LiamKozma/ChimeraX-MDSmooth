"""
RMSD plot for the MDSmooth bundle.

``build_figure`` builds a matplotlib Figure from a :class:`filter.FilterResult`
and has no Qt or ChimeraX dependency, so it can be rendered to a PNG in a
headless test. ``MDSmoothPlot`` embeds that figure in a ChimeraX tool window so the
plot pops up next to the structure.
"""

import numpy as np

# A flat, modern palette: faint raw trace, one confident "hero" line for the
# filtered signal, and two accent dot colors for the extrema.
_BG_COLOR = "#ffffff"        # clean white canvas
_RAW_COLOR = "#c3c9d4"       # faint cool grey: noisy, unfiltered RMSD
_FILTERED_COLOR = "#4f46e5"  # indigo: the zero-phase filtered RMSD (the hero)
_MAX_COLOR = "#ef4444"       # red dots: maxima
_MIN_COLOR = "#0ea5e9"       # sky-blue dots: minima
_USER_COLOR = "#16a34a"      # green diamonds: user-added significant frames
_PREVIEW_COLOR = "#16a34a"   # green: the not-yet-committed add-frame preview
_INK = "#1f2937"             # near-black for the title
_MUTED = "#6b7280"           # grey for labels, ticks, secondary text
_GRID = "#eceef2"            # very light gridlines
_SPINE = "#d5d8df"           # soft axis lines


def _draw_spectrum(ax, result, sampling_rate=1.0):
    """Draw the RMSD power spectrum with the chosen cutoff marked.

    An optional diagnostic for choosing the cutoff by eye: the low-frequency
    structured content (real conformational motion) usually sits above a flatter
    high-frequency shelf (thermal jitter), and the cutoff line shows where the
    filter currently splits them.  Uses the same :func:`power_spectrum` numbers
    the automatic selection does, so the picture and the math agree.
    """
    try:
        from .filter import power_spectrum
    except ImportError:  # headless test: src on sys.path, not imported as a pkg
        from filter import power_spectrum

    freqs, power, _ = power_spectrum(result.raw, sampling_rate)
    # Skip the DC bin (freq 0); the mean was removed, so it carries no signal.
    f, p = freqs[1:], np.clip(power[1:], 1e-12, None)

    ax.set_facecolor(_BG_COLOR)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=_GRID, linewidth=1.0)
    if f.size:
        ax.semilogy(f, p, color=_RAW_COLOR, linewidth=1.0, zorder=2,
                    label="Power spectrum")
    ax.axvline(result.cutoff_frequency, color=_FILTERED_COLOR, linewidth=1.6,
               zorder=3, label="cutoff")
    # Caption sits above the spectrum panel (not on the curve) so it never
    # overlaps the data or the cutoff line.
    ax.annotate(
        "cutoff %.4g cycles/frame" % result.cutoff_frequency,
        xy=(0, 1), xycoords="axes fraction", xytext=(0, 6),
        textcoords="offset points", ha="left", va="bottom",
        fontsize=9, color=_FILTERED_COLOR)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_SPINE)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=_MUTED, labelsize=8, length=0)
    ax.set_xlabel("Frequency (cycles/frame)", color=_MUTED, fontsize=9)
    ax.set_ylabel("Power", color=_MUTED, fontsize=9)
    ax.margins(x=0.01)


def build_figure(result, figure=None, show_spectrum=False, sampling_rate=1.0,
                 signal_label="RMSD"):
    """Return a matplotlib Figure comparing the raw and filtered signal.

    Parameters
    ----------
    result : filter.FilterResult
        Output of :func:`filter.filter_rmsd`.
    figure : matplotlib.figure.Figure, optional
        Draw into this figure (used when embedding in a Qt canvas). If omitted a
        new standalone Figure is created.
    show_spectrum : bool
        Add a second panel below the trace showing the power spectrum with the
        chosen cutoff marked, an optional aid for picking the cutoff.
    sampling_rate : float
        Passed to :func:`power_spectrum` for the spectrum panel's frequency axis.
    signal_label : str
        Names the filtered signal ("RMSD" or "PC1"); sets the y-axis label and
        title so a PC1 plot doesn't claim to show RMSD.
    """
    if figure is None:
        from matplotlib.figure import Figure
        figure = Figure(figsize=(7.5, 4.0), tight_layout=True)

    figure.set_facecolor(_BG_COLOR)
    if show_spectrum:
        ax, ax_spec = figure.subplots(
            2, 1, gridspec_kw={"height_ratios": [3, 1]})
    else:
        ax = figure.add_subplot(111)
        ax_spec = None
    ax.set_facecolor(_BG_COLOR)

    frames = result.frames
    kinds = result.kinds
    maxima = [f for f, k in zip(frames, kinds) if k == "max"]
    minima = [f for f, k in zip(frames, kinds) if k == "min"]
    user = [f for f, k in zip(frames, kinds) if k == "user"]

    n = len(result.filtered)
    x = range(n)

    # Horizontal-only grid, sitting behind everything.
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=_GRID, linewidth=1.0)

    # Raw trace: faint, thin, in the background.
    ax.plot(x, result.raw, color=_RAW_COLOR, linewidth=0.9, alpha=0.9,
            zorder=1, label="Unfiltered")
    # Filtered trace: the hero line, with a soft fill beneath it.
    if n:
        ax.fill_between(x, result.filtered, float(result.filtered.min()),
                        color=_FILTERED_COLOR, alpha=0.06, linewidth=0,
                        zorder=1)
    ax.plot(x, result.filtered, color=_FILTERED_COLOR, linewidth=2.0,
            solid_capstyle="round", zorder=3, label="Filtered")
    # Extrema: filled dots with a white halo so they read on top of the line.
    if maxima:
        ax.plot(maxima, result.filtered[maxima], "o", color=_MAX_COLOR,
                markersize=5.5, markeredgecolor=_BG_COLOR, markeredgewidth=1.0,
                linestyle="none", zorder=4, label="Maxima")
    if minima:
        ax.plot(minima, result.filtered[minima], "o", color=_MIN_COLOR,
                markersize=5.5, markeredgecolor=_BG_COLOR, markeredgewidth=1.0,
                linestyle="none", zorder=4, label="Minima")
    # User-added frames sit on the raw (unfiltered) trace, that is where the
    # user picked them, with a faint guide line so they read as deliberate
    # cut points rather than detected extrema.
    if user:
        for f in user:
            ax.axvline(f, color=_USER_COLOR, linewidth=0.8, alpha=0.35, zorder=2)
        ax.plot(user, result.raw[user], "D", color=_USER_COLOR, markersize=6,
                markeredgecolor=_BG_COLOR, markeredgewidth=1.0,
                linestyle="none", zorder=4, label="User-added")

    # Strip the box down to two soft axis lines.
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_SPINE)
        ax.spines[side].set_linewidth(1.0)

    ax.tick_params(colors=_MUTED, labelsize=9, length=0)
    ax.set_xlabel("Frame", color=_MUTED, fontsize=10)
    ylabel = "RMSD (Å)" if signal_label == "RMSD" else "%s projection" % signal_label
    ax.set_ylabel(ylabel, color=_MUTED, fontsize=10)

    # Left-aligned title with a lighter metadata line under it.
    ax.set_title("Filtered vs. unfiltered %s" % signal_label, loc="left", pad=18,
                 fontsize=13, fontweight="bold", color=_INK)
    meta = ("cutoff %.4g cycles/frame   ·   %d significant frames"
            % (result.cutoff_frequency, frames.size))
    cc = getattr(result, "cosine_content", None)
    high = getattr(result, "cosine_content_high", False)
    if cc is not None and high:
        meta += "   ·   cosine content %.2f ⚠ (random-walk-like)" % cc
    ax.annotate(
        meta,
        xy=(0, 1), xycoords="axes fraction", xytext=(0, 6),
        textcoords="offset points", ha="left", va="bottom",
        fontsize=9, color=_MUTED if not high else _MAX_COLOR)

    # Park the legend in its own column to the right of the plot so it never
    # overlaps the data.  Reserve that column with fixed margins (tight_layout
    # would otherwise ignore the outside-axes legend and clip it).
    figure.set_tight_layout(False)
    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.03, 1.0),
                       fontsize=9, frameon=False, handlelength=1.4,
                       labelcolor=_MUTED, borderaxespad=0.0)
    if legend is not None:
        legend.set_zorder(5)

    ax.margins(x=0.01)

    if ax_spec is not None:
        _draw_spectrum(ax_spec, result, sampling_rate)
        figure.subplots_adjust(left=0.09, right=0.80, top=0.88, bottom=0.12,
                               hspace=0.55)
    else:
        figure.subplots_adjust(left=0.09, right=0.80, top=0.85, bottom=0.14)
    return figure


def clamp_frame_index(xdata, n_frames):
    """Map a clicked x-coordinate to a valid 0-based frame index (or None)."""
    if xdata is None:
        return None
    return max(0, min(int(round(xdata)), n_frames - 1))


def clamp_cutoff_frequency(xdata, sampling_rate=1.0):
    """Map a clicked spectrum x-coordinate to a usable cutoff (or None).

    Keeps the value strictly inside ``(0, Nyquist)`` so it is always a valid
    Butterworth cutoff, a click on the axis edge can't produce a degenerate
    filter.
    """
    if xdata is None:
        return None
    nyquist = 0.5 * sampling_rate
    return max(1e-4, min(float(xdata), nyquist * 0.999))


try:
    from chimerax.core.tools import ToolInstance

    class MDSmoothPlot(ToolInstance):
        """RMSD plot window that doubles as a scrubber and a frame editor.

        Click anywhere on the plot to jump the trajectory to that frame; a cursor
        line marks the current frame and tracks playback.  ``structures`` are the
        models driven on click (typically just the raw trajectory, the smoothed
        copy follows it via the panel's playback link).

        The "Add significant frame" button turns the plot into an editor: the
        next click drops a preview marker on the unfiltered trace showing exactly
        where the new frame would land, and confirming it calls ``rebuild`` (the
        callback the command supplies) to fold that frame into the morph.
        """

        SESSION_ENDURING = False
        SESSION_SAVE = False

        def __init__(self, session, result, structures=None,
                     tool_name="MDSmooth", rebuild=None,
                     show_spectrum=False, on_pick_cutoff=None,
                     signal_label="RMSD"):
            super().__init__(session, tool_name)

            from chimerax.ui import MainToolWindow
            from matplotlib.backends.backend_qtagg import FigureCanvas
            from matplotlib.figure import Figure
            from Qt.QtWidgets import (
                QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
            )

            self._result = result
            self._structures = list(structures) if structures else []
            self._rebuild = rebuild
            self._show_spectrum = bool(show_spectrum)
            self._on_pick_cutoff = on_pick_cutoff
            self._signal_label = signal_label
            self._ax_spec = None
            self._cursor_idx = 0
            self._cursor = None

            # Interactive "add frame" state.
            self._add_mode = False
            self._pending_idx = None
            self._preview_artists = []
            self._user_frames = [int(f) for f, k in zip(result.frames, result.kinds)
                                 if k == "user"]

            self.tool_window = MainToolWindow(self)
            parent = self.tool_window.ui_area

            self._figure = Figure(figsize=(7.5, 4.0), tight_layout=True)
            self._canvas = FigureCanvas(self._figure)
            self._render()

            layout = QVBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self._canvas)

            # Control row: the add-frame button, its confirm/cancel pair, and a
            # shared hint label.  The confirm/cancel buttons only show while a
            # frame is being placed.  The whole row appears only when we can
            # actually rebuild the morph (i.e. in a real command run).
            if self._structures and self._rebuild is not None:
                row = QHBoxLayout()
                row.setContentsMargins(6, 0, 6, 4)
                self._add_btn = QPushButton("Add significant frame")
                self._add_btn.setCheckable(True)
                self._add_btn.setToolTip(
                    "Click, then click the plot where you want an extra "
                    "significant frame. Its preview lands on the unfiltered "
                    "(grey) trace so you can see the frame you are picking.")
                self._add_btn.toggled.connect(self._toggle_add_mode)
                self._confirm_btn = QPushButton("Add this frame")
                self._confirm_btn.clicked.connect(self._commit_frame)
                self._confirm_btn.hide()
                self._cancel_btn = QPushButton("Cancel")
                self._cancel_btn.clicked.connect(
                    lambda: self._add_btn.setChecked(False))
                self._cancel_btn.hide()
                row.addWidget(self._add_btn)
                row.addWidget(self._confirm_btn)
                row.addWidget(self._cancel_btn)
                row.addStretch(1)
                layout.addLayout(row)
            else:
                self._add_btn = None

            self._hint = QLabel(self._default_hint())
            self._hint.setWordWrap(True)
            self._hint.setContentsMargins(6, 0, 6, 4)
            layout.addWidget(self._hint)
            parent.setLayout(layout)

            self._click_cid = self._canvas.mpl_connect("button_press_event", self._on_click)
            self._frame_handler = session.triggers.add_handler("new frame", self._on_frame)

            self._canvas.draw()
            self.tool_window.manage("side")

        # --- rendering ----------------------------------------------------

        def _render(self):
            """(Re)draw the figure from the current result and restore the cursor.

            Called on first show and again after a frame is added, so the new
            significant-frame dots appear without spawning a second window.
            """
            self._figure.clear()
            build_figure(self._result, figure=self._figure,
                         show_spectrum=self._show_spectrum,
                         signal_label=self._signal_label)
            # The RMSD trace is always the first axis, whether or not the
            # spectrum panel is below it, that is the one the cursor rides.
            self._ax = self._figure.axes[0]
            self._ax_spec = (self._figure.axes[1]
                             if self._show_spectrum and len(self._figure.axes) > 1
                             else None)
            self._preview_artists = []
            self._cursor = self._ax.axvline(
                self._cursor_idx, color="0.25", linewidth=1.0, alpha=0.7, zorder=5)
            self._canvas.draw_idle()

        def update_result(self, result):
            """Swap in a freshly filtered result and redraw (no new window).

            Drives the panel's live cutoff dial: re-filtering the cached RMSD
            series is cheap, so each change to the frame count / cutoff calls this
            to refresh the same plot in place.
            """
            self._result = result
            self._user_frames = [int(f) for f, k in zip(result.frames, result.kinds)
                                 if k == "user"]
            self._render()

        def set_signal_label(self, label):
            """Update the signal name (y-axis label / title); redraw on change."""
            if label != self._signal_label:
                self._signal_label = label
                self._render()

        def set_show_spectrum(self, show):
            """Toggle the spectrum panel and redraw."""
            show = bool(show)
            if show != self._show_spectrum:
                self._show_spectrum = show
                self._render()
                if not self._add_mode:
                    self._hint.setText(self._default_hint())

        def _default_hint(self):
            parts = []
            if self._structures:
                parts.append("Click the plot to jump to that frame.")
            if self._show_spectrum and self._on_pick_cutoff is not None:
                parts.append("Click the spectrum to set the cutoff there.")
            return "  ".join(parts)

        # --- current-frame link -------------------------------------------

        def _first_id(self):
            s = self._structures[0]
            return s.coordset_ids[0]

        def _coordset_slider(self, structure):
            """The structure's native Coordinate Sets slider, if one is shown."""
            for sl in getattr(self.session, "_coord_set_sliders", ()):
                if getattr(sl, "structure", None) is structure:
                    return sl
            return None

        def _on_click(self, event):
            # A click in the spectrum panel sets the low-pass cutoff at that
            # frequency: re-filter through the supplied callback and redraw.
            if (self._ax_spec is not None and event.inaxes is self._ax_spec
                    and self._on_pick_cutoff is not None):
                freq = clamp_cutoff_frequency(event.xdata)
                if freq is not None:
                    new_result = self._on_pick_cutoff(freq)
                    if new_result is not None:
                        self.update_result(new_result)
                return
            if event.inaxes is not self._ax:
                return
            idx = clamp_frame_index(event.xdata, self._result.raw.size)
            if idx is None:
                return
            if self._add_mode:
                self._preview_frame(idx)
                return
            if not self._structures:
                return
            for s in self._structures:
                if s.deleted:
                    continue
                ids = s.coordset_ids
                cid = ids[min(idx, len(ids) - 1)]
                slider = self._coordset_slider(s)
                if slider is not None:
                    # Drive the slider itself: this jumps the frame AND updates
                    # the play loop's position in one synchronous call, so if it
                    # was playing it keeps playing from here instead of snapping
                    # back; if paused it just jumps and stays.
                    slider.set_slider(cid)
                else:
                    s.active_coordset_id = cid
            self._move_cursor(idx)

        def _on_frame(self, *args):
            if not self._structures:
                return
            s = self._structures[0]
            if s.deleted:
                return
            idx = s.active_coordset_id - self._first_id()
            if idx != self._cursor_idx:
                self._move_cursor(idx)

        def _move_cursor(self, idx):
            self._cursor_idx = idx
            if self._cursor is not None:
                self._cursor.set_xdata([idx, idx])
                self._canvas.draw_idle()

        # --- add-frame editor ---------------------------------------------

        def _toggle_add_mode(self, checked):
            self._add_mode = bool(checked)
            self._clear_preview()
            self._pending_idx = None
            self._confirm_btn.hide()
            if checked:
                self._add_btn.setText("Adding… (click the plot)")
                self._cancel_btn.show()
                self._hint.setText(
                    "Click on the plot where you want a significant frame - the "
                    "marker lands on the unfiltered (grey) trace so you can see "
                    "the frame you are adding.")
            else:
                self._add_btn.setText("Add significant frame")
                self._cancel_btn.hide()
                self._hint.setText(self._default_hint())
            self._canvas.draw_idle()

        def _clear_preview(self):
            for artist in self._preview_artists:
                try:
                    artist.remove()
                except Exception:
                    pass
            self._preview_artists = []

        def _preview_frame(self, idx):
            """Draw (or move) the not-yet-committed marker on the raw trace."""
            self._clear_preview()
            self._pending_idx = idx
            y = float(self._result.raw[idx])
            vline = self._ax.axvline(idx, color=_PREVIEW_COLOR, linewidth=1.2,
                                     linestyle="--", alpha=0.9, zorder=6)
            dot, = self._ax.plot([idx], [y], marker="D", color=_PREVIEW_COLOR,
                                 markersize=7, markeredgecolor="#ffffff",
                                 markeredgewidth=1.0, linestyle="none", zorder=7)
            label = self._ax.annotate(
                "frame %d" % idx, xy=(idx, y), xytext=(0, 10),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color=_PREVIEW_COLOR, zorder=7)
            self._preview_artists = [vline, dot, label]
            self._confirm_btn.setText("Add frame %d" % idx)
            self._confirm_btn.show()
            self._hint.setText(
                "Frame %d - RMSD %.2f Å. Click “Add frame %d” to keep it, or "
                "click elsewhere to move the marker." % (idx, y, idx))
            self._canvas.draw_idle()

        def _commit_frame(self):
            if self._pending_idx is None or self._rebuild is None:
                return
            idx = self._pending_idx
            frames = sorted(set(self._user_frames) | {idx})
            try:
                new_result, new_morph = self._rebuild(frames)
            except Exception as e:
                self.session.logger.warning(
                    "Could not add significant frame %d: %s" % (idx, e))
                return
            self._user_frames = frames
            self._result = new_result
            self._render()
            # Point the panel's playback link / overlay at the rebuilt morph.
            tool = self._find_filter_tool()
            if tool is not None and self._structures:
                try:
                    tool.relink_morph(self._structures[0], new_morph)
                except Exception:
                    pass
            self._add_btn.setChecked(False)  # exits add-mode, resets the hint
            n = len(self._user_frames)
            tail = (" The morph was rebuilt to include it."
                    if new_morph is not None else "")
            self._hint.setText(
                "Added frame %d (%d user-added frame%s).%s"
                % (idx, n, "" if n == 1 else "s", tail))

        def _find_filter_tool(self):
            try:
                from .tool import MDSmoothTool
            except Exception:
                return None
            for t in self.session.tools.list():
                if isinstance(t, MDSmoothTool):
                    return t
            return None

        # --- cleanup ------------------------------------------------------

        def delete(self):
            if getattr(self, "_frame_handler", None) is not None:
                self._frame_handler.remove()
                self._frame_handler = None
            if getattr(self, "_click_cid", None) is not None:
                self._canvas.mpl_disconnect(self._click_cid)
                self._click_cid = None
            super().delete()

except ImportError:
    # ChimeraX not present (e.g. headless unit tests). build_figure still works.
    MDSmoothPlot = None

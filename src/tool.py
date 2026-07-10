"""
The "MDSmooth" GUI panel (Tools > Trajectory Analysis > MDSmooth).

A two-stage front end over the ``mdsmooth`` command:

  1. **Analyze** (explicit button) reads the trajectory's coordinates once,
     computes the RMSD series, and shows a preview plot, the filtered curve,
     the significant frames, a live frame count, and (optionally) the power
     spectrum with the chosen cutoff marked.  Re-filtering the cached series is
     cheap, so pressing Analyze again after changing the cutoff re-previews
     instantly without re-reading coordinates.
  2. **Build movie** does the expensive part, the morph, only once you are
     happy with the frame count, then overlays the smoothed copy on the raw and
     links it to the raw's Coordinate Sets slider.

The cutoff is chosen through one "Choose cutoff by" dropdown (number of frames /
cutoff frequency / spectral power), so the three equivalent controls can never
conflict in the GUI the way they can on the command line.

Module-level helpers (``parse_cutoff``, ``resolve_cutoff_selection``,
``cutoff_keyword``, ``normalize_spec``, ``build_command``, ``mirror_frame``)
carry no ChimeraX/Qt dependency and are unit-tested headless; the Qt panel below
is only defined when ChimeraX is importable.
"""

try:
    from chimerax.core.tools import ToolInstance
    from chimerax.core.errors import UserError
    from chimerax.atomic import AtomicStructure
    _HAVE_CHIMERAX = True
except ImportError:  # headless (unit tests): the pure helpers below still import
    _HAVE_CHIMERAX = False


# The identifiers behind the "Choose cutoff by" dropdown, highest command
# priority first (matches mdsmooth's own targetFrames > cutoffFrequency >
# powerCutoff order).
CUTOFF_MODES = ("frames", "cutoff", "power")


def parse_cutoff(text):
    """Validate a typed smoothing cutoff. Returns the float or raises ValueError."""
    cutoff = float(text)
    if not (0.0 < cutoff < 0.5):
        raise ValueError("Smoothing cutoff must be between 0 and 0.5 cycles/frame.")
    return cutoff


def parse_percent(text):
    """Validate a 0-100 percentage. Returns the float or raises ValueError."""
    pct = float(text)
    if not (0.0 <= pct <= 100.0):
        raise ValueError("Transparency must be between 0 and 100 percent.")
    return pct


def resolve_cutoff_selection(mode, value_text):
    """Turn the dropdown mode + typed value into ``filter_rmsd`` keyword args.

    Returns a dict carrying exactly one of ``target_frames`` /
    ``cutoff_frequency`` / ``power_fraction``.  Raises ``ValueError`` with a
    user-facing message on empty or out-of-range input, so the panel can show it
    verbatim.
    """
    text = (value_text or "").strip()
    if not text:
        raise ValueError("Enter a value for the chosen cutoff mode.")
    if mode == "frames":
        try:
            v = int(text)
        except ValueError:
            raise ValueError("Number of frames must be a whole number (e.g. 50).")
        if v < 2:
            raise ValueError("Number of frames must be at least 2.")
        return {"target_frames": v}
    if mode == "cutoff":
        try:
            v = float(text)
        except ValueError:
            raise ValueError("Cutoff frequency must be a number (e.g. 0.03).")
        if not (0.0 < v < 0.5):
            raise ValueError(
                "Cutoff frequency must be between 0 and 0.5 cycles/frame.")
        return {"cutoff_frequency": v}
    if mode == "power":
        try:
            v = float(text)
        except ValueError:
            raise ValueError(
                "Spectral power fraction must be a number (e.g. 0.95).")
        if not (0.0 < v < 1.0):
            raise ValueError("Spectral power fraction must be between 0 and 1.")
        return {"power_fraction": v}
    raise ValueError("Unknown cutoff mode: %r" % (mode,))


def _fmt(x):
    """Compact but faithful float formatting for embedding in a command line."""
    return "%.6g" % x


def cutoff_keyword(mode, value_text):
    """The ``mdsmooth`` keyword+value for a dropdown selection.

    e.g. ``("frames", "50")`` -> ``"targetFrames 50"``.  Validates through
    :func:`resolve_cutoff_selection`, so it raises the same friendly errors.
    """
    kwargs = resolve_cutoff_selection(mode, value_text)
    if "target_frames" in kwargs:
        return "targetFrames %d" % kwargs["target_frames"]
    if "cutoff_frequency" in kwargs:
        return "cutoffFrequency %s" % _fmt(kwargs["cutoff_frequency"])
    return "powerCutoff %s" % _fmt(kwargs["power_fraction"])


def normalize_spec(model_id, highlight_spec):
    """Expand a possibly-bare highlight spec into a full atom-spec.

    Blank stays blank (meaning the whole structure); a bare ``:res`` / ``@name``
    / ``/chain`` is prefixed with ``#model_id``; anything else is left as typed.
    """
    spec = (highlight_spec or "").strip()
    if spec and spec[0] in ":@/":
        spec = "#%s%s" % (model_id, spec)
    return spec


def build_command(model_id, highlight_spec, cutoff_kw, method="corkscrew",
                  make_morph=True, show_plot=True, signal="rmsd", lag=None):
    """Assemble the ``mdsmooth`` command string for the given panel inputs.

    ``cutoff_kw`` is the already-formatted cutoff keyword+value (see
    :func:`cutoff_keyword`), e.g. ``"targetFrames 50"``.  ``make_morph`` /
    ``show_plot`` map to the command's booleans so the panel can run a cheap
    analysis-only pass (``makeMorph false``) or the full build.  ``signal`` adds
    the ``signal`` keyword only when it isn't the default ``"rmsd"``; ``lag`` adds
    the ``lag`` keyword only for the ``"ic1"`` (tICA) signal.
    """
    cmd = "mdsmooth #%s" % model_id
    spec = normalize_spec(model_id, highlight_spec)
    if spec:
        cmd += " toAtoms %s" % spec
    if signal and signal != "rmsd":
        cmd += " signal %s" % signal
    if signal in ("ic1", "deeptica") and lag is not None:
        cmd += " lag %d" % int(lag)
    cmd += " %s method %s makeMorph %s showPlot %s name smoothed" % (
        cutoff_kw, method,
        "true" if make_morph else "false",
        "true" if show_plot else "false",
    )
    return cmd


def build_kinetic_command(model_id, highlight_spec, states, method="corkscrew",
                          lag=None, make_morph=True):
    """Assemble the ``mdsmooth ... mode kinetic`` command for the panel.

    Kinetic mode picks one key frame per metastable state (MSM/PCCA+) instead of
    filtering a 1D signal, so it takes ``states`` rather than a cutoff.  ``lag`` is
    the shared tICA/MSM lag; it is added only when given.
    """
    cmd = "mdsmooth #%s" % model_id
    spec = normalize_spec(model_id, highlight_spec)
    if spec:
        cmd += " toAtoms %s" % spec
    cmd += " mode kinetic states %d" % int(states)
    if lag is not None:
        cmd += " lag %d" % int(lag)
    cmd += " method %s makeMorph %s showPlot false name smoothed" % (
        method, "true" if make_morph else "false")
    return cmd


def mirror_frame(raw, morph):
    """Copy the raw's current frame onto the morph (clamped to the morph's range).

    Returns without doing anything if either model is gone or already in sync, so
    it is cheap to call every graphics frame and never triggers a redraw loop.
    """
    if raw.deleted or morph.deleted:
        return
    target = raw.active_coordset_id
    lo, hi = morph.coordset_ids[0], morph.coordset_ids[-1]
    target = min(max(target, lo), hi)
    if morph.active_coordset_id != target:
        morph.active_coordset_id = target


# The per-mode value-field presentation: (label, default, unit hint, tooltip).
_MODE_FIELDS = {
    "frames": (
        "Number of frames:", "50", "frames",
        "How many significant frames the movie keeps.\n"
        "More = smoother motion retained, but more morph segments to build\n"
        "(slower). This is the recommended dial and adapts to the atoms you\n"
        "highlight."),
    "cutoff": (
        "Cutoff frequency:", "0.03", "cycles/frame",
        "Butterworth low-pass cutoff in cycles/frame (0-0.5).\n"
        "Lower = smoother (fewer, broader movements); higher = keeps more\n"
        "detail. Use to reproduce a specific published movie."),
    "power": (
        "Spectral power:", "0.95", "fraction (0-1)",
        "Keep the cutoff that retains this fraction of the RMSD's spectral\n"
        "power (mean removed). Expert option; a frame target is usually easier."),
}


if _HAVE_CHIMERAX:

    class MDSmoothTool(ToolInstance):

        SESSION_ENDURING = False
        SESSION_SAVE = False
        help = "help:user/commands/mdsmooth.html"

        def __init__(self, session, tool_name):
            super().__init__(session, tool_name)

            from chimerax.ui import MainToolWindow
            from Qt.QtWidgets import (
                QVBoxLayout, QGridLayout, QHBoxLayout, QLabel,
                QLineEdit, QPushButton, QComboBox, QCheckBox,
            )
            from Qt.QtCore import Qt

            self.tool_window = MainToolWindow(self)
            parent = self.tool_window.ui_area
            outer = QVBoxLayout()
            outer.setContentsMargins(8, 8, 8, 8)
            parent.setLayout(outer)

            grid = QGridLayout()
            grid.setColumnStretch(1, 1)
            outer.addLayout(grid)

            # Row 0: trajectory model chooser.
            grid.addWidget(QLabel("Trajectory:"), 0, 0)
            self._model_menu = QComboBox()
            grid.addWidget(self._model_menu, 0, 1, 1, 2)

            # Row 1: region to highlight (drives which atoms' RMSD is filtered).
            grid.addWidget(QLabel("Highlight:"), 1, 0)
            self._sel_edit = QLineEdit()
            self._sel_edit.setPlaceholderText(
                "atom-spec, e.g. :436-441  (blank = whole structure)")
            grid.addWidget(self._sel_edit, 1, 1)
            from_sel = QPushButton("From selection")
            from_sel.setToolTip("Use the atoms currently selected in the viewer")
            from_sel.clicked.connect(lambda: self._sel_edit.setText("sel"))
            grid.addWidget(from_sel, 1, 2)

            # Row 2: which 1D signal to analyze, RMSD, or the largest collective
            # motion (PC1). Both feed the same filter/frames/morph pipeline.
            grid.addWidget(QLabel("Signal:"), 2, 0)
            self._signal_menu = QComboBox()
            self._signal_menu.addItem("RMSD to reference frame", "rmsd")
            self._signal_menu.addItem("PC1 (largest collective motion)", "pc1")
            self._signal_menu.addItem("IC1 (slowest motion - tICA)", "ic1")
            self._signal_menu.addItem("dPC1 (internal dihedral motion)", "dpca")
            self._signal_menu.addItem(
                "DeepTICA (learned slow CV - needs setup)", "deeptica")
            self._signal_menu.setToolTip(
                "RMSD: distance of each frame from the reference.\n"
                "PC1: projection onto the single largest collective motion of the\n"
                "highlighted atoms (essential dynamics), captures the direction\n"
                "of the dominant motion, not just distance.\n"
                "IC1: projection onto the *slowest* collective motion (tICA).\n"
                "Slowness is usually a better proxy for a functional transition\n"
                "than variance, but needs a trajectory that samples the motion;\n"
                "watch the cosine-content warning.\n"
                "dPC1: PCA of backbone phi/psi dihedrals (internal coordinates),\n"
                "so no alignment is needed and global tumbling can't leak in, "
                "good for flexible/looping backbones (protein backbone only).\n"
                "PC1/IC1/dPC1 cost a little more to compute; keep the highlight\n"
                "focused to keep them fast.")
            self._signal_menu.currentIndexChanged.connect(self._on_signal_changed)
            grid.addWidget(self._signal_menu, 2, 1, 1, 2)

            # Row 3: tICA lag, only meaningful for the IC1 signal, so it is
            # enabled only when IC1 is chosen (see _on_signal_changed).
            self._lag_label = QLabel("tICA lag (frames):")
            grid.addWidget(self._lag_label, 3, 0)
            self._lag_edit = QLineEdit("10")
            self._lag_edit.setToolTip(
                "Lag time for IC1 (tICA), in frames. The one finicky knob:\n"
                "too small behaves like PC1, too large gets noisy. Only used\n"
                "when the signal is IC1.")
            grid.addWidget(self._lag_edit, 3, 1)
            self._lag_unit = QLabel("(IC1 only)")
            grid.addWidget(self._lag_unit, 3, 2)

            # Row 4: how to choose the cutoff, one dropdown, so the three
            # equivalent controls cannot conflict.
            grid.addWidget(QLabel("Choose cutoff by:"), 4, 0)
            self._mode_menu = QComboBox()
            self._mode_menu.addItem("Number of frames", "frames")
            self._mode_menu.addItem("Cutoff frequency", "cutoff")
            self._mode_menu.addItem("Spectral power", "power")
            self._mode_menu.setToolTip(
                "All three set the same low-pass cutoff, pick the units you\n"
                "want to think in. 'Number of frames' is recommended.")
            self._mode_menu.currentIndexChanged.connect(self._update_mode_fields)
            grid.addWidget(self._mode_menu, 4, 1, 1, 2)

            # Row 5: the value for the chosen mode (label + unit change with it).
            self._value_label = QLabel()
            grid.addWidget(self._value_label, 5, 0)
            self._value_edit = QLineEdit()
            grid.addWidget(self._value_edit, 5, 1)
            self._value_unit = QLabel()
            grid.addWidget(self._value_unit, 5, 2)

            # Row 6: optional spectrum panel in the preview plot.
            self._spectrum_check = QCheckBox(
                "Show power spectrum in the preview (helps pick the cutoff)")
            self._spectrum_check.setChecked(False)
            self._spectrum_check.toggled.connect(self._on_spectrum_toggled)
            grid.addWidget(self._spectrum_check, 6, 0, 1, 3)

            # Row 7: overlay option.
            self._overlay_check = QCheckBox(
                "Overlay: fade the raw to a grey ghost under the smoothed copy")
            self._overlay_check.setChecked(True)
            grid.addWidget(self._overlay_check, 7, 0, 1, 3)

            # Row 8: how faded the raw ghost is (typed; applies live once overlaid).
            grid.addWidget(QLabel("Raw transparency:"), 8, 0)
            self._transp_edit = QLineEdit("50")
            self._transp_edit.setToolTip(
                "How faded the raw ghost is, in percent.\n"
                "0 = fully opaque, 100 = invisible. Applies live once an overlay exists.")
            self._transp_edit.editingFinished.connect(self._apply_transparency)
            grid.addWidget(self._transp_edit, 8, 1)
            grid.addWidget(QLabel("% (0=opaque, 100=invisible)"), 8, 2)

            # Row 9: link playback to the native slider.
            self._link_check = QCheckBox(
                "Link the smoothed copy to the raw's Coordinate Sets slider")
            self._link_check.setChecked(True)
            self._link_check.setToolTip(
                "The raw trajectory's slider (play / pause / speed) drives the\n"
                "smoothed copy too, so they stay in lockstep.")
            grid.addWidget(self._link_check, 9, 0, 1, 3)

            # Row 10: interpolation method for the morph.
            grid.addWidget(QLabel("Interpolation:"), 10, 0)
            self._method_menu = QComboBox()
            self._method_menu.addItem(
                "Corkscrew - natural rotation (slower)", "corkscrew")
            self._method_menu.addItem("Linear - fast, straight-line", "linear")
            self._method_menu.setToolTip(
                "Corkscrew interpolates through ChimeraX's morph engine so\n"
                "domains rotate naturally; it runs once per segment, so more\n"
                "frames take longer to build. Linear is a fast straight line.")
            grid.addWidget(self._method_menu, 10, 1, 1, 2)

            # Row 11: kinetic (MSM) mode, a different key-frame strategy.
            self._kinetic_check = QCheckBox(
                "Kinetic (MSM) mode - one key frame per metastable state "
                "(needs deeptime)")
            self._kinetic_check.setChecked(False)
            self._kinetic_check.setToolTip(
                "Available with the IC1 (tICA) signal - kinetic mode clusters in\n"
                "tICA space. Instead of sampling one motion and cutting at its\n"
                "extrema, it groups the trajectory into kinetically distinct\n"
                "metastable states (tICA → MSM → PCCA+) and builds a tour of one\n"
                "representative frame per state - 'show me the N distinct\n"
                "conformations'. Needs the optional 'deeptime' package.")
            self._kinetic_check.toggled.connect(self._on_kinetic_toggled)
            grid.addWidget(self._kinetic_check, 11, 0, 1, 3)

            # Row 12: number of metastable states (kinetic mode only).
            self._states_label = QLabel("States:")
            grid.addWidget(self._states_label, 12, 0)
            self._states_edit = QLineEdit("5")
            self._states_edit.setToolTip(
                "How many metastable states (key frames) to find. Kinetic mode only.")
            grid.addWidget(self._states_edit, 12, 1)
            self._states_unit = QLabel("(kinetic only)")
            grid.addWidget(self._states_unit, 12, 2)

            # Button row: the two explicit stages.
            btn_row = QHBoxLayout()
            self._analyze_btn = QPushButton("Analyze")
            self._analyze_btn.setToolTip(
                "Compute the RMSD and preview the significant frames at the\n"
                "current cutoff, no movie is built yet.")
            self._analyze_btn.clicked.connect(self._on_analyze)
            self._build_btn = QPushButton("Build movie")
            self._build_btn.setDefault(True)
            self._build_btn.setToolTip(
                "Build the smoothed morph at the current cutoff and overlay it\n"
                "on the raw trajectory. This is the slow step.")
            self._build_btn.clicked.connect(self._on_build)
            self._help_btn = QPushButton("Help")
            self._help_btn.setToolTip("Open the mdsmooth command documentation.")
            self._help_btn.clicked.connect(self._on_help)
            btn_row.addWidget(self._analyze_btn)
            btn_row.addWidget(self._build_btn)
            btn_row.addStretch(1)
            btn_row.addWidget(self._help_btn)
            outer.addLayout(btn_row)

            # Status line.
            self._status = QLabel("")
            self._status.setWordWrap(True)
            self._status.setTextInteractionFlags(Qt.TextSelectableByMouse)
            outer.addWidget(self._status)

            # State ---------------------------------------------------------
            self._pair = None            # (raw, morph) currently linked
            self._preview_plot = None    # the Analyze preview MDSmoothPlot
            self._series = None          # cached RMSD series
            self._series_key = None      # (model id, n frames, spec) the series is for
            self._series_natoms = None

            self._update_mode_fields()
            self._update_lag_enabled()
            self._update_kinetic_enabled()
            self._update_kinetic_availability()

            from chimerax.core.models import ADD_MODELS, REMOVE_MODELS
            self._handlers = [
                session.triggers.add_handler(ADD_MODELS, self._populate_models),
                session.triggers.add_handler(REMOVE_MODELS, self._populate_models),
                session.triggers.add_handler("new frame", self._on_frame),
            ]
            self._populate_models()

            self.tool_window.manage("side")

        # --- signal / cutoff-mode fields ---------------------------------

        def _signal(self):
            return self._signal_menu.currentData() or "rmsd"

        def _signal_label(self):
            return {"pc1": "PC1", "ic1": "IC1", "dpca": "dPC1",
                    "deeptica": "DeepTICA"}.get(self._signal(), "RMSD")

        def _update_lag_enabled(self):
            """Show the lag field only for a lag-using signal (IC1/DeepTICA).

            Hidden, not just greyed, otherwise, so the row collapses and the
            lag knob never appears for signals that ignore it.
            """
            uses_lag = self._signal() in ("ic1", "deeptica")
            for w in (self._lag_label, self._lag_edit, self._lag_unit):
                w.setVisible(uses_lag)

        def _lag_value(self):
            """Parse the tICA lag field; raise ValueError with a friendly message."""
            text = (self._lag_edit.text() or "").strip()
            try:
                v = int(text)
            except ValueError:
                raise ValueError("tICA lag must be a whole number of frames (e.g. 10).")
            if v < 1:
                raise ValueError("tICA lag must be at least 1 frame.")
            return v

        # --- kinetic (MSM) mode ------------------------------------------

        def _kinetic(self):
            return self._kinetic_check.isChecked()

        def _states_value(self):
            """Parse the states field; raise ValueError with a friendly message."""
            text = (self._states_edit.text() or "").strip()
            try:
                v = int(text)
            except ValueError:
                raise ValueError("States must be a whole number (e.g. 5).")
            if v < 2:
                raise ValueError("States must be at least 2 (a morph needs two).")
            return v

        def _update_kinetic_enabled(self):
            """Kinetic mode swaps the cutoff/signal controls for the states field."""
            kin = self._kinetic()
            # The states field only exists in kinetic mode; hide it otherwise so
            # the row collapses.
            for w in (self._states_label, self._states_edit, self._states_unit):
                w.setVisible(kin)
            # The 1D-signal cutoff machinery does not apply in kinetic mode.
            for w in (self._mode_menu, self._value_label, self._value_edit,
                      self._value_unit, self._signal_menu, self._spectrum_check):
                w.setEnabled(not kin)
            # Kinetic mode is only reachable under IC1, which already shows the
            # tICA/MSM lag, so lag visibility is driven purely by the signal.
            self._update_lag_enabled()

        def _update_kinetic_availability(self):
            """Kinetic mode clusters in tICA space, so offer it only under IC1.

            When the signal isn't IC1 the checkbox is hidden (so kinetic mode is
            invisible entirely); if it happened to be on, turn it off first (which
            restores the cutoff / signal controls).
            """
            is_ic1 = self._signal() == "ic1"
            if not is_ic1 and self._kinetic_check.isChecked():
                self._kinetic_check.setChecked(False)  # fires _on_kinetic_toggled
            self._kinetic_check.setVisible(is_ic1)

        def _on_kinetic_toggled(self, checked):
            self._update_kinetic_enabled()
            if checked:
                self._status.setText(
                    "Kinetic mode: click Build movie to cluster the trajectory "
                    "into metastable states and morph a tour of them. (No cheap "
                    "preview - the clustering runs at Build. Needs deeptime.)")
            else:
                self._status.setText("")

        def _on_signal_changed(self, *args):
            # A different signal is a different cached series; drop the cache so
            # the next Analyze recomputes, and nudge the user to re-run.
            self._series = None
            self._series_key = None
            self._update_lag_enabled()
            self._update_kinetic_availability()
            # A frame *target* means the same thing on either signal, but a fixed
            # cutoff / power fraction does not, the same cutoff yields very
            # different frame counts on RMSD vs PC1 (PC1 carries more
            # high-frequency content). So if the cutoff control is on one of those
            # (e.g. left there by clicking the spectrum), reset it to the frame
            # target so switching signals doesn't silently blow up the count.
            reset = ""
            if self._mode() != "frames":
                default = _MODE_FIELDS["frames"][1]
                self._set_mode("frames", default)
                reset = " (cutoff reset to a %s-frame target)" % default
            if getattr(self, "_status", None) is not None:
                self._status.setText(
                    "Signal changed to %s - click Analyze to preview it.%s"
                    % (self._signal_label(), reset))

        def _mode(self):
            return self._mode_menu.currentData() or "frames"

        def _update_mode_fields(self, *args):
            label, default, unit, tip = _MODE_FIELDS[self._mode()]
            self._value_label.setText(label)
            self._value_edit.setText(default)
            self._value_edit.setToolTip(tip)
            self._value_unit.setText(unit)

        def _set_mode(self, mode, value_text):
            """Switch the dropdown to ``mode`` and put ``value_text`` in the field.

            Used when a spectrum click chooses a cutoff for the user: setting the
            index first fires ``_update_mode_fields`` (which resets to the mode's
            default), so the value is written afterwards to stick.
            """
            idx = self._mode_menu.findData(mode)
            if idx >= 0:
                self._mode_menu.setCurrentIndex(idx)
            self._value_edit.setText(value_text)

        def _pick_cutoff_from_spectrum(self, freq):
            """Preview-plot callback: the user clicked the spectrum at ``freq``.

            Switches the panel to explicit-cutoff mode with that frequency (so a
            later Build uses exactly what was clicked), re-filters the cached RMSD
            series, and returns the result for the plot to redraw.  Returns
            ``None`` if there is no analysis to re-filter.
            """
            from .filter import filter_rmsd
            if self._series is None:
                return None
            # Show the same rounded value the field will hold, and filter on it,
            # so the preview and a subsequent Build agree to the last digit.
            text = "%.4g" % freq
            self._set_mode("cutoff", text)
            result = filter_rmsd(self._series, cutoff_frequency=float(text))
            self._report_analysis(self._current_model(), result)
            return result

        # --- model menu ---------------------------------------------------

        def _trajectory_models(self):
            return [m for m in self.session.models.list(type=AtomicStructure)
                    if m.num_coordsets > 1]

        def _populate_models(self, *args):
            current = self._current_model()
            self._model_menu.clear()
            self._models = self._trajectory_models()
            for m in self._models:
                self._model_menu.addItem(
                    "%s  %s  (%d frames)" % (m.id_string, m.name, m.num_coordsets))
            if current in self._models:
                self._model_menu.setCurrentIndex(self._models.index(current))
            if not self._models:
                self._status.setText(
                    "No trajectory loaded (need a model with multiple frames).")

        def _current_model(self):
            i = self._model_menu.currentIndex()
            models = getattr(self, "_models", [])
            if 0 <= i < len(models):
                return models[i]
            return None

        # --- atom-spec resolution ----------------------------------------

        def _resolve_atoms(self, model, spec_text):
            """Evaluate the highlight spec to an Atoms collection (or None).

            Blank means the whole structure (``None`` -> command's default fit
            selection).  Raises ``UserError`` if the spec is unparsable or empty.
            """
            spec = normalize_spec(model.id_string, spec_text)
            if not spec:
                return None
            from chimerax.core.commands import AtomSpecArg
            try:
                aspec, _, _ = AtomSpecArg.parse(spec, self.session)
                objects = aspec.evaluate(self.session)
            except Exception as e:
                raise UserError(
                    "Could not understand the highlight spec %r: %s"
                    % (spec_text, e))
            atoms = objects.atoms
            if len(atoms) == 0:
                raise UserError("The highlight spec %r matched no atoms." % spec_text)
            return atoms

        def _atoms_fingerprint(self, atoms):
            """A cache fingerprint for the *resolved* highlight atoms.

            Keys the Analyze cache on the atoms themselves rather than the spec
            text, so a dynamic spec (e.g. ``sel``, whose text is constant while the
            atoms it resolves to change with the viewer selection) invalidates the
            cache. ``None`` means the whole structure. Prefers the atoms' stable
            pointers; falls back to their current coordinates; and, if neither can
            be fingerprinted, returns a fresh token each call so the series is
            recomputed rather than reused stale.
            """
            if atoms is None:
                return ("all",)
            import numpy as np
            ptrs = getattr(atoms, "pointers", None)
            if ptrs is not None:
                try:
                    return ("id", len(atoms), hash(np.asarray(ptrs).tobytes()))
                except Exception:
                    pass
            try:
                coords = np.ascontiguousarray(atoms.coords)
                return ("xyz", len(atoms), hash(coords.tobytes()))
            except Exception:
                self._fp_counter = getattr(self, "_fp_counter", 0) + 1
                return ("uncacheable", self._fp_counter)

        # --- stage 1: analyze --------------------------------------------

        def _on_analyze(self):
            from Qt.QtWidgets import QApplication
            from .filter import filter_rmsd

            model = self._current_model()
            if model is None:
                self._status.setText(
                    "Pick a trajectory first (load one with multiple frames).")
                return

            if self._kinetic():
                self._status.setText(
                    "Kinetic mode has no cheap preview - click Build movie to "
                    "cluster into metastable states and morph the tour.")
                return

            if self._signal() == "deeptica":
                self._status.setText(
                    "DeepTICA trains a network (minutes) and has no cheap "
                    "preview - click Build movie. Needs the learned-CV setup: "
                    "run  mdsmooth installLearnedCV  once if you haven't.")
                return

            try:
                atoms = self._resolve_atoms(model, self._sel_edit.text())
                kwargs = resolve_cutoff_selection(
                    self._mode(), self._value_edit.text())
            except (ValueError, UserError) as e:
                self._status.setText(str(e))
                return

            # Read + cache the signal series, but only when the trajectory, the
            # highlighted atoms, the signal, or (for IC1) the lag actually
            # changed, re-previewing a new cutoff on the same selection reuses
            # the cached series and is instant. The key fingerprints the resolved
            # atoms, not the spec text, so a dynamic spec like "sel" (same text,
            # different atoms once the viewer selection changes) invalidates the
            # cache instead of previewing a stale series.
            signal = self._signal()
            lag = None
            if signal == "ic1":
                try:
                    lag = self._lag_value()
                except ValueError as e:
                    self._status.setText(str(e))
                    return
            key = (id(model), model.num_coordsets,
                   self._atoms_fingerprint(atoms), signal, lag)
            if key != self._series_key or self._series is None:
                self._status.setText(
                    "Reading coordinates and computing %s…" % self._signal_label())
                QApplication.processEvents()
                from .cmd import (
                    compute_rmsd_series, compute_pc1_series, compute_ic1_series,
                    compute_dpca_series,
                )
                try:
                    if signal == "ic1":
                        series, natoms = compute_ic1_series(
                            model, to_atoms=atoms, lag=lag)
                    elif signal == "pc1":
                        series, natoms = compute_pc1_series(model, to_atoms=atoms)
                    elif signal == "dpca":
                        series, natoms = compute_dpca_series(model, to_atoms=atoms)
                    else:
                        series, natoms = compute_rmsd_series(model, to_atoms=atoms)
                except UserError as e:
                    self._status.setText(str(e))
                    return
                self._series = series
                self._series_key = key
                self._series_natoms = natoms

            result = filter_rmsd(self._series, **kwargs)
            self._show_preview(model, result)
            self._report_analysis(model, result)

        def _show_preview(self, model, result):
            """Open or update the preview plot for an analysis result."""
            show_spec = self._spectrum_check.isChecked()
            plot = self._preview_plot
            if plot is not None:
                try:
                    plot.set_signal_label(self._signal_label())
                    plot.set_show_spectrum(show_spec)
                    plot.update_result(result)
                    return
                except Exception:
                    # The preview window was closed; fall through and remake it.
                    self._preview_plot = None
            from .graph import MDSmoothPlot
            if MDSmoothPlot is None:
                return
            self._preview_plot = MDSmoothPlot(
                self.session, result, structures=[model],
                tool_name="RMSD preview: %s" % model, rebuild=None,
                show_spectrum=show_spec,
                on_pick_cutoff=self._pick_cutoff_from_spectrum,
                signal_label=self._signal_label())

        def _report_analysis(self, model, result):
            atoms = self._series_natoms
            warn = ""
            if result.cosine_content_high:
                warn = (" ⚠ High cosine content (%.2f): the signal resembles a "
                        "random walk, so these key frames may be undersampling "
                        "artifacts rather than real motion - consider a more "
                        "localized highlight or the RMSD signal."
                        % result.cosine_content)
            self._status.setText(
                "Preview (%s): %d significant frames (cutoff %.4g cycles/frame, "
                "%s fit atoms). Adjust and Analyze again, or Build movie when "
                "you're happy - more frames means a slower build.%s"
                % (self._signal_label(), result.frames.size,
                   result.cutoff_frequency,
                   atoms if atoms is not None else "?", warn))

        def _on_help(self):
            """Open the command help page."""
            from chimerax.core.commands import run
            try:
                run(self.session, "help %s" % self.help)
            except Exception:
                run(self.session, "help mdsmooth")

        def _on_spectrum_toggled(self, checked):
            if self._preview_plot is not None:
                try:
                    self._preview_plot.set_show_spectrum(bool(checked))
                except Exception:
                    self._preview_plot = None

        # --- stage 2: build ----------------------------------------------

        def _on_build(self):
            from Qt.QtWidgets import QApplication
            from chimerax.core.commands import run

            model = self._current_model()
            if model is None:
                self._status.setText(
                    "Pick a trajectory first (load one with multiple frames).")
                return

            mid = model.id_string
            method = self._method_menu.currentData() or "corkscrew"

            if self._kinetic():
                try:
                    states = self._states_value()
                    lag = self._lag_value()
                except ValueError as e:
                    self._status.setText(str(e))
                    return
                cmd = build_kinetic_command(mid, self._sel_edit.text(), states,
                                            method, lag=lag, make_morph=True)
            else:
                signal = self._signal()
                lag = None
                try:
                    cutoff_kw = cutoff_keyword(self._mode(), self._value_edit.text())
                    if signal in ("ic1", "deeptica"):
                        lag = self._lag_value()
                except ValueError as e:
                    self._status.setText(str(e))
                    return
                cmd = build_command(mid, self._sel_edit.text(), cutoff_kw, method,
                                    make_morph=True, show_plot=True,
                                    signal=signal, lag=lag)

            # The full interactive plot comes from the command; close the cheap
            # preview so the user isn't left with two plot windows.
            if self._preview_plot is not None:
                try:
                    self._preview_plot.delete()
                except Exception:
                    pass
                self._preview_plot = None

            kin = self._kinetic()
            if kin:
                self._status.setText(
                    "Building #%s - clustering into metastable states "
                    "(tICA → MSM → PCCA+), then morphing the tour…" % mid)
            elif method == "corkscrew":
                self._status.setText(
                    "Building #%s with corkscrew interpolation - this can take a "
                    "moment (more key frames means more segments to morph)…" % mid)
            else:
                self._status.setText("Building #%s…" % mid)
            QApplication.processEvents()

            before = set(self.session.models.list(type=AtomicStructure))
            try:
                run(self.session, cmd)
            except UserError as e:
                self._status.setText(str(e))
                return
            new = [m for m in self.session.models.list(type=AtomicStructure)
                   if m not in before]
            if not new:
                self._status.setText(
                    "No morph was built. In kinetic mode try fewer states or a "
                    "different lag; otherwise try more frames or a higher cutoff."
                    if kin else
                    "No morph was built (too few significant frames). "
                    "Try more frames or a higher cutoff.")
                return
            morph = new[0]

            if self._overlay_check.isChecked():
                try:
                    transp = parse_percent(self._transp_edit.text().strip())
                except ValueError:
                    transp = 50.0
                    self._transp_edit.setText("50")
                run(self.session, "transparency #%s %s target acbps" % (mid, transp))
                run(self.session, "color #%s gray target ac" % mid)
                run(self.session, "transparency #%s 0 target acbps" % morph.id_string)

            if kin:
                # A kinetic tour is a conformation sequence, not a timeline, so it
                # must NOT be linked to the raw slider (that would clamp/freeze it).
                # The command already opened its own Coordinate Sets slider.
                self._pair = None
                self._status.setText(
                    "Built #%s → #%s '%s' (%d frames) - a tour of the metastable "
                    "states (see the log for the state table). Play it with its OWN "
                    "Coordinate Sets slider (#%s), not the raw's. Measure on raw #%s."
                    % (mid, morph.id_string, morph.name, morph.num_coordsets,
                       morph.id_string, mid))
            else:
                self._pair = (model, morph)
                if self._link_check.isChecked():
                    run(self.session, "coordset slider #%s" % mid)
                    mirror_frame(model, morph)
                self._status.setText(
                    "Built #%s → #%s '%s' (%d frames). RMSD plot opened; click it "
                    "to jump to a frame. Use the Coordinate Sets slider (play / "
                    "pause / speed) - the smoothed copy follows it. Measure on the "
                    "raw #%s."
                    % (mid, morph.id_string, morph.name, morph.num_coordsets, mid))

        # --- playback link ------------------------------------------------

        def _on_frame(self, *args):
            pair = self._pair
            if pair is None or not self._link_check.isChecked():
                return
            raw, morph = pair
            if raw.deleted or morph.deleted:
                self._pair = None
                return
            mirror_frame(raw, morph)

        def _apply_transparency(self):
            """Re-fade the raw ghost when the typed value changes (no re-build)."""
            from chimerax.core.commands import run
            if self._pair is None or not self._overlay_check.isChecked():
                return
            raw, morph = self._pair
            if raw.deleted:
                return
            try:
                transp = parse_percent(self._transp_edit.text().strip())
            except ValueError:
                self._status.setText("Raw transparency must be a number from 0 to 100.")
                return
            run(self.session, "transparency #%s %s target acbps"
                % (raw.id_string, transp))

        # --- interactive add-frame relink --------------------------------

        def relink_morph(self, raw, morph):
            """Point the panel at a morph the RMSD plot just rebuilt.

            When the user adds a significant frame from the (post-build) plot, the
            command builds a fresh morph and closes the old one; this re-applies
            the overlay styling and the playback link to the new copy so the panel
            keeps driving it.
            """
            from chimerax.core.commands import run
            if morph is None or morph.deleted or raw.deleted:
                return
            if self._overlay_check.isChecked():
                try:
                    transp = parse_percent(self._transp_edit.text().strip())
                except ValueError:
                    transp = 50.0
                run(self.session, "transparency #%s %s target acbps"
                    % (raw.id_string, transp))
                run(self.session, "color #%s gray target ac" % raw.id_string)
                run(self.session, "transparency #%s 0 target acbps" % morph.id_string)
            self._pair = (raw, morph)
            if self._link_check.isChecked():
                run(self.session, "coordset slider #%s" % raw.id_string)
                mirror_frame(raw, morph)

        # --- cleanup ------------------------------------------------------

        def delete(self):
            for h in getattr(self, "_handlers", []):
                h.remove()
            self._handlers = []
            self._pair = None
            if self._preview_plot is not None:
                try:
                    self._preview_plot.delete()
                except Exception:
                    pass
                self._preview_plot = None
            super().delete()

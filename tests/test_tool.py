"""Unit tests for the GUI panel's pure helpers and the preview figure.

These run without ChimeraX or Qt -- ``tool.py`` guards its ChimeraX imports so
the module-level helpers import headless, and ``graph.build_figure`` renders to a
detached matplotlib Figure.

    pip install numpy scipy matplotlib pytest
    pytest ChimeraX-MDSmooth/tests
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tool  # noqa: E402
import filter as rf  # noqa: E402


def _synthetic_rmsd(n=1600, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    slow = 1.5 + 0.6 * np.sin(2 * np.pi * 3 * t) + 0.4 * np.sin(2 * np.pi * 7 * t)
    return slow + 0.15 * rng.standard_normal(n)


# --- resolve_cutoff_selection -------------------------------------------------

def test_resolve_frames():
    assert tool.resolve_cutoff_selection("frames", "50") == {"target_frames": 50}


def test_resolve_frames_rejects_below_two():
    with pytest.raises(ValueError):
        tool.resolve_cutoff_selection("frames", "1")


def test_resolve_frames_rejects_non_integer():
    with pytest.raises(ValueError):
        tool.resolve_cutoff_selection("frames", "4.5")


def test_resolve_cutoff_in_band():
    assert tool.resolve_cutoff_selection("cutoff", "0.03") == {"cutoff_frequency": 0.03}


def test_resolve_cutoff_out_of_band():
    with pytest.raises(ValueError):
        tool.resolve_cutoff_selection("cutoff", "0.9")


def test_resolve_power_fraction():
    assert tool.resolve_cutoff_selection("power", "0.95") == {"power_fraction": 0.95}


def test_resolve_power_out_of_range():
    with pytest.raises(ValueError):
        tool.resolve_cutoff_selection("power", "1.5")


def test_resolve_empty_value():
    with pytest.raises(ValueError):
        tool.resolve_cutoff_selection("frames", "   ")


def test_every_mode_has_field_metadata():
    # The panel indexes _MODE_FIELDS by the dropdown's mode ids; a missing entry
    # would KeyError at runtime, so keep them in lockstep.
    for mode in tool.CUTOFF_MODES:
        assert mode in tool._MODE_FIELDS


# --- resolved kwargs actually drive filter_rmsd -------------------------------

def test_resolved_kwargs_feed_filter_rmsd():
    rmsd = _synthetic_rmsd()
    for mode, value, checker in (
        ("frames", "30", lambda r: r.target_frames == 30),
        ("cutoff", "0.03", lambda r: abs(r.cutoff_frequency - 0.03) < 1e-9),
        ("power", "0.9", lambda r: r.target_frames is None),
    ):
        result = rf.filter_rmsd(rmsd, **tool.resolve_cutoff_selection(mode, value))
        assert checker(result)


# --- cutoff_keyword -----------------------------------------------------------

def test_cutoff_keyword_frames():
    assert tool.cutoff_keyword("frames", "50") == "targetFrames 50"


def test_cutoff_keyword_cutoff():
    assert tool.cutoff_keyword("cutoff", "0.03") == "cutoffFrequency 0.03"


def test_cutoff_keyword_power():
    assert tool.cutoff_keyword("power", "0.95") == "powerCutoff 0.95"


def test_cutoff_keyword_validates():
    with pytest.raises(ValueError):
        tool.cutoff_keyword("frames", "1")


# --- normalize_spec / build_command ------------------------------------------

def test_normalize_spec_prefixes_bare():
    assert tool.normalize_spec("1", ":436-441") == "#1:436-441"
    assert tool.normalize_spec("2", "@CA") == "#2@CA"


def test_normalize_spec_blank_and_full():
    assert tool.normalize_spec("1", "") == ""
    assert tool.normalize_spec("1", "  ") == ""
    assert tool.normalize_spec("2", "#3:5") == "#3:5"


def test_build_command_analysis_pass():
    cmd = tool.build_command("1", ":5", "targetFrames 40",
                             make_morph=False, show_plot=True)
    assert cmd.startswith("mdsmooth #1")
    assert "toAtoms #1:5" in cmd
    assert "targetFrames 40" in cmd
    assert "makeMorph false" in cmd
    assert "showPlot true" in cmd


def test_build_command_whole_structure_build():
    cmd = tool.build_command("1", "", "cutoffFrequency 0.03")
    assert "toAtoms" not in cmd
    assert "makeMorph true" in cmd
    assert "showPlot true" in cmd


def test_build_command_includes_signal_pc1():
    cmd = tool.build_command("1", "", "targetFrames 50", signal="pc1")
    assert "signal pc1" in cmd


def test_build_command_omits_default_signal_rmsd():
    cmd = tool.build_command("1", "", "targetFrames 50", signal="rmsd")
    assert "signal" not in cmd


def test_build_command_ic1_includes_signal_and_lag():
    cmd = tool.build_command("1", "", "targetFrames 50", signal="ic1", lag=15)
    assert "signal ic1" in cmd
    assert "lag 15" in cmd


def test_build_command_lag_only_for_ic1():
    # A lag passed with a non-ic1 signal is ignored (no stray "lag" keyword).
    cmd = tool.build_command("1", "", "targetFrames 50", signal="pc1", lag=15)
    assert "signal pc1" in cmd
    assert "lag" not in cmd


def test_build_command_ic1_without_lag_omits_lag():
    cmd = tool.build_command("1", "", "targetFrames 50", signal="ic1")
    assert "signal ic1" in cmd
    assert "lag" not in cmd


def test_build_command_dpca_includes_signal_no_lag():
    cmd = tool.build_command("1", "", "targetFrames 50", signal="dpca", lag=15)
    assert "signal dpca" in cmd
    assert "lag" not in cmd  # lag is IC1-only


def test_build_kinetic_command():
    cmd = tool.build_kinetic_command("1", ":5", 6, "linear", lag=12)
    assert cmd.startswith("mdsmooth #1")
    assert "toAtoms #1:5" in cmd
    assert "mode kinetic" in cmd
    assert "states 6" in cmd
    assert "lag 12" in cmd
    assert "method linear" in cmd
    assert "makeMorph true" in cmd
    # No cutoff keywords in kinetic mode.
    assert "targetFrames" not in cmd
    assert "cutoffFrequency" not in cmd


def test_build_kinetic_command_omits_lag_when_absent():
    cmd = tool.build_kinetic_command("2", "", 4)
    assert "mode kinetic" in cmd
    assert "states 4" in cmd
    assert "lag" not in cmd
    assert "toAtoms" not in cmd


def test_parse_cutoff_and_percent_still_validate():
    assert tool.parse_cutoff("0.02") == 0.02
    with pytest.raises(ValueError):
        tool.parse_cutoff("0.9")
    assert tool.parse_percent("50") == 50.0
    with pytest.raises(ValueError):
        tool.parse_percent("150")


# --- preview figure with spectrum --------------------------------------------

def test_clamp_cutoff_frequency():
    import graph  # noqa: E402
    assert graph.clamp_cutoff_frequency(None) is None
    assert graph.clamp_cutoff_frequency(0.03) == 0.03
    # Clicks on/below zero clamp up to a small positive; above Nyquist clamp down.
    assert 0.0 < graph.clamp_cutoff_frequency(0.0) < 0.01
    assert graph.clamp_cutoff_frequency(10.0) < 0.5


def test_build_figure_spectrum_adds_a_second_axis():
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure
    import graph  # noqa: E402

    rmsd = _synthetic_rmsd()
    result = rf.filter_rmsd(rmsd, target_frames=30)

    with_spec = Figure()
    graph.build_figure(result, figure=with_spec, show_spectrum=True)
    assert len(with_spec.axes) == 2

    without = Figure()
    graph.build_figure(result, figure=without, show_spectrum=False)
    assert len(without.axes) == 1

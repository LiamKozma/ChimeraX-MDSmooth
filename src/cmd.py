"""
The ``mdsmooth`` ChimeraX command.

This is the ChimeraX-facing layer.  It reads coordinates out of a loaded
trajectory, hands the per-frame RMSD series to the pure-Python core in
``filter.py``, reports the significant frames, and (optionally) builds a new
morph trajectory that keeps only those frames.
"""

import os
from contextlib import contextmanager

import numpy as np

from chimerax.core.commands import (
    CmdDesc,
    BoolArg,
    EnumOf,
    FloatArg,
    IntArg,
    ListOf,
    StringArg,
)
from chimerax.core.errors import UserError
from chimerax.atomic import AtomicStructureArg, AtomsArg


# Residue names treated as bulk solvent / free ions.  These diffuse freely and,
# if left in the default RMSD selection, swamp the macromolecule's own motion
# (a box of water moves far more than the protein does).  They are dropped from
# the *default* fit selection only, an explicit ``toAtoms`` is always honoured
# verbatim, and ligands / nonstandard residues are never dropped (see
# :func:`_default_fit_atoms`).
_SOLVENT_RESIDUES = frozenset({
    "HOH", "WAT", "H2O", "DOD", "D2O", "TIP", "TIP2", "TIP3", "TIP4", "TIP5",
    "T3P", "T4P", "SPC", "SPCE", "OPC", "PL3",
})
_ION_RESIDUES = frozenset({
    "NA", "NA+", "SOD", "CL", "CL-", "CLA", "K", "K+", "POT", "LI", "LI+",
    "RB", "CS", "MG", "MG2", "CA", "CA2", "ZN", "ZN2", "FE", "FE2", "FE3",
    "MN", "MN2", "CU", "CU1", "CU2", "CO", "NI", "CD", "HG", "BR", "BR-",
    "IOD", "I", "F", "F-", "IB", "IP", "IM",
})


def _default_fit_atoms(structure):
    """Atoms used for the RMSD fit when the user does not pass ``toAtoms``.

    Everything except bulk solvent and monatomic ions, so protein, nucleic
    acids, ligands, and nonstandard residues all contribute by default, but
    diffusing water and counter-ions do not.  Falls back to every atom if the
    filter would remove all of them (or none of them), so the command still
    works on solvent-only or already-stripped systems.
    """
    atoms = structure.atoms
    skip = _SOLVENT_RESIDUES | _ION_RESIDUES
    names = atoms.residues.names
    mask = np.array([str(n).upper() not in skip for n in names], dtype=bool)
    if mask.any() and not mask.all():
        return atoms.filter(mask)
    return atoms


@contextmanager
def _quiet_coordset_walk(structure):
    """Walk the coordsets without spamming atomic's "changes" trigger.

    Prefers ``structure.suppress_coordset_change_notifications()`` when the
    running ChimeraX provides it (per reviewer feedback), then the
    ``active_coordset_change_notify`` flag, and finally a plain save/restore,
    so the frame walk stays quiet where the API allows and still works on builds
    that lack it.  In every case the active coordset is restored on exit, so the
    command has no visible side effect on the input trajectory.
    """
    suppressor = getattr(structure, "suppress_coordset_change_notifications", None)
    if callable(suppressor):
        with suppressor():
            yield
        return

    saved_id = structure.active_coordset_id
    has_flag = hasattr(structure, "active_coordset_change_notify")
    prev_notify = None
    if has_flag:
        prev_notify = structure.active_coordset_change_notify
        structure.active_coordset_change_notify = False
    try:
        yield
    finally:
        if has_flag:
            structure.active_coordset_change_notify = prev_notify
        structure.active_coordset_id = saved_id


def _frame_coords(structure, atoms, align_atoms, all_atoms):
    """Return per-coordset coordinates for the fit, alignment, and all atoms.

    ``atoms`` drives the RMSD analysis (which frames are significant).
    ``align_atoms`` is optional and only used for the *opt-in* de-tumbling of the
    morph; when it is ``None`` no alignment coordinates are collected and the
    morph keeps the raw simulation coordinates (staying registered on, and
    drifting with, the input trajectory).

    Reads every coordset once inside :func:`_quiet_coordset_walk`, which keeps
    the frame walk from firing atomic's "changes" trigger for every coordset we
    touch and restores the active coordset on exit, so the command has no visible
    side effect on the input trajectory.
    """
    fit_frames = []
    align_frames = [] if align_atoms is not None else None
    full_frames = []
    with _quiet_coordset_walk(structure):
        for cs_id in structure.coordset_ids:
            structure.active_coordset_id = cs_id
            fit_frames.append(atoms.coords.copy())
            if align_atoms is not None:
                align_frames.append(align_atoms.coords.copy())
            full_frames.append(all_atoms.coords.copy())
    return fit_frames, align_frames, full_frames


def _alignment_transforms(align_frames, ref_index):
    """Per-frame rigid transforms that superimpose ``align_atoms`` onto the
    reference frame.  Applied to all atoms in the morph, these hold the alignment
    selection still, used only when the caller opts into de-tumbling by passing
    ``alignAtoms``.  By default the morph is not aligned at all.
    """
    from chimerax.geometry import align_points

    ref = align_frames[ref_index]
    return [align_points(coords, ref)[0] for coords in align_frames]


def _rmsd_series(fit_frames, ref_index):
    """Best-fit RMSD of every frame to the reference frame, over the fit atoms.

    This is the series the filter uses to pick significant frames.  The
    superposition used to *build* the morph is computed separately from the
    alignment atoms (see :func:`_alignment_transforms`).
    """
    from chimerax.geometry import align_points

    ref = fit_frames[ref_index]
    rmsds = np.empty(len(fit_frames), dtype=float)
    for i, coords in enumerate(fit_frames):
        # align_points returns (Place, rmsd) for the least-squares fit of
        # `coords` onto `ref`; we keep only the residual RMSD here.
        _, rmsds[i] = align_points(coords, ref)
    return rmsds


def _read_frame_coords(structure, atoms, frame_indices):
    """Read all-atom coordinates for specific 0-based frame indices.

    Used by the interactive "add significant frame" rebuild, which only needs
    the handful of key frames again rather than the whole trajectory.  Reads
    inside :func:`_quiet_coordset_walk`, so it restores the active coordset and
    fires no "changes" trigger.
    """
    ids = structure.coordset_ids
    out = []
    with _quiet_coordset_walk(structure):
        for f in frame_indices:
            structure.active_coordset_id = ids[int(f)]
            out.append(atoms.coords.copy())
    return out


def _fit_frames_only(structure, fit_atoms):
    """Read only the fit atoms' coordinates for every frame (no all-atom read).

    The RMSD analysis needs just the fit atoms, so the panel's cheap "Analyze"
    step uses this instead of :func:`_frame_coords`. It defers reading the far
    larger all-atom coordinates until a morph is actually built.  Reads inside
    :func:`_quiet_coordset_walk`, so it restores the active coordset and fires no
    "changes" trigger.
    """
    frames = []
    with _quiet_coordset_walk(structure):
        for cs_id in structure.coordset_ids:
            structure.active_coordset_id = cs_id
            frames.append(fit_atoms.coords.copy())
    return frames


def compute_rmsd_series(structure, to_atoms=None, reference=1):
    """Per-frame best-fit RMSD to the reference frame, the analysis series.

    Public helper for the GUI panel's two-stage flow: it computes the exact
    series that :func:`mdsmooth` filters, without building the (expensive)
    morph, so the panel can cache it and re-filter live as the user dials the
    cutoff.  Uses the same default fit-atom selection as the command (everything
    but bulk solvent and free ions).  Returns ``(rmsd_array, fit_atom_count)``.
    """
    all_atoms = structure.atoms
    if to_atoms is None:
        fit_atoms = _default_fit_atoms(structure)
    else:
        fit_atoms = to_atoms.intersect(all_atoms)
        if len(fit_atoms) == 0:
            raise UserError("None of the specified atoms belong to %s." % structure)
    n_frames = structure.num_coordsets
    if not (1 <= reference <= n_frames):
        raise UserError(
            "reference frame %d is out of range (1-%d)." % (reference, n_frames)
        )
    fit_frames = _fit_frames_only(structure, fit_atoms)
    rmsds = _rmsd_series(fit_frames, reference - 1)
    return rmsds, len(fit_atoms)


def compute_pc1_series(structure, to_atoms=None, reference=1):
    """PC1 projection over frames, the "largest collective motion" signal.

    The essential-dynamics analog of :func:`compute_rmsd_series` for the panel's
    two-stage flow: reads only the fit atoms (deferring the all-atom read) and
    projects them onto their top principal component.  Returns
    ``(pc1_array, fit_atom_count)``.
    """
    from .filter import principal_component_series

    all_atoms = structure.atoms
    if to_atoms is None:
        fit_atoms = _default_fit_atoms(structure)
    else:
        fit_atoms = to_atoms.intersect(all_atoms)
        if len(fit_atoms) == 0:
            raise UserError("None of the specified atoms belong to %s." % structure)
    n_frames = structure.num_coordsets
    if not (1 <= reference <= n_frames):
        raise UserError(
            "reference frame %d is out of range (1-%d)." % (reference, n_frames)
        )
    fit_frames = _fit_frames_only(structure, fit_atoms)
    pc1 = principal_component_series(fit_frames, reference_index=reference - 1)[:, 0]
    return pc1, len(fit_atoms)


def compute_ic1_series(structure, to_atoms=None, reference=1, lag=None):
    """IC1 projection over frames, the "slowest collective motion" signal.

    The tICA analog of :func:`compute_pc1_series` for the panel's two-stage flow:
    reads only the fit atoms (deferring the all-atom read) and projects them onto
    their slowest time-lagged independent component.  ``lag`` defaults to
    :data:`~.filter.DEFAULT_TICA_LAG`.  Returns ``(ic1_array, fit_atom_count)``.
    """
    from .filter import tica_series, DEFAULT_TICA_LAG

    if lag is None:
        lag = DEFAULT_TICA_LAG
    all_atoms = structure.atoms
    if to_atoms is None:
        fit_atoms = _default_fit_atoms(structure)
    else:
        fit_atoms = to_atoms.intersect(all_atoms)
        if len(fit_atoms) == 0:
            raise UserError("None of the specified atoms belong to %s." % structure)
    n_frames = structure.num_coordsets
    if not (1 <= reference <= n_frames):
        raise UserError(
            "reference frame %d is out of range (1-%d)." % (reference, n_frames)
        )
    if not (1 <= lag < n_frames):
        raise UserError(
            "lag %d is out of range (1-%d)." % (lag, n_frames - 1)
        )
    fit_frames = _fit_frames_only(structure, fit_atoms)
    ic1 = tica_series(fit_frames, lag=lag, reference_index=reference - 1)[:, 0]
    return ic1, len(fit_atoms)


def _backbone_dihedral_quartets(structure, wanted_residues):
    """Atom quartets for every backbone phi/psi whose central residue is wanted.

    phi(i) = C(i-1)-N(i)-CA(i)-C(i); psi(i) = N(i)-CA(i)-C(i)-N(i+1).  Walks each
    chain in sequence order so the i-1 / i+1 neighbours are correct, and keeps a
    dihedral only when every atom it needs exists.  ``wanted_residues`` scopes it
    to the fit selection's residues (a set of Residue objects).
    """
    quartets = []
    chains = getattr(structure, "chains", None)
    if not chains:
        return quartets
    for chain in chains:
        residues = [r for r in chain.existing_residues if r is not None]
        for i, r in enumerate(residues):
            if r not in wanted_residues:
                continue
            n = r.find_atom("N")
            ca = r.find_atom("CA")
            c = r.find_atom("C")
            if not (n and ca and c):
                continue
            if i > 0:
                prev_c = residues[i - 1].find_atom("C")
                if prev_c:
                    quartets.append((prev_c, n, ca, c))   # phi
            if i < len(residues) - 1:
                next_n = residues[i + 1].find_atom("N")
                if next_n:
                    quartets.append((n, ca, c, next_n))   # psi
    return quartets


def _dpca_series_for(structure, fit_atoms):
    """Per-frame dPC1 (dihedral-PCA) series for the fit selection's residues.

    Extracts the backbone phi/psi angles for every frame (reusing the tested
    :func:`~.filter.dihedral_angles` and :func:`~.filter.dihedral_pca_series`
    core) and returns ``(dpc1_array, n_dihedrals)``.  Raises ``UserError`` when
    the selection has no backbone dihedrals (e.g. a bare ligand), so the caller
    can suggest a different signal.
    """
    from chimerax.atomic import Atoms
    from .filter import dihedral_angles, dihedral_pca_series

    wanted = set(fit_atoms.unique_residues)
    quartets = _backbone_dihedral_quartets(structure, wanted)
    if not quartets:
        raise UserError(
            "No backbone phi/psi dihedrals found in the selection; the dPCA "
            "signal needs protein backbone atoms. Try the rmsd, pc1, or ic1 "
            "signal instead."
        )

    # Unique atoms + per-quartet index arrays, so each frame is a single coord
    # read plus a vectorized dihedral evaluation.
    order = {}
    uniq = []
    for quartet in quartets:
        for a in quartet:
            if a not in order:
                order[a] = len(uniq)
                uniq.append(a)
    q_atoms = Atoms(uniq)
    idx = [np.array([order[q[j]] for q in quartets], dtype=int) for j in range(4)]

    angles = []
    with _quiet_coordset_walk(structure):
        for cs_id in structure.coordset_ids:
            structure.active_coordset_id = cs_id
            coords = q_atoms.coords
            angles.append(
                dihedral_angles(coords[idx[0]], coords[idx[1]],
                                coords[idx[2]], coords[idx[3]])
            )
    dih = np.asarray(angles, dtype=float)      # (n_frames, n_dihedrals)
    dpc1 = dihedral_pca_series(dih)[:, 0]
    return dpc1, len(quartets)


def compute_dpca_series(structure, to_atoms=None, reference=1):
    """dPC1 projection over frames, the "slowest internal (dihedral) motion" signal.

    The dihedral-PCA analog of :func:`compute_pc1_series` for the panel's
    two-stage flow.  dPCA works on internal coordinates (backbone phi/psi), so it
    needs no structural alignment and is insensitive to global tumbling.  Returns
    ``(dpc1_array, n_dihedrals)``, note the second value is the dihedral count,
    not an atom count.
    """
    all_atoms = structure.atoms
    if to_atoms is None:
        fit_atoms = _default_fit_atoms(structure)
    else:
        fit_atoms = to_atoms.intersect(all_atoms)
        if len(fit_atoms) == 0:
            raise UserError("None of the specified atoms belong to %s." % structure)
    return _dpca_series_for(structure, fit_atoms)


def _key_coords(full_frames, transforms, frames):
    """All-atom coordinates of each significant frame, the morph's key frames.

    By default (``transforms is None``) each key frame keeps its raw simulation
    coordinates, so the morph stays registered on the input trajectory: it drifts
    and tumbles exactly as the raw does, and only the high-frequency jitter is
    smoothed out.  When ``transforms`` is supplied (the caller passed
    ``alignAtoms``) each frame is first de-tumbled onto the reference, removing
    global motion, an opt-in, not the default.
    """
    if transforms is None:
        return [np.asarray(full_frames[f], dtype=np.float64) for f in frames]
    return [
        np.asarray(transforms[f].transform_points(full_frames[f]), dtype=np.float64)
        for f in frames
    ]


def _linear_morph_coords(key_coords, steps):
    """Straight-line interpolation between consecutive key frames.

    Uses each segment's ``steps`` value so the total frame count, and therefore
    the apparent passage of time, matches the original trajectory.  Fast and
    dependency-free, but a straight line in Cartesian space can stretch bonds
    across large conformational jumps (see :func:`_corkscrew_morph_coords`).
    """
    morph_coords = [key_coords[0]]
    for i in range(1, len(key_coords)):
        a = key_coords[i - 1]
        b = key_coords[i]
        k = int(steps[i])
        if k <= 0:
            k = 1
        # s = 1..k so the segment ends exactly on frame b (no duplicated frame).
        for s in range(1, k + 1):
            t = s / k
            morph_coords.append((1.0 - t) * a + t * b)
    return morph_coords


def _corkscrew_morph_coords(session, structure, key_coords, steps):
    """Interpolate between key frames with ChimeraX's own ``morph`` engine.

    For each segment we build two single-frame copies of the structure (the
    bracketing key frames) and run ``morph ... method corkscrew``, which
    interpolates the rigid-body change between them as a screw motion instead of
    a straight Cartesian slide, so domains rotate naturally rather than atoms
    cutting through each other, which is what makes the default linear morph look
    unnatural on large motions.  ``corkscrew`` is already ``morph``'s default
    method, so the interpolation survives even the minimal fallback command in
    :func:`_run_segment_morph`.

    Each segment is morphed with its own ``steps`` count, so per-segment timing
    is preserved just like the linear path.  The segments are stitched back
    together, dropping only the single shared key frame at each join.

    Uses the ChimeraX ``morph`` command, so it only runs inside ChimeraX; any
    failure raises so the caller can fall back to :func:`_linear_morph_coords`.
    """
    from chimerax.core.commands import run

    # A single-frame template makes each per-key copy cheap: copying the full
    # input trajectory (all its coordsets) once per key frame would be wasteful.
    template = structure.copy("mdsmooth morph template")
    template.add_coordsets(
        np.asarray([key_coords[0]], dtype=np.float64), replace=True
    )

    morph_coords = [np.asarray(key_coords[0], dtype=np.float64)]
    temp_models = []
    try:
        for i in range(1, len(key_coords)):
            k = int(steps[i])
            if k <= 0:
                k = 1
            a = template.copy("mdsmooth kf %d" % (i - 1))
            b = template.copy("mdsmooth kf %d" % i)
            a.atoms.coords = np.asarray(key_coords[i - 1], dtype=np.float64)
            b.atoms.coords = np.asarray(key_coords[i], dtype=np.float64)
            session.models.add([a, b])
            temp_models.append(a)
            temp_models.append(b)

            # The morph command returns None, so grab the model it created by
            # diffing the session's structures around the call.
            morph_model = _run_segment_morph(session, a, b, k)
            temp_models.append(morph_model)

            seg = []
            for cid in morph_model.coordset_ids:
                morph_model.active_coordset_id = cid
                seg.append(morph_model.atoms.coords.copy())

            for m in (a, b, morph_model):
                if not m.deleted:
                    run(session, "close #%s" % m.id_string)
                if m in temp_models:
                    temp_models.remove(m)

            # Drop a segment's first frame only when it merely repeats the join
            # point, so consecutive segments don't duplicate the shared key
            # frame.  At most that one boundary frame is ever dropped (never
            # interior frames), so per-segment timing is preserved.
            start = 0
            if seg and np.allclose(seg[0], morph_coords[-1], atol=1e-3):
                start = 1
            morph_coords.extend(seg[start:])
    finally:
        for m in temp_models:
            if not m.deleted:
                run(session, "close #%s" % m.id_string)
        _safe_delete(template)

    if len(morph_coords) < 2:
        raise RuntimeError("corkscrew morph produced no interpolated frames")
    return morph_coords


def _run_segment_morph(session, a, b, k):
    """Morph the pair ``a`` -> ``b`` over ``k`` frames; return the new model.

    The ``morph`` command returns ``None`` even when it succeeds, so the created
    trajectory is found by diffing the session's structures around the call
    rather than trusting a return value.

    Tries the explicit ``method corkscrew play false slider false`` form first
    (quietest, no auto-play or per-segment slider), then progressively simpler
    commands if a ChimeraX version rejects one of those keywords.  The bare form
    still interpolates with corkscrew, since that is ``morph``'s default method,
    so the natural rotation survives the fallback.  Raises if no command
    produced a trajectory, so the caller reverts to linear interpolation.
    """
    from chimerax.core.commands import run
    from chimerax.atomic import AtomicStructure

    commands = (
        "morph #%s #%s frames %d method corkscrew play false slider false"
        % (a.id_string, b.id_string, k),
        "morph #%s #%s frames %d method corkscrew play false"
        % (a.id_string, b.id_string, k),
        "morph #%s #%s frames %d" % (a.id_string, b.id_string, k),
    )
    last_error = RuntimeError("morph command produced no result")
    for cmd in commands:
        before = set(session.models.list(type=AtomicStructure))
        try:
            run(session, cmd)
        except Exception as e:
            last_error = e
            continue
        new = [m for m in session.models.list(type=AtomicStructure)
               if m not in before and m is not a and m is not b]
        if new:
            return new[0]
        last_error = RuntimeError("morph command created no trajectory")
    raise last_error


def _safe_delete(model):
    """Delete a not-yet-added helper model, ignoring any teardown error.

    The template models are never added to the session, so their disposal must
    not be allowed to sink an otherwise-successful morph.
    """
    try:
        if model is not None and not model.deleted:
            model.delete()
    except Exception:
        pass


def _finalize_morph(session, structure, morph_coords, name):
    """Wrap a list of per-frame coordinate arrays in a new trajectory model."""
    coordset_array = np.asarray(morph_coords, dtype=np.float64)
    new_structure = structure.copy(name)
    # Replace the copied coordsets with the interpolated morph.
    new_structure.add_coordsets(coordset_array, replace=True)
    new_structure.active_coordset_id = new_structure.coordset_ids[0]
    session.models.add([new_structure])
    return new_structure


def _build_morph(session, structure, key_coords, steps, name, method):
    """Create a new AtomicStructure whose coordsets are the filtered morph.

    ``method`` selects the interpolation: ``"corkscrew"`` routes through
    ChimeraX's morph engine (natural rotations) and ``"linear"`` uses the
    dependency-free straight-line path.  Corkscrew falls back to linear if the
    morph engine is unavailable or errors, so a morph is always produced.
    """
    if method == "corkscrew":
        try:
            morph_coords = _corkscrew_morph_coords(
                session, structure, key_coords, steps
            )
        except Exception as e:
            session.logger.warning(
                "Corkscrew morph failed (%s); falling back to linear "
                "interpolation." % e
            )
            morph_coords = _linear_morph_coords(key_coords, steps)
    else:
        morph_coords = _linear_morph_coords(key_coords, steps)

    return _finalize_morph(session, structure, morph_coords, name)


def _quote(path):
    """Quote a path for embedding in a ChimeraX command line."""
    return '"%s"' % path.replace('"', '\\"')


def _save_frame_pdbs(session, structure, key_coords, frames, directory, name):
    """Write one PDB per significant frame into ``directory``.

    These are the exact key frames the morph interpolates between (de-tumbled if
    ``alignAtoms`` was given, raw otherwise), so they can be inspected, handed to
    another tool, or morphed yourself.  File names carry the 1-based frame number
    of the original trajectory.  Returns the list of paths written.
    """
    from chimerax.core.commands import run

    directory = os.path.expanduser(directory)
    os.makedirs(directory, exist_ok=True)

    template = structure.copy("mdsmooth pdb template")
    template.add_coordsets(
        np.asarray([key_coords[0]], dtype=np.float64), replace=True
    )
    paths = []
    try:
        for i, coords in enumerate(key_coords):
            ks = template.copy(name)
            ks.atoms.coords = np.asarray(coords, dtype=np.float64)
            session.models.add([ks])
            path = os.path.join(
                directory, "%s_frame%04d.pdb" % (name, int(frames[i]) + 1)
            )
            run(session, "save %s models #%s" % (_quote(path), ks.id_string))
            run(session, "close #%s" % ks.id_string)
            paths.append(path)
    finally:
        _safe_delete(template)
    return paths


def _report(session, structure, result, made_morph, method=None,
            fit_atom_count=None, saved_paths=None, signal_label="RMSD",
            lag_robustness=None):
    """Log a summary table of the significant frames."""
    lines = [
        "MDSmooth on %s" % structure,
        "  frames analyzed: %d" % result.raw.size,
    ]
    _signal_gloss = {
        "PC1": "largest collective motion",
        "IC1": "slowest collective motion (tICA)",
        "dPC1": "largest internal (dihedral) motion",
        "DeepTICA": "slowest collective motion (learned, nonlinear)",
    }
    if signal_label and signal_label != "RMSD":
        lines.append("  signal: %s (%s)"
                     % (signal_label, _signal_gloss.get(signal_label, signal_label)))
    if fit_atom_count is not None:
        lines.append("  fit atoms: %d" % fit_atom_count)
    lines += [
        "  cutoff frequency: %.6g cycles/frame" % result.cutoff_frequency,
        "  significant frames: %d" % result.frames.size,
    ]
    if result.cosine_content is not None:
        note = "  (high - signal resembles a random walk; extrema may be " \
               "undersampling artifacts)" if result.cosine_content_high else ""
        lines.append("  cosine content: %.2f%s" % (result.cosine_content, note))
    if lag_robustness is not None:
        from .filter import TICA_LAG_ROBUST_WARN
        lags, _corrs, min_corr = lag_robustness
        note = "  (low - IC1 shape is lag-sensitive; may be undersampled)" \
               if min_corr < TICA_LAG_ROBUST_WARN else ""
        lines.append("  IC1 lag robustness: %.2f across lags %s%s"
                     % (min_corr, ", ".join(str(x) for x in lags), note))
    if getattr(result, "target_frames", None) is not None:
        lines.append(
            "  (cutoff solved for a target of %d significant frames)"
            % result.target_frames
        )
    lines += [
        "",
        "  %-8s %-10s %-6s %s" % ("Frame", signal_label, "Type", "Interp steps"),
    ]
    kind_label = {"max": "Maxima", "min": "Minima", "end": "End", "user": "User"}
    for i, frame in enumerate(result.frames):
        lines.append(
            "  %-8d %-10.4f %-6s %d"
            % (
                int(frame),
                float(result.filtered[frame]),
                kind_label.get(result.kinds[i], result.kinds[i]),
                int(result.steps[i]),
            )
        )
    if made_morph is not None:
        lines.append("")
        lines.append("  Created morph: %s (%d frames, %s interpolation)"
                     % (made_morph, made_morph.num_coordsets, method or "linear"))
        lines.append("  Play it:  coordset slider #%s" % made_morph.id_string)
        lines.append("  Grey-ghost overlay (raw faded under the smoothed copy): use "
                     "the MDSmooth panel, or  transparency #%s 60 target acbps ; "
                     "color #%s gray target ac"
                     % (structure.id_string, structure.id_string))
    if saved_paths:
        lines.append("")
        lines.append("  Saved %d significant-frame PDB(s) to %s"
                     % (len(saved_paths), os.path.dirname(saved_paths[0])))
    session.logger.info("\n".join(lines))


def _report_kinetic(session, structure, kr, made_morph, method,
                    fit_atom_count, saved_paths):
    """Log a summary of the metastable-state key frames (kinetic mode)."""
    lines = [
        "MDSmooth on %s  (kinetic / MSM mode)" % structure,
        "  frames analyzed: %d" % kr.frame_macrostate.size,
        "  fit atoms: %d" % fit_atom_count,
        "  MSM lag: %d frames   microstates: %d" % (kr.lag, kr.n_microstates),
        "  metastable states (key frames): %d" % kr.n_states,
        "",
        "  %-6s %-8s %-12s %s" % ("State", "Frame", "IC1", "State size"),
    ]
    for i, frame in enumerate(kr.frames):
        size = int((kr.frame_macrostate == kr.frame_macrostate[frame]).sum())
        lines.append("  %-6d %-8d %-12.4f %d"
                     % (i + 1, int(frame), float(kr.order_values[i]), size))
    if made_morph is not None:
        lines.append("")
        lines.append("  Created morph: %s (%d frames, %s interpolation) - a tour "
                     "of the %d metastable states (superimposed, ordered by a "
                     "nearest-neighbour path through tICA space)."
                     % (made_morph, made_morph.num_coordsets, method or "linear",
                        kr.n_states))
        lines.append("  Play it:  coordset slider #%s" % made_morph.id_string)
    if saved_paths:
        lines.append("")
        lines.append("  Saved %d state PDB(s) to %s"
                     % (len(saved_paths), os.path.dirname(saved_paths[0])))
    session.logger.info("\n".join(lines))


def _run_kinetic_mode(session, structure, fit_frames, full_frames,
                      align_transforms, fit_atom_count, ref_index, lag, states,
                      microstates, state_steps, tica_dim, name, method,
                      make_morph, save_pdbs, show_plot):
    """Kinetic (MSM/PCCA+) keyframe path: metastable states instead of extrema.

    Reduces the trajectory with tICA, hands the slow components to
    :func:`~.kinetic.kinetic_keyframes`, then reuses the same morph builder and
    PDB export as the extrema path, only the choice of key frames differs.
    """
    from .filter import tica_series, DEFAULT_TICA_LAG
    from .kinetic import (
        kinetic_keyframes, DEFAULT_N_STATES, DEFAULT_N_MICROSTATES,
        DEFAULT_STATE_STEPS,
    )

    tica_lag = DEFAULT_TICA_LAG if lag is None else int(lag)
    n_frames = len(fit_frames)
    if not (1 <= tica_lag < n_frames):
        raise UserError("lag %d is out of range (1-%d)." % (tica_lag, n_frames - 1))
    dim = 4 if tica_dim is None else int(tica_dim)
    if dim < 1:
        raise UserError("ticaDim must be at least 1.")
    n_states = DEFAULT_N_STATES if states is None else int(states)
    n_micro = DEFAULT_N_MICROSTATES if microstates is None else int(microstates)
    seg_steps = DEFAULT_STATE_STEPS if state_steps is None else int(state_steps)
    if seg_steps < 1:
        raise UserError("stateSteps must be at least 1.")

    components = tica_series(
        fit_frames, lag=tica_lag, n_components=dim, reference_index=ref_index
    )
    try:
        kr = kinetic_keyframes(
            components, n_states=n_states, n_microstates=n_micro, lag=tica_lag
        )
    except ImportError as e:
        raise UserError(str(e))
    except ValueError as e:
        raise UserError("kinetic mode: %s" % e)

    # Kinetic representatives are drawn from all over the trajectory, so unlike the
    # extrema path they MUST be superimposed, otherwise the tour shows the
    # molecule drifting/tumbling through the box rather than its conformational
    # change.  Default to aligning on the fit atoms; honour an explicit alignAtoms
    # selection if the caller gave one.
    if align_transforms is None:
        align_transforms = _alignment_transforms(fit_frames, ref_index)
    key_coords = _key_coords(full_frames, align_transforms, kr.frames)

    saved_paths = None
    if save_pdbs:
        saved_paths = _save_frame_pdbs(
            session, structure, key_coords, kr.frames, save_pdbs, name
        )

    made_morph = None
    if make_morph:
        # Equal steps per segment: representatives are ordered by conformation,
        # not chronology, so real per-frame timing does not apply.
        steps = np.full(kr.frames.size, seg_steps, dtype=int)
        steps[0] = 0
        morph_name = "%s (kinetic, %d states)" % (name, kr.n_states)
        made_morph = _build_morph(
            session, structure, key_coords, steps, morph_name, method
        )
        # The kinetic tour is NOT time-aligned to the raw (its frames are a
        # conformation sequence, not a timeline), so give it its own playback
        # slider rather than relying on the raw link.
        if getattr(session.ui, "is_gui", False):
            from chimerax.core.commands import run
            run(session, "coordset slider #%s" % made_morph.id_string)

    _report_kinetic(session, structure, kr, made_morph, method,
                    fit_atom_count, saved_paths)
    return kr


def _make_rebuild(session, structure, all_atoms, rmsds, cutoff_frequency,
                  power_cutoff, order, include_ends, align_transforms,
                  name, method, make_morph, initial_morph):
    """Build the callback the plot uses to add significant frames interactively.

    Returns ``rebuild(extra_frames_0based) -> (FilterResult, morph_or_None)``.
    It recomputes the significant frames with the user's picks merged in and
    rebuilds the morph in place, reusing the exact cutoff frequency, so the
    filtered curve is unchanged and only the key-frame set differs.  The morph
    it built on the previous call (starting with the one the command just made)
    is closed so morphs do not pile up.  Only the handful of key frames are read
    back from the trajectory, so a rebuild is cheap.
    """
    from .filter import filter_rmsd
    from chimerax.core.commands import run

    # Mutable cell holding the morph currently on screen, so each rebuild can
    # close the one it replaces.
    state = {"morph": initial_morph}

    def rebuild(extra_frames_0based):
        new_result = filter_rmsd(
            rmsds, sampling_rate=1.0, power_fraction=power_cutoff,
            cutoff_frequency=cutoff_frequency, order=order,
            include_ends=include_ends, extra_frames=extra_frames_0based,
        )
        new_morph = None
        if make_morph and new_result.frames.size >= 2:
            raw_key = _read_frame_coords(structure, all_atoms, new_result.frames)
            if align_transforms is None:
                key_coords = [np.asarray(c, dtype=np.float64) for c in raw_key]
            else:
                key_coords = [
                    np.asarray(align_transforms[int(f)].transform_points(c),
                               dtype=np.float64)
                    for f, c in zip(new_result.frames, raw_key)
                ]
            new_morph = _build_morph(
                session, structure, key_coords, new_result.steps, name, method
            )
            old = state.get("morph")
            if old is not None and not old.deleted:
                # The new morph is a copy of the raw model, which the overlay may
                # have recoloured grey and made transparent; inherit the previous
                # morph's colours instead so its appearance is preserved across
                # rebuilds (the panel re-applies opacity via relink_morph).
                try:
                    new_morph.atoms.colors = old.atoms.colors
                    new_morph.residues.ribbon_colors = old.residues.ribbon_colors
                except Exception:
                    pass
                run(session, "close #%s" % old.id_string)
            state["morph"] = new_morph
        return new_result, new_morph

    return rebuild


def mdsmooth(
    session,
    structure,
    to_atoms=None,
    align_atoms=None,
    reference=1,
    signal="rmsd",
    lag=None,
    seeds=None,
    mode="extrema",
    states=None,
    microstates=None,
    state_steps=None,
    tica_dim=None,
    target_frames=None,
    power_cutoff=None,
    cutoff_frequency=None,
    order=5,
    include_ends=True,
    make_morph=True,
    method="corkscrew",
    save_pdbs=None,
    show_plot=True,
    name="smoothed",
    extra_frames=None,
):
    """Filter a trajectory's RMSD and build a morph of its significant frames.

    Parameters mirror the ``mdsmooth`` command keywords; see the command help
    (``src/docs/user/commands/mdsmooth.html``) for the user-facing docs.
    """
    from .filter import (
        filter_rmsd, principal_component_series, tica_series,
        tica_lag_robustness, DEFAULT_TICA_LAG, TICA_LAG_ROBUST_WARN,
    )

    # targetFrames, cutoffFrequency and powerCutoff are three ways of naming the
    # same low-pass cutoff.  Honour them in that priority order, and if the user
    # gave more than one, say which one won so the behaviour is never a surprise.
    specified = [
        label for label, value in (
            ("targetFrames", target_frames),
            ("cutoffFrequency", cutoff_frequency),
            ("powerCutoff", power_cutoff),
        ) if value is not None
    ]
    if len(specified) > 1:
        session.logger.warning(
            "mdsmooth: %s all set the same low-pass cutoff; using %s and "
            "ignoring %s."
            % (", ".join(specified), specified[0], ", ".join(specified[1:]))
        )
    if target_frames is not None:
        if target_frames < 2:
            raise UserError(
                "targetFrames must be at least 2 (a morph needs two key frames)."
            )
        cutoff_frequency = None
        power_cutoff = None
    elif cutoff_frequency is not None:
        power_cutoff = None

    if structure.num_coordsets < 2:
        raise UserError(
            "%s has only %d coordinate set(s); mdsmooth needs a trajectory "
            "with multiple frames." % (structure, structure.num_coordsets)
        )

    all_atoms = structure.atoms
    if to_atoms is None:
        # Default: everything but bulk solvent / free ions, so the RMSD tracks
        # the macromolecule and any ligand or nonstandard residue rather than
        # diffusing water (see _default_fit_atoms).
        fit_atoms = _default_fit_atoms(structure)
    else:
        # Restrict the fit selection to atoms of this structure.
        fit_atoms = to_atoms.intersect(all_atoms)
        if len(fit_atoms) == 0:
            raise UserError(
                "None of the specified atoms belong to %s." % structure
            )

    # Optional de-tumbling.  By DEFAULT the morph keeps the raw simulation
    # coordinates, so it stays registered on the input trajectory and drifts /
    # tumbles right along with it, the smoothing only removes jitter.  Pass
    # alignAtoms only if you deliberately want to strip global motion by
    # superimposing those atoms onto the reference frame.
    if align_atoms is None:
        align_fit_atoms = None
    else:
        align_fit_atoms = align_atoms.intersect(all_atoms)
        if len(align_fit_atoms) == 0:
            raise UserError(
                "None of the alignAtoms belong to %s." % structure
            )

    n_frames = structure.num_coordsets
    if not (1 <= reference <= n_frames):
        raise UserError(
            "reference frame %d is out of range (1-%d)." % (reference, n_frames)
        )
    ref_index = reference - 1  # command is 1-based, arrays are 0-based

    # User-added significant frames arrive 1-based (like `reference`); validate
    # and convert to the 0-based indices the filter core works in.
    extra_indices = None
    if extra_frames:
        extra_indices = []
        for fr in extra_frames:
            if not (1 <= fr <= n_frames):
                raise UserError(
                    "extraFrames value %d is out of range (1-%d)."
                    % (fr, n_frames)
                )
            extra_indices.append(fr - 1)

    fit_frames, align_frames, full_frames = _frame_coords(
        structure, fit_atoms, align_fit_atoms, all_atoms
    )
    align_transforms = (
        _alignment_transforms(align_frames, ref_index)
        if align_frames is not None
        else None
    )

    # Kinetic (MSM) mode is a different keyframe-selection path, metastable
    # states rather than 1D-signal extrema.  It always clusters in tICA space and
    # reuses the frames + morph builder, but skips the filter / cutoff pipeline and
    # the 1D signal entirely.  Say so if the caller set keywords it will ignore, so
    # "signal rmsd mode kinetic" isn't silently misleading.
    if mode == "kinetic":
        ignored = []
        if signal != "rmsd":
            ignored.append("signal (kinetic always uses tICA)")
        for label, value in (("targetFrames", target_frames),
                             ("cutoffFrequency", cutoff_frequency),
                             ("powerCutoff", power_cutoff)):
            if value is not None:
                ignored.append(label)
        if ignored:
            session.logger.warning(
                "mdsmooth: kinetic mode clusters in tICA space and ignores %s. "
                "It is controlled by states / microstates / stateSteps / ticaDim / "
                "lag." % ", ".join(ignored)
            )
        return _run_kinetic_mode(
            session, structure, fit_frames, full_frames, align_transforms,
            len(fit_atoms), ref_index, lag, states, microstates, state_steps,
            tica_dim, name, method, make_morph, save_pdbs, show_plot,
        )

    # The 1D signal whose extrema become the significant frames: RMSD to the
    # reference, the projection onto the largest collective motion (PC1), or the
    # projection onto the slowest collective motion (IC1, via tICA).  The rest of
    # the pipeline (filter -> frames -> morph) is identical for all three, only
    # which frames are "significant" changes.
    lag_robustness = None
    if signal == "pc1":
        series = principal_component_series(
            fit_frames, reference_index=ref_index
        )[:, 0]
    elif signal == "ic1":
        tica_lag = DEFAULT_TICA_LAG if lag is None else int(lag)
        if not (1 <= tica_lag < n_frames):
            raise UserError(
                "lag %d is out of range (1-%d)." % (tica_lag, n_frames - 1)
            )
        series = tica_series(
            fit_frames, lag=tica_lag, reference_index=ref_index
        )[:, 0]
        # Lag-robustness sanity check: is the slow mode stable across lag times?
        lag_robustness = tica_lag_robustness(
            fit_frames, lag=tica_lag, reference_index=ref_index
        )
    elif signal == "dpca":
        series, _ = _dpca_series_for(structure, fit_atoms)
    elif signal == "deeptica":
        from .filter import reduced_coordinates
        from .learned import (
            run_deeptica, venv_ready, SEED_ROBUST_WARN, DEFAULT_N_SEEDS,
        )
        tica_lag = DEFAULT_TICA_LAG if lag is None else int(lag)
        if not (1 <= tica_lag < n_frames):
            raise UserError(
                "lag %d is out of range (1-%d)." % (tica_lag, n_frames - 1)
            )
        if not venv_ready():
            raise UserError(
                "The DeepTICA signal needs the optional learned-CV environment "
                "(PyTorch + mlcolvar), which is not installed. Install it once "
                "with:\n    mdsmooth installLearnedCV\n(this downloads several "
                "GB; it is a one-time setup)."
            )
        features = reduced_coordinates(fit_frames, ref_index)
        n_seeds = DEFAULT_N_SEEDS if seeds is None else int(seeds)
        session.logger.status(
            "Training DeepTICA (%d seeds) in the learned-CV venv…" % n_seeds
        )
        try:
            series, seed_min_corr = run_deeptica(
                features, lag=tica_lag, n_seeds=n_seeds
            )
        except RuntimeError as e:
            raise UserError("DeepTICA training failed: %s" % e)
        if seed_min_corr < SEED_ROBUST_WARN:
            session.logger.warning(
                "mdsmooth: DeepTICA is seed-unstable (minimum cross-seed "
                "correlation %.2f across %d seeds). Different random "
                "initializations gave different collective variables, so this "
                "learned signal may be unreliable - compare against the IC1 / "
                "PC1 signals before trusting it." % (seed_min_corr, n_seeds)
            )
    else:
        series = _rmsd_series(fit_frames, ref_index)

    result = filter_rmsd(
        series,
        sampling_rate=1.0,
        target_frames=target_frames,
        power_fraction=power_cutoff,
        cutoff_frequency=cutoff_frequency,
        order=order,
        include_ends=include_ends,
        extra_frames=extra_indices,
    )

    signal_label = {"pc1": "PC1", "ic1": "IC1", "dpca": "dPC1",
                    "deeptica": "DeepTICA"}.get(signal, "RMSD")
    # Descriptive model name so the morph reads as, e.g., "smoothed (IC1, 50 key
    # frames)" in the model panel instead of a bare "filtered".
    morph_name = "%s (%s, %d key frames)" % (
        name, signal_label, result.frames.size)

    # Random-walk / undersampling guardrail: if the signal looks like a diffusive
    # cosine, the extrema are probably noise artifacts rather than real motion.
    if result.cosine_content_high:
        session.logger.warning(
            "mdsmooth: %s has high cosine content (%.2f). The signal resembles a "
            "diffusive random walk, so its extrema may be undersampling artifacts "
            "rather than real transitions, the trajectory may not sample the "
            "motion often enough. Treat the movie's key frames with caution; a "
            "more localized selection (toAtoms) or a physical signal (RMSD to a "
            "reference) is often safer."
            % (signal_label, result.cosine_content)
        )

    # The key frames (all-atom coords of each significant frame) feed both the
    # morph and the optional PDB export, so compute them once.
    need_key_coords = (make_morph or save_pdbs) and result.frames.size >= 1
    key_coords = (
        _key_coords(full_frames, align_transforms, result.frames)
        if need_key_coords
        else None
    )

    saved_paths = None
    if save_pdbs and key_coords is not None:
        saved_paths = _save_frame_pdbs(
            session, structure, key_coords, result.frames, save_pdbs, name
        )

    made_morph = None
    if make_morph:
        if result.frames.size < 2:
            session.logger.warning(
                "Only %d significant frame(s) found; need at least 2 to build a "
                "morph. Try a larger targetFrames (or a higher cutoffFrequency)."
                % result.frames.size
            )
        else:
            made_morph = _build_morph(
                session, structure, key_coords, result.steps, morph_name, method
            )

    # IC1 lag-robustness caution: a slow mode whose shape flips across lag times
    # is likely spurious / undersampled rather than a real transition.
    if lag_robustness is not None:
        lags, _corrs, min_corr = lag_robustness
        if min_corr < TICA_LAG_ROBUST_WARN:
            session.logger.warning(
                "mdsmooth: IC1 is lag-sensitive (shape correlation as low as "
                "%.2f across lags %s). The slow mode changes with the lag time, "
                "which suggests it may be undersampled or spurious rather than a "
                "robust transition. Try a different lag, or compare against the "
                "PC1 / RMSD signal."
                % (min_corr, ", ".join(str(x) for x in lags))
            )

    if show_plot:
        if getattr(session.ui, "is_gui", False):
            from .graph import MDSmoothPlot
            if MDSmoothPlot is not None:
                rebuild = _make_rebuild(
                    session, structure, all_atoms, series, result.cutoff_frequency,
                    power_cutoff, order, include_ends, align_transforms,
                    morph_name, method, make_morph, made_morph,
                )
                MDSmoothPlot(session, result, structures=[structure],
                         tool_name="MDSmooth: %s" % structure,
                         rebuild=rebuild, signal_label=signal_label)
        else:
            session.logger.info(
                "showPlot skipped: no graphical session (running with --nogui)."
            )

    _report(session, structure, result, made_morph, method=method,
            fit_atom_count=len(fit_atoms), saved_paths=saved_paths,
            signal_label=signal_label, lag_robustness=lag_robustness)
    return result


def install_learned_cv(session, gpu=False, index_url=None):
    """Create the optional learned-CV venv (PyTorch + mlcolvar) for DeepTICA.

    A one-time, several-GB install that runs in a **dedicated virtual environment**
    (not ChimeraX's Python), following the pattern of ChimeraX's own ML tools.
    ``gpu`` picks a CUDA torch build; ``indexUrl`` overrides the pip index (e.g. a
    specific ``https://download.pytorch.org/whl/cuXXX`` or ``.../whl/cpu``).
    """
    from .learned import create_venv, default_venv_dir, venv_ready

    if venv_ready():
        session.logger.info(
            "Learned-CV environment already installed at %s." % default_venv_dir()
        )
        return
    # By default use no extra index: plain PyPI serves the correct torch (CPU/MPS
    # on macOS, CUDA on Linux) and also has mlcolvar.  Only add a PyTorch index to
    # force a specific CUDA build; it is passed as an *extra* index so mlcolvar is
    # still found on PyPI.
    url = index_url
    if url is None and gpu:
        url = "https://download.pytorch.org/whl/cu126"
    session.logger.warning(
        "Installing the learned-CV environment (PyTorch + mlcolvar). This "
        "downloads several GB and can take several minutes; ChimeraX may appear "
        "busy. See the log for progress."
    )
    try:
        create_venv(index_url=url, logger=session.logger)
    except Exception as e:
        raise UserError(
            "Learned-CV install failed: %s\nYou can retry, or set up the venv "
            "yourself (see the mdsmooth docs)." % e
        )
    if venv_ready():
        session.logger.info(
            "Learned-CV environment ready. The DeepTICA signal is now available: "
            "mdsmooth #N signal deeptica"
        )
    else:
        raise UserError(
            "Install finished but torch / mlcolvar still do not import in the "
            "venv. Check the log for pip errors."
        )


install_learned_cv_desc = CmdDesc(
    keyword=[
        ("gpu", BoolArg),
        ("index_url", StringArg),
    ],
    synopsis="Install the optional PyTorch/mlcolvar environment for DeepTICA",
)


mdsmooth_desc = CmdDesc(
    required=[("structure", AtomicStructureArg)],
    keyword=[
        ("to_atoms", AtomsArg),
        ("align_atoms", AtomsArg),
        ("reference", IntArg),
        ("signal", EnumOf(["rmsd", "pc1", "ic1", "dpca", "deeptica"])),
        ("seeds", IntArg),
        ("lag", IntArg),
        ("mode", EnumOf(["extrema", "kinetic"])),
        ("states", IntArg),
        ("microstates", IntArg),
        ("state_steps", IntArg),
        ("tica_dim", IntArg),
        ("target_frames", IntArg),
        ("power_cutoff", FloatArg),
        ("cutoff_frequency", FloatArg),
        ("order", IntArg),
        ("include_ends", BoolArg),
        ("make_morph", BoolArg),
        ("method", EnumOf(["corkscrew", "linear"])),
        ("save_pdbs", StringArg),
        ("show_plot", BoolArg),
        ("name", StringArg),
        ("extra_frames", ListOf(IntArg)),
    ],
    synopsis="Low-pass filter a trajectory's RMSD and morph its significant frames",
)

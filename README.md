# ChimeraX-MDSmooth

A [UCSF ChimeraX](https://www.rbvi.ucsf.edu/chimerax/) bundle that turns a noisy
molecular-dynamics trajectory into a clean morph movie. It reads a 1D signal from
the trajectory, keeps only the frames where the structure actually changes, and
builds a morph through those frames, so the movie shows real conformational motion
instead of thermal jitter. One command, `mdsmooth`, runs the whole pipeline, and a
**Tools > Trajectory Analysis > MDSmooth** panel does the same point-and-click.

![The MDSmooth plot: the noisy unfiltered signal, the smoothed filtered curve, and
the significant frames it keeps.](images/plot.png)

## How it works

`mdsmooth` runs this pipeline on a loaded trajectory:

1. **Compute a signal.** A single value per frame that summarizes the structure.
   By default this is the best-fit RMSD of every frame to a reference frame (frame
   1 by default) over a chosen atom set. The default atom set is the whole
   structure minus bulk solvent and free ions, so the signal tracks the
   macromolecule and any ligand rather than diffusing water. Four other signals
   track the direction of collective motion instead of distance. See
   [Signals](#signals-what-motion-drives-frame-selection).
2. **Filter.** A zero-phase Butterworth low-pass filter (`scipy.signal.filtfilt`,
   run forward and backward so peak timing is never shifted) removes the
   high-frequency jitter. How hard it smooths comes from one cutoff frequency,
   which you set indirectly with a frame target (default: about 50 frames) or
   directly. See [Choosing the cutoff](#choosing-the-cutoff-the-power-spectrum).
3. **Find significant frames.** The local maxima and minima of the *filtered*
   signal, plus the first and last frames. These are the turning points, the
   frames where the motion changes direction.
4. **Build the morph.** A new trajectory interpolates between the significant
   frames. Each segment gets a step count equal to the real frame spacing, so the
   morph keeps the simulation's true timing. Interpolation defaults to
   **corkscrew** (ChimeraX's own morph engine), which moves rotating domains along
   a screw path. `method linear` selects straight-line Cartesian interpolation.
5. **Plot.** An interactive graph of the raw and filtered signal opens, with the
   kept frames marked (maxima red, minima blue). Click it to scrub the trajectory,
   or add frames the filter would otherwise skip.

## Signals: what motion drives frame selection

The filter, frame-selection, and morph steps are identical for every signal. Only
what you track changes. RMSD measures distance from a reference frame. It needs no
tuning and always works, which is why it is the default. The other four track the
direction of the motion, which can reveal a change RMSD misses.

| Signal     | What it is | Good for | Cost / caveat |
|------------|-----------|----------|---------------|
| `rmsd`     | Distance to the reference frame | The safe default; needs no tuning | Direction-blind: a single magnitude, so distinct conformations can share a value, and it depends on the reference and atom selection |
| `pc1`      | Largest collective motion (PCA) | The dominant concerted motion | High variance, not function: large thermal breathing can overwhelm the real motion |
| `ic1`      | Slowest collective motion (tICA) | Functional, slow transitions | Needs the `lag` parameter set and good sampling |
| `dpca`     | Backbone dihedral PCA (phi/psi) | Flexible or looping backbones | Protein backbone only |
| `deeptica` | Learned nonlinear slow CV (DeepTICA) | Slow motions a linear method misses | Optional PyTorch setup; slow to train |

A little more on each:

- **`rmsd`** is the best-fit RMSD to the reference frame over the selected atoms,
  after rigid-body superposition. It tracks how far the structure has moved, not
  in which direction. It needs no parameters, which is why it is the default.
- **`pc1`** is the first principal component of the aligned trajectory (essential
  dynamics): the single highest-variance collective motion. It is a good choice
  when one big concerted motion dominates, but large-amplitude thermal breathing
  also has high variance, so PC1 can track wobble rather than function.
- **`ic1`** is the first time-lagged independent component (tICA): the slowest
  collective motion, meaning the coordinate that stays correlated longest across a
  lag of `lag` frames. Slowness is usually a much better proxy for a functional
  transition than variance. The `lag` keyword (default 10 frames) is the one
  parameter to tune. The implementation reduces onto the leading PCA subspace first, so the
  slow mode stays stable instead of changing shape at every lag.
- **`dpca`** is PCA on the backbone phi/psi dihedral angles. Because dihedrals are
  internal coordinates, no structural alignment is needed, so it does not couple
  internal motion to a fit. That helps for flexible or looping backbones. Each
  angle is mapped to (cos, sin) first so the periodic wrap-around is handled
  correctly. Protein backbone only.
- **`deeptica`** is a neural-network slow collective variable (DeepTICA) that can
  capture nonlinear slow motions a linear method cannot. It is optional and heavy.
  A one-time `mdsmooth installLearnedCV` sets up a separate PyTorch/mlcolvar
  environment, and training runs several random seeds and reports how consistent
  they are, because a learned CV is only trustworthy when independent seeds agree.

**Which one should you pick?** Start with `rmsd`. It needs no tuning and is the right
choice for a quick, honest smoothing of any trajectory. Move to a direction-aware
signal only when RMSD flattens out a change you know is there:

- Reach for `pc1` when one large, obvious motion dominates (a domain opening, a hinge
  bending) and you just want the biggest movement.
- Reach for `ic1` (tICA) when the interesting motion is slow but not the largest, for
  example a functional transition hidden under bigger thermal breathing. This is the
  most common upgrade from RMSD.
- Reach for `dpca` when the motion lives in the backbone itself (a flexible loop
  rearranging) rather than in the overall shape. Protein backbone only.
- Reach for `deeptica` only when you suspect a slow motion that is nonlinear, a linear
  method has failed to separate it, and you are willing to install the extra packages
  and check that the training seeds agree.

All of the direction-aware signals need enough sampling to be meaningful: if the
trajectory never actually crosses the barrier of interest, they describe noise, not
function. The cosine-content guardrail below is what tells you when that has happened.

### Cosine content: the random-walk guardrail

Every signal reports a **cosine content** from 0 to 1. An undersampled trajectory
that never actually crosses an energy barrier diffuses like a random walk, and a
random walk projects onto a near-perfect half-cosine, with cosine content
approaching 1. So a high value (the tool cautions above **0.85**) warns that the
"slow mode" may be undersampled diffusion that only looks like a transition, and the
extrema the filter found may be noise artifacts rather than real conformational
states.

It is deliberately a caution, not a verdict. A single genuine slow transition is
also roughly cosine-shaped, so a low threshold would raise a false alarm on
perfectly good signals. Treat a high value as a prompt to sample more or pick a more localized
selection, not as a failure. For the `ic1` signal the tool additionally checks
whether IC1 keeps its shape across nearby lag times. A real slow mode is stable, a
spurious one flips around, and the tool cautions if it is lag-sensitive.

```
mdsmooth #1 signal ic1 toAtoms :436-441 lag 10 targetFrames 50
```

## Choosing the cutoff: the power spectrum

The filter has exactly one degree of freedom, the low-pass **cutoff frequency**,
in cycles per frame. Everything below the cutoff is kept as real motion. The
high-frequency jitter above it is smoothed away. There are three equivalent ways
to set it, applied in this priority order:

1. **`targetFrames`** (the default, about 50). Ask for about this many significant
   frames and the command solves for the cutoff that produces them by bisection.
   This is the most intuitive control. It re-fits itself to whatever atoms and
   signal you give it, and it also acts as a speed control, since more frames means
   more morph segments to build.
2. **`cutoffFrequency`** sets the cutoff directly. Lower smooths harder and keeps
   fewer frames.
3. **`powerCutoff`** keeps the cutoff that retains this fraction of the signal's
   spectral power. It is an expert-level control and numerically sensitive (see below).

The **power spectrum** shows why a cutoff works. Real conformational motion is
slow, so it lives in the low-frequency, high-power part of the spectrum. Thermal
jitter is fast and spreads into a flatter high-frequency shelf. A good cutoff sits
in the valley between them. The panel can draw the spectrum with the current
cutoff marked (the **Show power spectrum** checkbox, or the second panel below),
and you can click the spectrum to place the cutoff by eye:

![The trace with its power spectrum below. The indigo line marks the cutoff.
Structured, high-power motion sits to its left, and the flat jitter shelf sits to
its right.](images/plot_spectrum.png)

In the spectrum panel (bottom), power is on a log scale against frequency. The
tall low-frequency content on the left is the conformational signal. The long flat
stretch to the right is jitter. The indigo cutoff line is where the filter splits
them, and everything to its right is removed.

As a rough guide on a trajectory of about 1600 frames, `cutoffFrequency 0.01`
gives about 20 frames (a short movie) and `0.024` gives about 50. The `powerCutoff`
route is touchy because an RMSD series is dominated by its mean offset, so a
"fraction of total power" rule can swing from no valid filter to keeping most of
the noise with a small change. A frame target is usually the better starting
point.

## Kinetic (MSM) mode

Instead of filtering a 1D signal, kinetic mode groups the trajectory into
kinetically distinct **metastable states** (the long-lived shapes the molecule
settles into) and builds a short tour that visits one representative frame per
state. It answers "show me the N different shapes the molecule adopts" rather than
"sample the main motion".

**Kinetic mode always runs on tICA; it ignores the `signal` keyword.** The metastable
states are defined by the slow tICA coordinates, so `signal rmsd`, `pc1`, `dpca`,
and `deeptica` have no effect here. The only shared control that carries over is the
tICA `lag`.

This is also why, in the graphical panel, the **Kinetic (MSM) mode** checkbox only
appears when you have selected the **IC1 (tICA)** signal. Choosing any other signal
hides it, since kinetic mode has no meaning outside tICA space. On the command line
you get the same effect by passing `mode kinetic` directly; the `signal` keyword is
ignored either way.

The pipeline is:

1. **tICA** reduces the trajectory to its slowest coordinates (`ticaDim` of them).
2. **k-means** clusters those into many fine-grained microstates (`microstates`).
3. A **Markov State Model** estimates the transition rates between microstates.
4. **PCCA+** coarse-grains the microstates into the final `states` metastable
   groups, which are then ordered along the slowest coordinate.

For each state the tour uses the frame that belongs to it most strongly (its highest
PCCA+ membership), interpolating `stateSteps` frames between consecutive states so
the transitions play smoothly.

```
mdsmooth #1 mode kinetic states 5 toAtoms :436-441
```

A kinetic tour is a sequence of conformations, not a timeline, so play it on its own
Coordinate Sets slider rather than syncing it to the raw trajectory. Kinetic mode
needs the optional `deeptime` package (`pip install deeptime`). If it is missing,
the command tells you rather than failing silently.

**When kinetic mode helps, and when it does not.** It pays off when the trajectory is
long enough that the molecule visits each shape several times and actually transitions
between them: only then can the Markov State Model estimate real rates and the states
mean something. Use it when you want to see the distinct shapes a molecule occupies
rather than smooth its main motion.

It is the wrong tool for a short or undersampled trajectory. If the molecule barely
leaves one basin, there are no transitions to count, and the MSM either fails outright
or returns states that are just arbitrary slices of noise. In practice the command
stops with a clear message when the data cannot support it, for example "only N
kinetically-connected microstates; not enough for an MSM", and suggests a smaller
`lag`, fewer `microstates`, or a longer trajectory. If you hit that, or if the same
high cosine content that warns the 1D signals is present, the trajectory is telling
you to sample more before trusting a metastable-state picture. A safe first attempt is
to leave `states`, `microstates`, and `lag` at their defaults (5 states, 100
microstates, lag 10) and lower `states` only if PCCA+ cannot populate them.

## Install / build

You need ChimeraX 1.5 or newer. From a ChimeraX command line:

```
devel build  /path/to/ChimeraX-MDSmooth
devel install /path/to/ChimeraX-MDSmooth
```

`numpy` and `scipy` ship with ChimeraX, so the core has no extra dependencies. Two
features are optional and pull in their own packages only when used. Kinetic mode
needs `deeptime`, and the `deeptica` signal needs a one-time learned-CV environment
(`mdsmooth installLearnedCV`). Restart ChimeraX, or run `devel install` again after
edits, to pick up changes.

## Quick start

```
open  trajectory.pdb          # or a DCD/XTC/NetCDF trajectory, etc.
mdsmooth #1                   # analyze all atoms, ~50 frames, build the morph
```

Common variations:

```
mdsmooth #1 toAtoms #1@CA               # pick frames from the C-alphas only
mdsmooth #1 toAtoms #1 & #1:LIG :< 5    # track a binding pocket (within 5 A of ligand LIG)
mdsmooth #1 signal ic1 lag 10           # pick frames from the slowest tICA motion
mdsmooth #1 signal dpca                 # pick frames from backbone dihedral PCA
mdsmooth #1 mode kinetic states 5       # tour 5 metastable states instead
mdsmooth #1 targetFrames 30             # aim for a shorter, cleaner movie
mdsmooth #1 cutoffFrequency 0.01        # set the cutoff directly (smooth harder)
mdsmooth #1 method linear               # straight-line morph (no morph engine)
mdsmooth #1 savePdbs ~/keyframes        # also export a PDB per significant frame
mdsmooth #1 makeMorph false             # just report the significant frames
mdsmooth #1 showPlot false              # skip the graph popup
mdsmooth #1 extraFrames 250,600         # force extra significant frames into the morph
```

## Command reference

`mdsmooth <models> [options]`

| Option | Default | Meaning |
|--------|---------|---------|
| `toAtoms` | whole structure minus bulk solvent/ions | Atoms whose signal picks the significant frames |
| `alignAtoms` | none (no de-tumbling) | Superimpose these atoms onto the reference to strip global tumbling from the morph |
| `reference` | 1 | 1-based reference frame for RMSD and alignment |
| `signal` | `rmsd` | Signal to track: `rmsd`, `pc1`, `ic1`, `dpca`, `deeptica` |
| `lag` | 10 | tICA lag in frames (used by `ic1`, `deeptica`, kinetic) |
| `seeds` | (built-in) | DeepTICA training seeds; more is more consistent but slower |
| `mode` | `extrema` | `extrema` (filter a 1D signal) or `kinetic` (metastable-state tour) |
| `states` | (auto) | Number of metastable states in kinetic mode |
| `microstates` | (auto) | k-means microstates before coarse-graining (kinetic) |
| `stateSteps` | (auto) | Interpolation steps between kinetic states |
| `ticaDim` | (auto) | tICA dimensions used for kinetic clustering |
| `targetFrames` | 50 | Aim for about this many significant frames (solves for the cutoff) |
| `cutoffFrequency` | none | Set the low-pass cutoff directly (cycles/frame) |
| `powerCutoff` | none | Keep the cutoff that retains this fraction of spectral power |
| `order` | 5 | Butterworth filter order |
| `includeEnds` | true | Keep the first and last frames as significant frames |
| `makeMorph` | true | Build the morph (false just reports the frames) |
| `method` | `corkscrew` | Interpolation: `corkscrew` (morph engine) or `linear` |
| `savePdbs` | none | Directory to also write one PDB per significant frame |
| `showPlot` | true | Pop up the interactive plot |
| `name` | auto | Name for the new morph model |
| `extraFrames` | none | Comma-separated 1-based frames to force into the morph |

`mdsmooth installLearnedCV [gpu true] [indexUrl <url>]` sets up the optional
DeepTICA (PyTorch/mlcolvar) environment, once.

Full argument docs are also in `src/docs/user/commands/mdsmooth.html`, or use
`help mdsmooth` inside ChimeraX.

## The panel

Open **Tools > Trajectory Analysis > MDSmooth** for a point-and-click version. Pick
a loaded trajectory, choose a region and a signal, then use the two buttons.
**Analyze** previews the significant frames at the current settings without
building anything. **Build movie** does the morph once you are happy with the frame
count. The form hides controls that do not apply. The tICA `lag` and kinetic
options only appear once you select a signal that uses them, so it stays short for
the common RMSD case.

A single **Choose cutoff by** dropdown switches between the three equivalent cutoff
controls (number of frames, cutoff frequency, or spectral power), so they can never
disagree. A **Show power spectrum** checkbox adds the spectrum panel to the preview
plot. Clicking the spectrum sets the cutoff at that frequency and re-previews.

The panel can also overlay the smoothed morph on the raw trajectory. The raw fades
to a grey ghost in the background while the smoothed copy sits solid and coloured
on top, both driven by one playback slider so they stay in lockstep. A **Raw
transparency** control sets how faded the ghost is.

### Adding your own significant frames

The filter marks only local maxima and minima, so a conformation sitting on a flat
stretch of the filtered signal is skipped. To force one in:

- **By command:** `mdsmooth #1 extraFrames 250,600`, a comma-separated list of
  1-based frame numbers. Frames already significant are left alone. The rest are
  labelled `User` in the table and drawn as green diamonds on the plot.
- **By clicking:** in the plot, click **Add significant frame**, then click the
  graph where you want it. A green preview marker lands on the *unfiltered* (grey)
  trace so you can see the exact frame you are picking. Click again to move it,
  then **Add frame N** to commit. The morph rebuilds in place with the new key
  frame.

![A user-added significant frame (green diamond) placed on a flat stretch of the
unfiltered trace where the filter found no extremum.](images/plot_userframe.png)

### Selecting a region: binding pockets, ligands, nonstandard residues

`toAtoms` chooses which atoms' signal picks the significant frames. Selection is by
atom, so ligands and nonstandard residues are treated like any other atoms:

- **Default** (`toAtoms` omitted): the whole structure except bulk solvent and free
  ions. Protein, nucleic acids, **ligands, and nonstandard residues are all kept**,
  so a bound ligand's motion drives frame selection. Only water and monatomic
  counter-ions are dropped, because a box of freely diffusing solvent moves far more
  than the macromolecule and would swamp its RMSD. Anything you pass to `toAtoms`
  yourself is honoured verbatim, with nothing dropped, so you never lose a ligand or a
  nonstandard residue by naming it explicitly.
- **A binding pocket:** a zone spec, for example `toAtoms #1 & #1:LIG :< 5` (atoms
  within 5 A of ligand `LIG`), or an explicit residue list `toAtoms #1:45,88,120`.
- **A ligand alone:** name it, for example `toAtoms #1:LIG`.

**Example: track RMSD of just the binding pocket, not the whole protein.**
Suppose the pocket is residues 45, 88, 120, and 133. RMSD over only those residues
picks up a local induced-fit rearrangement that whole-protein RMSD would average
away, so the movie zooms in on the motion you actually care about:

```
mdsmooth #1 toAtoms #1:45,88,120,133
```

`rmsd` is the default signal, so no `signal` keyword is needed. To let the pocket
follow the ligand instead of listing residues by number, use a distance zone, for
example every atom within 5 A of ligand `LIG`:

```
mdsmooth #1 toAtoms #1 & #1:LIG :< 5
```

Either way the pocket, ligand, and any nonstandard residues in the selection all
contribute to the RMSD on equal footing; only the atoms you leave out are ignored.

Whatever you pick only affects frame selection. The morph itself always carries the
entire structure (ligands and ions included), so the movie shows everything moving
together. Use `alignAtoms` if you want to additionally de-tumble the morph by
superimposing a chosen set onto the reference frame. By default the morph is not
realigned.

### Carbohydrates and other ring systems

Sugar rings have preferred shapes (for a pyranose, the chair) that matter for their
function, so it is worth knowing where the tool can and cannot affect them.

MDSmooth never changes the geometry of the frames it selects. The significant
frames, and the PDBs written by `savePdbs`, are exact snapshots from your
trajectory, so their ring shapes are whatever the simulation produced. The only
synthetic geometry is the in-between frames of the morph, and a ring can look
distorted there only when its pucker changes between two neighbouring key frames.
Corkscrew (the default) interpolates local geometry rather than dragging atoms
along straight lines, so it holds ring shape far better than `linear`, which can
flatten a ring as it flips. For carbohydrate-heavy systems, keep the default
corkscrew method and read the in-between frames as illustration. The real geometry
is in the key frames.

None of the signals tracks ring puckering directly (`dpca` uses protein backbone
dihedrals, not sugar rings), so if a pucker transition is itself the motion you
want to show, select the frames around it yourself with `extraFrames`.

### Loading Amber trajectories

ChimeraX reads Amber `.nc` **coordinates** but has no reader for Amber `.prmtop`
**topology** files, so you cannot `open` a prmtop directly. You first need a
structure (PDB or mol2) with the same atoms, then load the trajectory onto it:

```
open  topology.pdb                                 # defines the atoms
open  production.nc format amber structureModel #1 # loads all frames as coordsets
mdsmooth #1 toAtoms #1@CA
```

Two details trip people up. `format amber` is required because `.nc` is ambiguous
(it is also a density-map suffix), and the topology PDB's atom order must match the
trajectory. If you only have a prmtop and an nc, the helper `tools/amber_to_pdb.py`
writes a matching topology PDB from the prmtop and the trajectory's first frame:

```
python tools/amber_to_pdb.py system.prmtop production.nc topology.pdb
```

### Recording the movie

The command logs a table of significant frames (frame, filtered value, type,
interpolation steps) and adds the morph as a new model. Record it with the
[`movie`](https://www.rbvi.ucsf.edu/chimerax/docs/user/commands/movie.html)
command.

## Project layout

```
ChimeraX-MDSmooth/
├── bundle_info.xml     # bundle metadata; drives `devel build`
├── src/
│   ├── __init__.py     # BundleAPI: registers the mdsmooth command and panel
│   ├── cmd.py          # ChimeraX layer: reads coordsets, builds the morph
│   ├── filter.py       # pure numpy/scipy core (no chimerax imports)
│   ├── kinetic.py      # tICA / MSM / PCCA+ metastable-state clustering
│   ├── learned.py      # optional DeepTICA learned-CV support
│   ├── graph.py        # raw-vs-filtered plot + power spectrum (popup + headless)
│   ├── tool.py         # the Tools > Trajectory Analysis panel
│   └── docs/user/commands/mdsmooth.html
└── tests/
    └── test_*.py       # unit tests for the core, run without ChimeraX
```

The scientific core in `src/filter.py` and `src/kinetic.py` deliberately has **no
ChimeraX dependency**, so it can be unit-tested and reused anywhere:

```
pip install numpy scipy pytest
pytest tests
```

## Status and known limitations

- The core filtering, extrema, tICA, and clustering logic is covered by the unit
  tests in `tests/`, which run without ChimeraX.
- Tested end-to-end on **ChimeraX 1.5** (macOS): it builds and installs with `devel
  install`, runs `mdsmooth` on a trajectory, and produces a morph whose frame count
  matches the original. The coordset-building calls (`structure.copy`,
  `add_coordsets(..., replace=True)`), `chimerax.geometry.align_points`, and the Qt
  plot popup all work.
- Morph interpolation defaults to **corkscrew**. Each segment is interpolated
  through ChimeraX's own `morph` engine, which turns the rigid-body change between
  key frames into a screw motion (plus internal-coordinate change), so large domain
  rotations look natural instead of atoms sliding through one another. The
  per-segment step counts still come from the frame spacing, so the movie keeps the
  simulation's real timing. `method linear` restores straight-line Cartesian
  interpolation, which is faster and dependency-free but can distort bond lengths
  and ring shapes across large jumps. Corkscrew also falls back to it automatically
  if the morph engine is unavailable.
- Because corkscrew runs `morph` once per segment, build time grows with structure
  size and with the number of key frames, so a higher frame target means a slower
  build. Linear is nearly instant either way. The stitching logic lives in
  `_corkscrew_morph_coords` if a future ChimeraX changes the `morph` command's
  coordset conventions.
- The `deeptica` signal and kinetic mode are optional and pull in heavy
  dependencies (PyTorch and mlcolvar, or deeptime). If those packages are absent,
  the command reports what to install rather than failing silently. The rest of the
  tool works without them.

## License

MIT, see [LICENSE](LICENSE).

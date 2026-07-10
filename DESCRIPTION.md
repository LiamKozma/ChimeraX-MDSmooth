# MDSmooth

MDSmooth is a UCSF ChimeraX bundle that turns a noisy molecular dynamics
trajectory into a smooth, clean morph movie. It keeps only the frames where the
structure actually changes, so the movie shows real conformational motion
instead of thermal jitter.

## How it works

The bundle runs this pipeline on a loaded trajectory:

1. Read a 1D signal from the trajectory. By default this is the RMSD of every
   frame against a reference frame (the first frame by default), measured over a
   chosen set of atoms.
2. Apply a zero-phase Butterworth low-pass filter to the signal. This removes the
   high-frequency jitter without shifting the timing of the peaks, so the
   smoothed curve still lines up with the original frames.
3. Find the significant frames, which are the local maxima and minima of the
   filtered curve, plus the first and last frame. These are the turning points of
   the real motion.
4. Build a morph trajectory that interpolates between those significant frames.
   The number of interpolation steps in each segment matches the original frame
   spacing, so the movie keeps the real timing of the simulation.
5. Show a plot of the raw and filtered signal with the significant frames marked,
   so you can see what was kept.

The result is a new trajectory model that plays as a smooth movie of the
meaningful motion. You can record it with the ChimeraX movie command.

The easiest way to control the movie is `targetFrames`: ask for about that many
significant frames and the command solves for the cutoff that produces them. You
can instead set the cutoff frequency directly with `cutoffFrequency`, in cycles
per frame, where lower values smooth harder and give fewer, cleaner frames.

## Choosing which signal to track

RMSD measures distance from a reference frame. It is robust and needs no tuning,
which is why it is the default. Four other signals are available with the
`signal` option, and each tracks the direction of a motion rather than mere
distance:

- `pc1`: the largest collective motion (principal component analysis). Good for
  the dominant concerted motion, though it can be swamped by global tumbling.
- `ic1`: the slowest collective motion (tICA). Slowness is often a better proxy
  for a functional transition than size, so IC1 can recover a change that PC1
  buries under fast wobble. It uses a lag time (the `lag` option) and wants a
  trajectory that samples the motion.
- `dpca`: dihedral PCA of the backbone phi/psi angles. Because dihedrals are
  internal coordinates, it needs no alignment and ignores global tumbling, which
  suits flexible or looping backbones. Protein backbone only.
- `deeptica`: a learned nonlinear slow coordinate trained with PyTorch. It can
  capture slow motions a linear method misses, but it is an optional, heavy
  feature that you install once and that takes minutes to train.

Every signal feeds the same filter, frame-selection, and morph steps, so only the
choice of what to track changes. Each one reports a cosine content from 0 to 1: a
high value warns that the signal resembles a random walk, so on one short
trajectory the slow mode may be undersampled diffusion rather than a real
transition.

## Kinetic mode: a tour of the distinct shapes

Kinetic mode is a different strategy, chosen with `mode kinetic`. Instead of
filtering a 1D signal, it groups the trajectory into kinetically distinct
metastable states and picks one representative frame per state, then morphs a tour
of them. It answers "show me the N different shapes" rather than "sample the main
motion". The steps are tICA, then k-means microstates, then a Markov State Model,
then PCCA+ coarse-graining. A kinetic tour is a conformation sequence, not a
timeline, so play it on its own slider rather than syncing it to the raw
trajectory. Kinetic mode needs the optional deeptime package.

## Choosing which atoms to track

By default the signal is measured over the whole structure, but that is often not
what you want. If you care about a binding pocket, you should measure just that
pocket, not the whole protein. A large protein moving somewhere else would
dominate the signal and hide the pocket's own motion.

Use the `toAtoms` option to point the analysis at any atom selection. For
example:

- Backbone alpha carbons only: `toAtoms #1@CA`
- A list of pocket residues: `toAtoms #1:45,88,120,145`
- Everything within 5 Angstroms of a ligand named LIG: `toAtoms #1 & #1:LIG :< 5`

`toAtoms` only changes which atoms decide the significant frames. The morph itself
always carries the entire structure, so the movie still shows the whole system
moving even when the frame selection was driven by a small region.

## How ligands and nonstandard residues are treated

Selection is by atom, so ligands and nonstandard residues are treated the same as
everything else. In the default atom set they are kept. Only bulk solvent (water)
and free ions such as sodium and chloride are dropped, because those diffuse
freely and would swamp the signal of the actual molecule. Protein, nucleic acids,
bound ligands, and nonstandard residues all contribute to the default signal.

So a bound ligand's motion helps pick the significant frames by default. If you
want to study a ligand on its own, name it with `toAtoms`, for example
`toAtoms #1:LIG`. To follow a pocket together with its ligand, use a zone
selection such as `toAtoms #1 & #1:LIG :< 5`. Whatever you select, the ligand and
the ions are still carried through the morph.

## Carbohydrates and other ring systems

Sugar rings have preferred shapes, such as the chair of a pyranose, that matter
for their function. MDSmooth never changes the geometry of the frames it selects:
the significant frames, and the PDBs written by `savePdbs`, are exact snapshots
from your trajectory, so their ring shapes are whatever the simulation produced.
The only synthetic geometry is the in-between frames of the morph, and a ring can
look distorted there only when its pucker changes between two neighbouring key
frames. Corkscrew, the default interpolation, keeps ring shape far better than
linear, which can flatten a ring as it flips. For carbohydrate-heavy systems, keep
the default corkscrew method and treat the in-between frames as illustration; the
real geometry is in the key frames. If a pucker transition is itself the motion
you want to show, pick the frames around it by hand with `extraFrames`, since none
of the signals tracks ring puckering directly.

## Interpolation: corkscrew or linear

There are two ways the bundle can interpolate between significant frames, chosen
with the `method` option.

Corkscrew is the default. For each pair of neighboring significant frames, the
bundle hands the two structures to ChimeraX's own `morph` command, which uses
corkscrew interpolation. Corkscrew turns the rigid-body part of the change into a
screw motion, so a rotating domain turns in an arc instead of having its atoms
slide in straight lines through one another. This is the same interpolation
ChimeraX's `morph` command uses by default.

Linear is the bundle's own built-in interpolation. It moves each atom in a
straight line between key frames. It is fast and does not depend on the morph
engine, but a straight line in space can stretch bonds across a large motion, so
big domain rotations and ring flips can look unnatural.

Either way the per-segment step counts are the same, so only the path between key
frames changes, not the timing. If the morph engine is not available, corkscrew
falls back to linear on its own.

Because corkscrew runs the morph engine once per segment, it takes longer to build
on larger structures, and also when there are more significant frames, because
each one adds another segment to morph. Linear is close to instant either way.

## Exporting the significant frames

If you want the raw key frames instead of, or in addition to, the built morph, use
`savePdbs` to write one PDB per significant frame into a folder. The files are
named with the original frame numbers. These are the exact frames the morph
interpolates between, so you can inspect them, hand them to another program, or
morph them yourself with ChimeraX's `morph` command if you want full control over
the interpolation.

## Adding your own significant frames

The filter only marks local maxima and minima, so a conformation that sits on a
flat stretch of the filtered curve is skipped. You can force one in two ways:

- With the `extraFrames` option, giving a comma-separated list of 1-based frame
  numbers, for example `extraFrames 250,600`.
- By clicking. In the plot window, click Add significant frame, then click the
  graph where you want it. A green marker lands on the unfiltered trace so you can
  see the exact frame you are picking. Click again to move it, then click Add
  frame to keep it. The morph rebuilds in place with the new frame included.

A frame that is already significant is left alone, so you never get a duplicate.
Frames you add are labeled User in the results table and drawn as green diamonds
on the plot.

## Using it

You can run MDSmooth as a command or from a panel.

As a command:

```
open trajectory.pdb
mdsmooth #1
mdsmooth #1 toAtoms #1 & #1:LIG :< 5 method corkscrew
mdsmooth #1 signal ic1 lag 10 targetFrames 50
mdsmooth #1 mode kinetic states 5
mdsmooth #1 cutoffFrequency 0.01 extraFrames 250,600
mdsmooth #1 method linear savePdbs ~/keyframes
```

From a panel: open Tools, then Trajectory Analysis, then MDSmooth. Pick a loaded
trajectory, choose a region to track and a signal, then use the two buttons.
Analyze previews the significant frames at the current settings without building
anything, and Build movie does the morph once you are happy with the frame count.
The panel hides controls that do not apply, so the tICA lag and kinetic options
appear only when you select the IC1 signal.

## Comparing the smoothed movie against the original

When you run the tool from the panel, it can overlay the smoothed morph directly
on the raw trajectory. The raw trajectory fades to a grey ghost in the
background, and the smoothed copy sits solid and colored on top. Both are driven
by one playback slider, so they play, pause, and step in lockstep. The Raw
transparency control sets how faded the ghost is, and you can turn the overlay off
if you only want the smoothed movie on its own.

## Requirements

Needs ChimeraX 1.5 or newer. numpy and scipy already ship with ChimeraX, so the
core has no extra dependencies to install. Two optional features pull in their own
packages only if you use them: kinetic mode needs deeptime, and the deeptica
signal needs a one-time PyTorch environment that you install with
`mdsmooth installLearnedCV`.

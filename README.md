# Create 3D models of Conway's Game of Life

Bring [computational irreducibility](https://mathworld.wolfram.com/ComputationalIrreducibility.html) to life.

# Usage

Install requirements:

`pip install -r requirements.txt`

Example command:

`python conway_to_stl.py die-hard.txt 150 --slice_count 5 --base_init`

Use [this converter](https://lasttheory.com/tools/rle-to-text-converter) to convert initializations in [RLE](https://conwaylife.com/wiki/Run_Length_Encoded) to text.

## Whole-strand tabletop sculptures

The default `seeds/sculptures/strands.json` contains two Kok's galaxy cycles,
one Gosper gun core cycle, and the interacting-acorn transient through t=28.
These open configurations use standard constant-width parallelepipeds, steeper
time axes and smaller integrated bases. Seed attribution and GFDL data terms
are in `seeds/sculptures/intricate/PROVENANCE.md`; software remains MIT.

```sh
uv venv .venv
uv pip install --python .venv/bin/python -r requirements-sculptures.txt
.venv/bin/python sculptures.py exports --font /path/to/CascadiaMono.ttf
.venv/bin/python sculptures.py exports --validate-only
.venv/bin/python -m pytest -q tests
```

Use a fresh output directory. A local `--config profile.json` can supply display
materials and recipes; generic committed palettes are not printer routing maps.
Cascadia Mono is available from Microsoft's `cascadia-code` releases.

Color is inherited through the **actual selected source** of each construction
link. Dominant roots surviving to the cut anchor up to four families. Minor root
trees join the nearest concurrent anchor trail. A root keeps its family throughout
its evolution: no head/stem phases, height bands, fixed XYZ bins, corridor masks,
rainbow cycles or per-face paint. This geometric construction forest is not unique
biological Life ancestry. Every family unions whole emitted links using the same
constructor and spatial/time transforms as the complete mesh.

Material zero owns the actual plinth; dominant anchor rank resolves overlapping
link volumes. Bounded microscopic interface repairs transfer only existing
multiple-family overlaps to an incident family, never changing the complete
geometry or adding decorative joints. Root/link records and repair measurements
are saved in settings. Section checks measure area agreement without forcing
colors into every height. Reload proof recomputes the forest and reports the
longest connected vertical span of each material.

Standard links use the original eight-vertex/twelve-triangle constructor.
Uniform XY width overlap is 0.0001 cells, necessary to avoid exact edge contacts
on STL reload. Width stays `cell_mm * 1.0001` throughout each link; each side moves
outward `cell_mm * 0.00005`, with no Z expansion, taper, waist or flared cap.
The shared double-precision Boolean path and ASCII STLs retain fine coordinates.
Earlier recipe files retain their seed/geometric profiles but now use whole-strand
coloring; old Git revisions and delivery bundles retain the earlier color methods.
No legacy slab/corridor coloring mode is offered.

Outputs include complete mm STL, closed registered per-material STLs, multipart
material 3MF, Y-up meter GLB, two actual-mesh JPEG views, overview, seed/settings,
and independent reload proofs. Import color STLs together without separately
centering, scaling, arranging or dropping parts. STL has no embedded color.
`--complete-from originals` can preserve an authoritative complete STL; recipe
hashes are enforced for authority copies, while fresh exports record their own
hash. Existing design output directories are never overwritten.

Coverage checks retain containment, overlap and volume-conservation residuals.
Strict lib3mf verifies closed/oriented meshes, material RGBA and assembly/buffers;
GLB buffers, colors, orientation and units are independently reloaded. Time zero
is the source plane; the last generation is a finite cut, not extinction. Padded
periodic simulation must stay away from its seam. Legacy CSG/Lego scratch files
never overwrite cwd `m.stl`.

Mesh validity is **not** print suitability: microscopic overlap contacts, diagonal
links, bridges, disconnected minor strands and small material interfaces need
process-specific review. Nominal link width is not certified minimum thickness.
No strength/support-free claim, slicing or physical action is performed.

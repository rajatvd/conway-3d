# Create 3D models of Conway's Game of Life

Bring [computational irreducibility](https://mathworld.wolfram.com/ComputationalIrreducibility.html) to life.

# Usage

Install requirements:

`pip install -r requirements.txt`

Example command:

`python conway_to_stl.py die-hard.txt 150 --slice_count 5 --base_init`

Use [this converter](https://lasttheory.com/tools/rle-to-text-converter) to convert initializations in [RLE](https://conwaylife.com/wiki/Run_Length_Encoded) to text.

## Tabletop sculptures and time-band colors

Three small deterministic recipes live in `seeds/sculptures/`: a glider, a
period-three pulsar, and a finite R-pentomino transient. Generate into a **new**
output directory (existing design directories are never overwritten):

```sh
uv venv .venv
uv pip install --python .venv/bin/python -r requirements-sculptures.txt
.venv/bin/python sculptures.py exports --font /path/to/CascadiaMono.ttf
.venv/bin/python sculptures.py exports --validate-only
```

Cascadia Mono is available from Microsoft's `cascadia-code` releases. Output:
complete uncolored STL, three registered color STLs, a core multipart colored
3MF, a color GLB, two bounded JPEG mesh views per design, a three-up overview,
settings, seeds, and independent reload checks. STL/3MF coordinates are mm;
GLB is converted to Y-up meters. Green includes the integrated plinth, followed
by amber and coral time bands. Preserve the shared origin when importing parts;
do not independently center, scale, or drop each part to the bed.

The generator reuses `conway_step` and the existing thin trimesh transitions.
Both exporters share the original birth-source traversal. Batch manifold union
avoids repeated CSG meshing; the legacy CSG/Lego exporter remains available and
now uses a private temporary STL instead of overwriting `m.stl`.

Recipes stop at the stated generation, not at extinction. The periodic Life
board is padded and the generator checks that no live cells approach its seam.
Time bands are intersections of the complete exported union, not separately
simulated objects. Half-generation color cuts avoid cap-plane topology
singularities. Checks compare their union and overlap against the complete STL,
reload 3MF mesh/material buffers, and verify GLB colors, scale, orientation, and
geometry. Upper pulsar bands contain four bodies; the complete sculpture is
connected through its base. Mesh validity is **not** a printability guarantee:
flared caps, sloping birth links, bridges, and narrow waists need process-specific
overhang/support and strength review. No slicing or printing is performed.

Focused checks: `uv pip install --python .venv/bin/python pytest`, then
`.venv/bin/python -m pytest -q`.

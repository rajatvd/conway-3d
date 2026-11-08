# Create 3D models of Conway's Game of Life

Bring [computational irreducibility](https://mathworld.wolfram.com/ComputationalIrreducibility.html) to life.

# Usage

Install requirements:

`pip install -r requirements.txt`

Example command:

`python conway_to_stl.py die-hard.txt 150 --slice_count 5 --base_init`

Use [this converter](https://lasttheory.com/tools/rle-to-text-converter) to convert initializations in [RLE](https://conwaylife.com/wiki/Run_Length_Encoded) to text.

## Structurally colored tabletop sculptures

Three deterministic recipes in `seeds/sculptures/` generate a glider, a
period-three pulsar, and a finite R-pentomino transient. Use a dedicated
environment and a **new** output directory:

```sh
uv venv .venv
uv pip install --python .venv/bin/python -r requirements-sculptures.txt
.venv/bin/python sculptures.py exports --font /path/to/CascadiaMono.ttf
.venv/bin/python sculptures.py exports --validate-only
```

For three broader, longer-history structures, select the separate recipe set:

```sh
.venv/bin/python sculptures.py intricate-exports --config seeds/sculptures/intricate.json --font /path/to/CascadiaMono.ttf
```

It contains a four-cycle Kok's galaxy vault, a two-cycle Gosper gun network,
and an authored interacting pair of acorns. Longest dimensions are about
114–166 mm, using 3.8–6 mm cells. Canonical seed attribution and the scoped
GFDL pattern-data license live in `seeds/sculptures/intricate/PROVENANCE.md`;
software remains MIT licensed. The independent seed tests verify the periods,
emitted glider, real collision, and unbounded sparse Life agreement.

To recolor previously generated shapes **without re-exporting the complete STL**,
add `--complete-from originals`, where `originals/<design-id>/complete.stl` contains
each authoritative shape. Recipe SHA-256 checks enforce byte-for-byte identity.
New generation uses one double-precision union path and ASCII complete STLs,
with a newly recorded output hash; historical recipe hashes are enforced for
explicit authority copies, not assumed invariant under reserialization.
Existing output design directories are never overwritten. Cascadia Mono is
available from Microsoft's `cascadia-code` releases.

Colors follow actual transition structure, not time bands: Evergreen plinth,
Ivory framework and joint rims, Coral birth-link stems, Violet birth-link heads.
The last two are volumetric corridors following the original deterministic
source-neighbor construction, not claims of unique biological ancestry. Local
head/neck motifs repeat on birth links while surviving links remain ivory, so
multiple materials occupy the same horizontal levels. Up to four named materials
are supported by the single 3MF/GLB export and independent validation pipeline;
assembly IDs are allocated after all material mesh IDs.

Outputs include the complete uncolored STL, registered closed per-material STLs,
a standard core multipart/material 3MF, color GLB, two JPEG mesh views per design,
a three-up overview, seeds, settings, and reload proofs. STL/3MF coordinates are
millimeters; GLB uses Y-up meters. Import color STLs together as one multipart
object; never independently center, scale, arrange, or drop them to the bed.

The complete mesh is the authority. Double-precision Boolean masks only assign
its volume to materials. Tiny kissing color-interface neighborhoods are assigned
to an unbroken joint material without changing the sculpture. ASCII STLs retain
fine interface precision. Coverage checks measure each part outside the authority,
pairwise overlap, summed volume conservation, and material areas in identical
horizontal cross-sections; this avoids a numerically fragile N-way re-weld of
coincident interfaces. Strict lib3mf reload verifies material assignment and
closed/oriented meshes; GLB buffers, colors, units, and orientation are reloaded.

Life and the existing thin trimesh transition geometry are unchanged. Time 0 is
the source plane and the last generation is a finite cut, not extinction. The
periodic simulation is padded and checked to stay away from its seam. The legacy
CSG/Lego geometry exporter still uses a private temporary STL, never cwd `m.stl`.
Mesh validity is **not** physical print validation: narrow waists, diagonal birth
struts, flared caps, bridges, and small color-interface features require
process-specific review. These colors can require many filament swaps; minimizing
swaps is not the design objective. No slicing or printing is performed.

Tests: install `pytest` in that environment, then
`.venv/bin/python -m pytest -q tests`.

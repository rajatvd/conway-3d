# Intricate Life seeds and attribution

All seeds use B3/S23, with rows mapped to X and columns to Y by the existing
exporter. Coordinate data were transcribed from the inline RLE at the cited
LifeWiki pages. No article prose or images are included. Direct downloadable
pattern endpoints returned HTTP 403 during retrieval; the article's inline
RLE and independent simulation supplied the data/behavior checks.

## Kok's galaxy → Galaxy Vault

- Historical discovery credit: Jan Kok (1971).
- Source and attribution: LifeWiki contributors, “Kok's galaxy”,
  https://mail.conwaylife.com/wiki/Kok%27s_galaxy
- Inline RLE, initial 28-cell phase:
  `2bo2bobob$2obob3ob$bo6bo$2o5bob2$bo5b2o$o6bob$b3obob2o$bobo2bo!`
- `koks-galaxy.txt` is its exact 9×9 plaintext decoding. Its full orbit reaches
  13×13. Independent verification finds first exact return at generation 8,
  populations 28,32,48,40,64,36,36,44,28. The sculpture contains four full cycles,
  t=0–32, not a tiled copy of another oscillator.

## Gosper glider gun → Gosper Loom

- Historical discovery credit: Bill Gosper and collaborators (1970).
- Source and attribution: LifeWiki contributors, “Gosper glider gun”,
  https://mail.conwaylife.com/wiki/Gosper_glider_gun
- Inline RLE, 36 cells in 36×9:
  `24bo11b$22bobo11b$12b2o6b2o12b2o$11bo3bo4b2o12b2o$2o8bo5bo3b2o14b$2o8bo3bob2o4bobo11b$10bo5bo7bo11b$11bo3bo20b$12b2o!`
- `gosper-glider-gun.txt` decodes this phase. Independent verification of the
  9×36 machine region finds repetition every 30 generations throughout t=0–90;
  whole-world populations at 0/30/60/90 are 36/41/46/51. An isolated emitted
  five-cell object at t=60 reproduces itself shifted diagonally by one cell after
  four steps. The sculpture is only t=0–64: two complete core cycles plus four
  steps and a finite emitted-stream cut, not an infinite object.

## Authored two-acorn interaction → Crossfire Arbor

- Canonical ingredient: Acorn, credited to Charles Corderman (no later than 1971).
- Source and attribution: LifeWiki contributors, “Acorn”,
  https://conwaylife.com/wiki/Acorn
- Ingredient RLE: `bo$3bo$2o2b3o!` (seven cells, 7×3).
- Authored arrangement: original Acorn at (row=0,column=0); the same seed rotated
  180 degrees at (row=0,column=12). `acorn-crossfire.txt` is their combined 3×19
  14-cell starting board, not a claimed canonical named oscillator.
- The combined evolution first differs from the OR of the two independently
  evolved seeds at t=11. Thus this is a real interaction, not duplicated streams
  placed cosmetically. Its population peaks at 110 at t=27 within the selected
  t=0–32 cut and ends at 78. No stabilization or full-Acorn-lifespan claim.

## Licensing

Pattern-coordinate data and their encodings in this directory are redistributed
under the GNU Free Documentation License, version 1.2 or any later version,
with no Invariant Sections, Front-Cover Texts, or Back-Cover Texts; attribution is
to the linked LifeWiki articles and their contributors. A full copy is included
as `LICENSE-GFDL-1.2.txt`. Upstream policy:
https://mail.conwaylife.com/wiki/LifeWiki%3ACopyrights

The authored combination is distributed on the same terms as its ingredient
coordinate data. This notice does not relicense the repository's software:
generator, renderer, tests, and original code retain the root MIT license.
Names and discovery credits identify historical patterns, not new discoveries.

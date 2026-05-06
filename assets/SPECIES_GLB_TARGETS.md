# Species GLB Assets

This branch includes the runtime species GLBs expected by the frontend catalogs:

- `assets/brain_mouse.glb`
- `assets/brain_rat.glb`
- `assets/brain_marmoset.glb`
- `assets/brain_macaque.glb`
- `assets/brain_zebrafish.glb`

The GLB mesh node names have been validated against
`frontend_data/species_catalogs/*.json` `regionsByMeshName` keys.

Validated mesh-node counts:

- Mouse: 840
- Rat: 224
- Marmoset: 309
- Macaque prototype: 365
- Zebrafish: 259

`scripts/generate_species_glbs.py` uses species-specific default per-mesh face
targets for runtime-sized exports: mouse 250, zebrafish 500, and 1500 for rat,
marmoset, and macaque. The generator still names every GLB node from the
frontend catalog mesh keys, so changing the face target must not change catalog
joins.

Rows present in the catalogs but absent from `regionsByMeshName` are metadata
rows with no generated mesh node, such as background labels or atlas labels that
are not present in the available source volume.

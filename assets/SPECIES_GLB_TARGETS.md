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

Rows present in the catalogs but absent from `regionsByMeshName` are metadata
rows with no generated mesh node, such as background labels or atlas labels that
are not present in the available source volume.

# Atlas Catalog Asset Manifest

This branch commits the frontend-ready catalog data, provenance catalogs, helper
scripts, and generated runtime GLB files. It does not commit raw atlas source
archives, NIfTI files, OBJ directories, or downloaded STL source directories.

Frontend code should consume:

- `frontend_data/species_catalogs.js`
- `frontend_data/species_catalogs/*.json`

For the current site ZIP layout, the same JSON/JS files are mirrored under:

- `data/species_catalogs.js`
- `data/species_catalogs/*.json`

Runtime GLBs are committed under:

- `assets/brain_mouse.glb`
- `assets/brain_rat.glb`
- `assets/brain_marmoset.glb`
- `assets/brain_macaque.glb`
- `assets/brain_zebrafish.glb`

The current mouse and rat GLBs are generated from BrainGlobe Atlas API mesh
packages (`allen_mouse_25um_v1.2` and `whs_sd_rat_39um_v1.2`) with node names
rewritten to the existing frontend catalog keys. Zebrafish remains generated
from mapZebrain STL source because the BrainGlobe zebrafish packages available
in the live registry do not match these mapZebrain mesh keys.

## Frontend Data Contract

Each species JSON includes species metadata, expected GLB asset path, license
caveat, mesh naming convention, `groupNormalization`, stable group
colors/counts, `regions`, and `regionsByMeshName`.

Frontend coders should not run Python, parse CSVs, or inspect atlas archives.
Use `regionsByMeshName[mesh.name]` after loading the matching GLB.

## Validated GLB Coverage

| Species | Catalog rows | GLB mesh nodes | Asset |
| --- | ---: | ---: | --- |
| mouse | 840 | 840 | `assets/brain_mouse.glb` |
| rat | 225 | 224 | `assets/brain_rat.glb` |
| marmoset | 332 | 309 | `assets/brain_marmoset.glb` |
| macaque_prototype | 368 | 365 | `assets/brain_macaque.glb` |
| zebrafish | 265 | 259 | `assets/brain_zebrafish.glb` |

Rows without mesh nodes are retained as metadata rows with empty `meshName`.
Examples include background labels, mapZebrain rows without STL paths, and atlas
labels absent from the available source volume.

## Catalog CSVs

The CSV catalogs are committed as provenance/intermediate data for maintainers
and GLB builders. They are not required by the frontend.

| Species | Catalog CSV | Rows |
| --- | --- | --- |
| mouse | `external/atlases/mouse/catalog_mouse_labels.csv` | 840 |
| rat | `external/atlases/rat/catalog_rat_labels.csv` | 225 label rows; 224 non-background mesh nodes |
| zebrafish | `external/atlases/zebrafish/catalog_zebrafish_regions.csv` | 265 |
| marmoset | `external/atlases/marmoset/catalog_marmoset_labels.csv` | 332 rows; 309 mesh nodes |
| macaque_prototype | `external/atlases/macaque_prototype/catalog_macaque_prototype_labels.csv` | 368 rows; 365 mesh nodes |

CSV schema:

```text
species,source_asset,source_mesh_file,glb_mesh_name,structure_id,acronym,label,group,parent_id,hemisphere,catalog_status,notes
```

Blank CSV `group` values are normalized to frontend group `Ungrouped`; the
original value is preserved in JSON as `sourceGroup`. Duplicate biological
labels are preserved when `structureId` or `meshName` differs.

## License Notes

- Rat WHS/NITRC is documented in this handoff as CC BY 4.0.
- NIH Marmoset Atlas v1.1 is documented in this handoff as CC BY-NC-SA 4.0.
- D99/AFNI states atlas datasets may not be modified or redistributed without
  prior consent. Treat macaque prototype data as internal unless permission is
  obtained.
- Verify Allen CCFv3 and mapZebrain reuse/citation terms before redistribution.

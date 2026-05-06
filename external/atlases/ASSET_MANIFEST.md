# Atlas Catalog Asset Manifest

This branch commits the frontend-ready catalog data and provenance catalogs.
It does not commit the raw atlas source downloads or generated species GLBs.

Frontend code should consume:

- `frontend_data/species_catalogs.js`
- `frontend_data/species_catalogs/*.json`

For the current site ZIP layout, the same JSON/JS files are also mirrored under:

- `data/species_catalogs.js`
- `data/species_catalogs/*.json`

## Frontend Data Contract

Each species JSON includes:

- species metadata and source notes
- expected GLB asset path
- license/reuse caveat
- mesh naming convention
- `groupNormalization`
- stable group colors and counts
- `regions`
- `regionsByMeshName`, keyed by expected GLB object name

Frontend coders should not run Python, parse CSVs, or inspect atlas archives.
Use `regionsByMeshName[mesh.name]` after loading the matching GLB.

## Catalog CSVs

The CSV catalogs are committed as provenance/intermediate data for maintainers
and GLB builders. They are not required by the frontend.

| Species | Catalog CSV | Rows |
| --- | --- | --- |
| mouse | `external/atlases/mouse/catalog_mouse_labels.csv` | 840 |
| rat | `external/atlases/rat/catalog_rat_labels.csv` | 225 label rows; 224 non-background expected meshes |
| zebrafish | `external/atlases/zebrafish/catalog_zebrafish_regions.csv` | 265 |
| marmoset | `external/atlases/marmoset/catalog_marmoset_labels.csv` | 332 volume-backed rows |
| macaque_prototype | `external/atlases/macaque_prototype/catalog_macaque_prototype_labels.csv` | 365 right-surface rows; 3 volume-backed rows |

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

## Expected Runtime GLB Outputs

The frontend catalogs point at these future runtime assets:

- `assets/brain_mouse.glb`
- `assets/brain_rat.glb`
- `assets/brain_marmoset.glb`
- `assets/brain_macaque.glb`
- `assets/brain_zebrafish.glb`

Those GLB files are not included in this branch. Mouse and zebrafish are
mesh-source-ready in the upstream atlas data; rat, marmoset, and macaque need
volume-to-mesh extraction before GLB packing.

## Species Notes

- Mouse: Allen Mouse CCFv3 catalog has 840 mesh-backed rows.
- Rat: NITRC describes 222 structures, while the downloaded ITK-SNAP label file
  contains 225 label entries including background. The frontend catalog
  preserves all 225 rows and exposes 224 non-background expected mesh names.
- Marmoset: merged NIH Marmoset Atlas v1.1 label-workbook catalog across five
  parcellation blocks. `group` means parcellation block, not anatomical
  hierarchy.
- Macaque prototype: D99 v2.0 has 368 rows. The staged source archive provided
  365 right-hemisphere GIFTI surface labels and 3 labels that require right
  atlas-volume extraction.
- Zebrafish: mapZebrain manifest has 265 regions total and 259 rows with STL
  paths. STL files are not committed here.

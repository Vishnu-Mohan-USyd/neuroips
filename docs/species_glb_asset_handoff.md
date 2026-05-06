# Species GLB Asset Handoff

This branch includes ready-to-use frontend species catalogs, CSV provenance
catalogs, and helper scripts. It does not include raw atlas source downloads or
generated species GLBs.

Target runtime assets should be written under `assets/` as:

- `assets/brain_mouse.glb`
- `assets/brain_rat.glb`
- `assets/brain_marmoset.glb`
- `assets/brain_macaque.glb`
- `assets/brain_zebrafish.glb`

The JSON catalogs already point at these paths through `catalog.asset.path`.
The GLB files still need to be generated separately.

## Frontend-Ready Catalogs

The frontend coder should consume the already generated static files under
`frontend_data/`. They do not need Python, pandas, openpyxl, or any atlas
scripts.

Ready files:

- `frontend_data/species_catalogs/mouse.json`
- `frontend_data/species_catalogs/rat.json`
- `frontend_data/species_catalogs/marmoset.json`
- `frontend_data/species_catalogs/macaque_prototype.json`
- `frontend_data/species_catalogs/zebrafish.json`
- `frontend_data/species_catalogs.js`
- `frontend_data/README.md`

For the existing site ZIP layout, the same files are mirrored under:

- `data/species_catalogs/*.json`
- `data/species_catalogs.js`
- `data/species_catalogs.README.md`

Use `frontend_data/species_catalogs.js` for fetchable JSON paths, or fetch the
JSON files directly. In frontend code, load the species JSON, load the expected
GLB at `catalog.asset.path`, then join GLB objects with:

```js
const region = catalog.regionsByMeshName[mesh.name];
```

The authoritative expected mesh/object name is `regions[].meshName`. Rows with
an empty `meshName` have no expected GLB object. Use `regions[].status` for
frontend filtering; it is the normalized alias of `catalogStatusKind`.

## Label Catalog CSVs

The CSV catalogs are already generated and are provenance/intermediate data for
GLB builders and maintainers. Frontend code should use `frontend_data/` instead
of parsing CSVs or running scripts.

Generated CSVs:

- `external/atlases/mouse/catalog_mouse_labels.csv`
- `external/atlases/rat/catalog_rat_labels.csv`
- `external/atlases/zebrafish/catalog_zebrafish_regions.csv`
- `external/atlases/marmoset/catalog_marmoset_labels.csv`
- `external/atlases/macaque_prototype/catalog_macaque_prototype_labels.csv`

Each CSV has this schema:

```text
species,source_asset,source_mesh_file,glb_mesh_name,structure_id,acronym,label,group,parent_id,hemisphere,catalog_status,notes
```

Use `glb_mesh_name` as the join key for mesh objects exported by the current
Blender packer when a source mesh exists. For frontend joins, prefer the static
JSON `meshName` field because it resolves any duplicate CSV mesh names. For
volume-derived species, name each derived STL/OBJ from the JSON `meshName`
before packing so labels can be joined back to the GLB after export.
`catalog_status` records whether a row came from an existing mesh, volume label
table, metadata-only manifest, or archive label.

Blank CSV `group` values are normalized to frontend group `Ungrouped`; the
original source value is preserved in JSON as `sourceGroup`. Duplicate
biological labels are preserved when `structureId` or `meshName` differs.

## Catalog Status

| Species | Committed catalog status | GLB path | Conversion status |
| --- | --- | --- | --- |
| mouse | 840 frontend rows / 840 mesh keys | `assets/brain_mouse.glb` | Mesh-source-ready from Allen OBJ data; GLB not committed |
| zebrafish | 265 frontend rows / 259 mesh keys | `assets/brain_zebrafish.glb` | Mesh-source-ready after mapZebrain STL download; GLB not committed |
| rat | 225 frontend rows / 224 mesh keys | `assets/brain_rat.glb` | Volume-to-mesh required; GLB not committed |
| marmoset | 332 frontend rows / 332 mesh keys | `assets/brain_marmoset.glb` | Volume-to-mesh across five parcellation blocks; GLB not committed |
| macaque | 368 frontend rows / 368 mesh keys | `assets/brain_macaque.glb` | 365 right-surface rows plus 3 volume-extraction rows; GLB not committed |

## License And Citation Caveats

These staged files come from third-party atlas projects. Before committing,
redistributing, or publishing generated GLBs, verify the current terms for each
source. Generated GLBs can inherit upstream license and citation obligations
even when they are simplified or repacked.

- Rat WHS/NITRC is documented in this handoff as CC BY 4.0.
- NIH Marmoset Atlas v1.1 is documented in this handoff as CC BY-NC-SA 4.0.
- D99/AFNI states atlas datasets may not be modified or redistributed without
  prior consent; treat macaque prototype data as internal unless permission is
  obtained.
- Verify Allen CCFv3 and mapZebrain reuse/citation terms before redistribution.

## Mesh-Ready Pack Commands

Create the output directory first:

```bash
mkdir -p assets
```

Mouse can be packed directly from Allen OBJ meshes after retrieving the raw
Allen source mesh directory. The default scale preserves source coordinates;
choose `--scale` only after confirming the desired runtime unit convention.
Mouse catalog `glb_mesh_name` values match the numeric OBJ stems produced by the
current Blender packer and the frontend JSON `meshName`.

```bash
blender -b --python scripts/blender_pack_meshes_to_glb.py -- \
  --input-dir external/atlases/mouse/source/structure_meshes \
  --patterns "*.obj" \
  --output assets/brain_mouse.glb \
  --recenter
```

For a smaller preview GLB, add conservative decimation:

```bash
blender -b --python scripts/blender_pack_meshes_to_glb.py -- \
  --input-dir external/atlases/mouse/source/structure_meshes \
  --patterns "*.obj" \
  --output assets/brain_mouse.glb \
  --recenter \
  --decimate-ratio 0.35
```

Zebrafish should first fetch metadata, then explicitly download region STLs:

```bash
python scripts/download_mapzebrain_regions.py
python scripts/download_mapzebrain_regions.py --download
blender -b --python scripts/blender_pack_meshes_to_glb.py -- \
  --input-dir external/atlases/zebrafish/source/regions \
  --patterns "*.stl" \
  --output assets/brain_zebrafish.glb \
  --recenter
```

The audited mapZebrain version `2.0 (MECE, 2024)` manifest reports 265 regions
total and 259 regions with STL paths. Confirm reuse terms before publishing
generated zebrafish GLBs. The zebrafish catalog uses the manifest
`local_file` stem as `glb_mesh_name`, matching the expected STL filename after
`--download`. The frontend JSON omits `meshName` for zebrafish rows that have no
STL path.

## Volume-To-Mesh Next Steps

For rat, marmoset, and macaque, generate one OBJ or STL per labeled region
before using the Blender packer.

Rat note: NITRC describes 222 structures, while the downloaded ITK-SNAP label
file contains 225 label entries including background. The catalogs intentionally
preserve all 225 rows and expose 224 non-background `meshName` values.

Marmoset note: this is a merged NIH Marmoset Atlas v1.1 label-workbook catalog
across five parcellation blocks. For marmoset, `group` means parcellation block,
not anatomical hierarchy. vL, vM, and vH source volumes are referenced as
members inside `source/NIH_Marmoset_Atlas_V1_master.zip`.

Macaque note: 365 catalog rows reference right-hemisphere GIFTI surfaces inside
the D99 archive. Three rows have no right-surface member and reference
`D99_atlas_v2.0_right.nii.gz` for volume extraction.

1. Load the label volume and label table for the species.
2. Extract a surface per nonzero label with a reproducible marching-cubes
   workflow such as 3D Slicer, ITK-SNAP export, or a scripted
   `nibabel`/`scikit-image` pipeline.
3. Preserve the atlas label id and region name in each mesh filename, for
   example `001_cortex.stl`.
4. Prefer the matching frontend JSON `regions[].meshName` for each derived mesh
   stem.
5. Save derived meshes under `external/atlases/<species>/derived/meshes`.
6. Pack the species GLB:

```bash
blender -b --python scripts/blender_pack_meshes_to_glb.py -- \
  --input-dir external/atlases/rat/derived/meshes \
  --patterns "*.stl" "*.obj" \
  --output assets/brain_rat.glb \
  --recenter

blender -b --python scripts/blender_pack_meshes_to_glb.py -- \
  --input-dir external/atlases/marmoset/derived/meshes \
  --patterns "*.stl" "*.obj" \
  --output assets/brain_marmoset.glb \
  --recenter

blender -b --python scripts/blender_pack_meshes_to_glb.py -- \
  --input-dir external/atlases/macaque_prototype/derived/meshes \
  --patterns "*.stl" "*.obj" \
  --output assets/brain_macaque.glb \
  --recenter
```

For macaque, first unpack `D99_v2.0_dist.tgz` into a derived work directory and
surface the atlas volume or reuse any included surfaces only after checking their
hemisphere and region coverage.

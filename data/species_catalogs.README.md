# Frontend Species Catalog Data

This directory is ready-to-use frontend data. Do not run Python or atlas scripts
from the frontend project. Load the JSON catalogs directly.

## Files

- `species_catalogs/mouse.json`
- `species_catalogs/rat.json`
- `species_catalogs/marmoset.json`
- `species_catalogs/macaque_prototype.json`
- `species_catalogs/zebrafish.json`
- `species_catalogs.js`

The JSON files are intentionally separate from the JS module because the mouse
catalog is large. `species_catalogs.js` exports fetchable paths and helper
functions.

## Usage

```js
import { fetchSpeciesCatalog, speciesAssetPaths } from "./frontend_data/species_catalogs.js";

const catalog = await fetchSpeciesCatalog("mouse");
const glbPath = catalog.asset.path || speciesAssetPaths.mouse;

// After loading the GLB, use each mesh object's name as the lookup key.
const region = catalog.regionsByMeshName[mesh.name];
if (region) {
  mesh.userData.region = region;
  mesh.material.color.set(region.groupColor);
}
```

If your app serves static files from a public root, keep this directory layout
intact or adjust `speciesCatalogPaths` in `species_catalogs.js`.

## Required Fields

Catalog root fields:

- `schemaVersion`: current value is `1`.
- `species.id`: one of `mouse`, `rat`, `marmoset`, `macaque_prototype`, `zebrafish`.
- `species.displayName`: UI label.
- `species.atlasName`: upstream atlas name.
- `species.catalogNotes`: species-specific provenance and semantics.
- `asset.path`: expected GLB path, such as `assets/brain_mouse.glb`.
- `licenseCaveat`: upstream license/citation reminder.
- `meshNamingConvention.description`: how GLB mesh names are expected to match.
- `groupNormalization`: documents that blank source groups become `Ungrouped`.
- `duplicateLabelPolicy`: labels are source text and may repeat across distinct structures.
- `counts.status`: normalized status counts for the frontend.
- `groups`: array of `{ id, label, color, count }`.
- `regions`: array of region records.
- `regionsByMeshName`: object keyed by exact expected GLB mesh/object names.

Region fields:

- `meshName`: authoritative expected GLB mesh/object name. Empty means no mesh is expected for that row.
- `catalogMeshName`: original CSV mesh-name value before frontend disambiguation.
- `structureId`: atlas label or structure id.
- `acronym`: atlas acronym or short name when available.
- `label`: human-readable region label.
- `sourceGroup`: original CSV group value. May be blank.
- `group`, `groupId`, `groupColor`: normalized frontend grouping and stable display color. Blank `sourceGroup` becomes `Ungrouped`.
- `parentId`: parent structure id when available.
- `hemisphere`: hemisphere label when available.
- `sourceAsset`: source metadata file used for the row.
- `sourceMeshFile`: source mesh or source volume path when available.
- `status`: normalized frontend status for UI filtering.
- `catalogStatus`: source-specific status from the generated CSV.
- `catalogStatusKind`: legacy alias/detail for `status`.
- `notes`: extra source notes.

Normalized `status` values currently used:

- `mesh-ready`: source row already maps to a mesh file.
- `volume-backed`: source row maps to a volume label and needs mesh extraction.
- `archive-surface`: source row maps to an archive surface file and needs conversion.
- `metadata-only`: source row has downloadable mesh metadata but no local mesh yet.
- `metadata-no-mesh`: source row has no mesh path.
- `background`: background/non-region label row.

## Mesh Name Contract

Use `regionsByMeshName[mesh.name]` after loading a GLB. The GLB builder should
name mesh objects with `regions[].meshName`.

Mouse and zebrafish mesh names are expected to match source OBJ/STL stems.
Rat, marmoset, and macaque prototype GLBs are not yet generated; derived meshes
must be named from `regions[].meshName` before Blender packing. The marmoset
JSON disambiguates three duplicated CSV names by prefixing the source group, so
the JSON `meshName` field is authoritative.

For marmoset, `group` means parcellation block, not anatomical hierarchy. The
JSON preserves duplicate biological labels when `structureId` or `meshName`
differs.

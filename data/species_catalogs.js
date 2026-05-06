// Static atlas catalog entrypoint for frontend code.
// The large catalogs live in JSON files so bundlers do not need to inline them.

export const speciesCatalogPaths = Object.freeze({
  mouse: "./species_catalogs/mouse.json",
  rat: "./species_catalogs/rat.json",
  marmoset: "./species_catalogs/marmoset.json",
  macaque_prototype: "./species_catalogs/macaque_prototype.json",
  zebrafish: "./species_catalogs/zebrafish.json",
});

export const speciesAssetPaths = Object.freeze({
  mouse: "assets/brain_mouse.glb",
  rat: "assets/brain_rat.glb",
  marmoset: "assets/brain_marmoset.glb",
  macaque_prototype: "assets/brain_macaque.glb",
  zebrafish: "assets/brain_zebrafish.glb",
});

export const speciesIds = Object.freeze(Object.keys(speciesCatalogPaths));

export function catalogUrl(speciesId) {
  const path = speciesCatalogPaths[speciesId];
  if (!path) {
    throw new Error(`Unknown species catalog: ${speciesId}`);
  }
  return new URL(path, import.meta.url).href;
}

export async function fetchSpeciesCatalog(speciesId, fetchImpl = fetch) {
  const response = await fetchImpl(catalogUrl(speciesId));
  if (!response.ok) {
    throw new Error(`Failed to load ${speciesId} catalog: ${response.status}`);
  }
  return response.json();
}

export async function fetchAllSpeciesCatalogs(fetchImpl = fetch) {
  const entries = await Promise.all(
    speciesIds.map(async (speciesId) => [speciesId, await fetchSpeciesCatalog(speciesId, fetchImpl)]),
  );
  return Object.fromEntries(entries);
}

export default speciesCatalogPaths;

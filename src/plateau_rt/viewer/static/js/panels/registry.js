import overview from "./overview.js";
import bundles from "./bundles.js";

export const HOME_PANEL = bundles;
// One line per panel, in tab order:
export const PANELS = [overview];

/**
 * List the panels supporting one member kind.
 * @param {string} kind member kind
 * @returns {Array<object>} matching panels in tab order
 */
export function panelsForKind(kind) {
  return PANELS.filter((panel) => panel.kinds.includes(kind));
}

/**
 * Find a panel by id.
 * @param {string} id panel id
 * @returns {object|null} the panel or null
 */
export function getPanel(id) {
  return PANELS.find((panel) => panel.id === id) || null;
}

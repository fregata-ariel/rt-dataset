/**
 * Every user-visible string in the viewer. Panels and the router import
 * these instead of embedding English sentences in markup code.
 */
export const S = Object.freeze({
  appTitle: "plateau_rt viewer",
  loading: "Loading…",
  deriving: "Deriving overview…",
  derivingStage: "Deriving overview ({stage})…",
  invalidLink: "Invalid link.",
  backHome: "Back to bundles",
  notRegistered: "This bundle is not registered in this viewer: {digest}",
  unknownMember: "Unknown member {member}.",
  noPanels: "No panels available for this member kind.",
  unsupportedLinkVersion: "This link uses an unsupported version; some selections may be ignored.",
  yes: "Yes",
  no: "No",
  retry: "Retry",
  cancel: "Cancel",
  delete: "Delete",
  confirmDelete: "Delete",
  upload: "Upload",
  chooseFile: "Choose file",
  reasons: Object.freeze({
    timeout: "Timed out",
    memory_limit: "Out of memory",
    restart: "Interrupted by restart",
    error: "Deriver error",
    crashed: "Crashed",
  }),
  bundlesTitle: "Bundles",
  uploadTitle: "Upload a bundle",
  dropzone: "Drop an archive here or choose a file",
  bundleName: "Name",
  uploading: "Uploading…",
  validating: "Validating…",
  uploadProgress: "{phase} {percent}%",
  uploaded: "Uploaded {name}: {digest}",
  alreadyRegistered: "Already registered {name}: {digest}",
  noBundles: "No bundles registered yet.",
  deletePrompt: "Type {prefix} to confirm deletion.",
  statusComplete: "Complete",
  statusFailed: "Failed ({n})",
  statusProgress: "{done} of {total} ready",
  statusUnavailable: "—",
  colName: "Name",
  colDigest: "Digest",
  colMembers: "Members",
  colViewsBs: "Views / BS",
  colSize: "Size",
  colRegistered: "Registered",
  colDerivations: "Derivations",
  colActions: "Actions",
  viewsBs: "{views} / {bs}",
  memberLabel: "{id} ({kind})",
  overviewTitle: "Overview",
  sectionDataset: "Dataset",
  sectionFrequency: "Frequency and delay",
  sectionCamera: "Camera model",
  sectionBaseStations: "Base stations",
  sectionViews: "Views",
  sectionPairs: "View / base-station pairs",
  sectionAxes: "Image axes",
  sectionContents: "Contents",
  keySchemaVersion: "Schema version",
  keyMode: "Mode",
  keySourceScene: "Source scene",
  keyNumViews: "Views",
  keyNumBs: "Base stations",
  keyCarrier: "Carrier frequency",
  keyBandwidth: "Bandwidth",
  keyNumBins: "Frequency bins",
  keyBinSpacing: "Bin spacing",
  keyDelayResolution: "Delay resolution",
  keyUnambiguousDelay: "Unambiguous delay",
  keyFft: "FFT rows x cols",
  keyRx: "RX rows x cols",
  keyHSpacing: "Horizontal spacing",
  keyVSpacing: "Vertical spacing",
  keyHemispheres: "Hemispheres",
  lambdaUnit: "{x} lambda",
  colId: "ID",
  colIndex: "Index",
  colPosition: "Position (m)",
  colLookAt: "Look-at (m)",
  colView: "View",
  colBs: "BS",
  colTotalEnergy: "Total energy",
  colBackFraction: "Back fraction",
  colFrontHemisphere: "BS in front hemisphere",
  axesFromManifest: "Axes from the manifest.",
  axesDefaultM1: "Default M1 axes (the manifest has no image axes).",
});

/**
 * Fill `{key}` placeholders in a template with string values.
 * @param {string} template template with `{key}` placeholders
 * @param {Object<string, unknown>} [values] replacement values
 * @returns {string} the template with known keys replaced
 */
export function fmt(template, values = {}) {
  return String(template).replace(/\{([^{}]+)\}/g, (match, key) => {
    if (Object.prototype.hasOwnProperty.call(values, key)) {
      return String(values[key]);
    }
    return match;
  });
}

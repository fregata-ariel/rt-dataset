import { clear, errorView, h, loadingView } from "../dom.js";
import {
  formatBool,
  formatEnergy,
  formatHz,
  formatNumber,
  formatSecondsAsNs,
  formatVector,
} from "../format.js";
import { S, fmt } from "../strings.js";

let controller = null;
let context = null;
let host = null;
let rows = [];

/**
 * Build a key/value table from [label, value] rows.
 * @param {Array<[string, string]>} entries label and value pairs
 * @returns {Element} the table element
 */
function kvTable(entries) {
  const body = h("tbody");
  for (const [label, value] of entries) {
    body.appendChild(h("tr", {}, h("th", { scope: "row" }, label), h("td", {}, value)));
  }
  return h("table", { class: "kv" }, body);
}

/**
 * Render one array value as joined text.
 * @param {unknown} value candidate array
 * @returns {string} joined items or the unavailable mark
 */
function joined(value) {
  if (!Array.isArray(value) || value.length === 0) {
    return S.statusUnavailable;
  }
  return value.map((item) => String(item)).join(", ");
}

/**
 * Mark the row matching the state as selected.
 * @param {object} state selection state
 * @returns {void}
 */
function applySelection(state) {
  for (const row of rows) {
    const match = row.view === state.view && row.bs === state.bs;
    row.el.classList.toggle("selected", match);
  }
}

/**
 * Render the overview document into the host element.
 * @param {object} overview parsed overview.json
 * @returns {void}
 */
function render(overview) {
  clear(host);
  rows = [];
  const frequency = overview.frequency || {};
  const camera = overview.camera_model || {};
  host.appendChild(h("h2", {}, S.sectionDataset));
  host.appendChild(
    kvTable([
      [S.keySchemaVersion, String(overview.schema_version)],
      [S.keyMode, String(overview.mode)],
      [
        S.keySourceScene,
        overview.source_scene === null || overview.source_scene === undefined
          ? S.statusUnavailable
          : String(overview.source_scene),
      ],
      [S.keyNumViews, formatNumber(overview.num_views)],
      [S.keyNumBs, formatNumber(overview.num_bs)],
    ]),
  );
  host.appendChild(h("h2", {}, S.sectionFrequency));
  host.appendChild(
    kvTable([
      [S.keyCarrier, formatHz(frequency.carrier_frequency_hz)],
      [S.keyBandwidth, formatHz(frequency.bandwidth_hz)],
      [S.keyNumBins, formatNumber(frequency.num_bins)],
      [S.keyBinSpacing, formatHz(frequency.bin_spacing_hz)],
      [S.keyDelayResolution, formatSecondsAsNs(frequency.delay_resolution_s)],
      [S.keyUnambiguousDelay, formatSecondsAsNs(frequency.unambiguous_delay_s)],
    ]),
  );
  host.appendChild(h("h2", {}, S.sectionCamera));
  host.appendChild(
    kvTable([
      [S.keyFft, `${formatNumber(camera.fft_rows)} x ${formatNumber(camera.fft_cols)}`],
      [S.keyRx, `${formatNumber(camera.rx_rows)} x ${formatNumber(camera.rx_cols)}`],
      [S.keyHSpacing, fmt(S.lambdaUnit, { x: formatNumber(camera.horizontal_spacing_lambda) })],
      [S.keyVSpacing, fmt(S.lambdaUnit, { x: formatNumber(camera.vertical_spacing_lambda) })],
      [S.keyHemispheres, joined(camera.hemispheres)],
    ]),
  );
  host.appendChild(h("h2", {}, S.sectionBaseStations));
  const bsBody = h("tbody");
  for (const bs of overview.base_stations || []) {
    bsBody.appendChild(
      h(
        "tr",
        {},
        h("td", {}, String(bs.bs_id)),
        h("td", {}, formatNumber(bs.index)),
        h("td", {}, formatVector(bs.position_m)),
        h("td", {}, formatVector(bs.look_at_m)),
      ),
    );
  }
  host.appendChild(
    h(
      "table",
      { class: "stations" },
      h(
        "thead",
        {},
        h(
          "tr",
          {},
          h("th", {}, S.colId),
          h("th", {}, S.colIndex),
          h("th", {}, S.colPosition),
          h("th", {}, S.colLookAt),
        ),
      ),
      bsBody,
    ),
  );
  host.appendChild(h("h2", {}, S.sectionViews));
  const viewBody = h("tbody");
  for (const view of overview.views || []) {
    viewBody.appendChild(
      h(
        "tr",
        {},
        h("td", {}, String(view.view_id)),
        h("td", {}, formatNumber(view.index)),
        h("td", {}, formatVector(view.position_m)),
        h("td", {}, formatVector(view.look_at_m)),
      ),
    );
  }
  host.appendChild(
    h(
      "table",
      { class: "views" },
      h(
        "thead",
        {},
        h(
          "tr",
          {},
          h("th", {}, S.colId),
          h("th", {}, S.colIndex),
          h("th", {}, S.colPosition),
          h("th", {}, S.colLookAt),
        ),
      ),
      viewBody,
    ),
  );
  host.appendChild(h("h2", {}, S.sectionPairs));
  const hemispheres = Array.isArray(camera.hemispheres) ? camera.hemispheres : [];
  const headRow = h(
    "tr",
    {},
    h("th", {}, S.colView),
    h("th", {}, S.colBs),
    ...hemispheres.map((name) => h("th", {}, String(name))),
    h("th", {}, S.colTotalEnergy),
    h("th", {}, S.colBackFraction),
    h("th", {}, S.colFrontHemisphere),
  );
  const pairBody = h("tbody");
  for (const pair of overview.pairs || []) {
    const energy = pair.hemisphere_energy || {};
    const cells = [
      h("td", {}, String(pair.view_id)),
      h("td", {}, String(pair.bs_id)),
      ...hemispheres.map((name) => h("td", {}, formatEnergy(energy[name] ?? null))),
      h("td", {}, formatEnergy(pair.total_energy)),
      h("td", {}, formatNumber(pair.back_fraction)),
      h("td", {}, formatBool(pair.bs_in_front_hemisphere)),
    ];
    const tr = h(
      "tr",
      {
        dataset: { view: String(pair.view_id), bs: String(pair.bs_id) },
        onclick: () => {
          context.store.set({ view: pair.view_id, bs: pair.bs_id });
        },
      },
      cells,
    );
    rows.push({ el: tr, view: pair.view_id, bs: pair.bs_id });
    pairBody.appendChild(tr);
  }
  host.appendChild(h("table", { class: "pairs" }, h("thead", {}, headRow), pairBody));
  const axes = overview.image_axes || {};
  host.appendChild(h("h2", {}, S.sectionAxes));
  host.appendChild(
    h("p", {}, axes.source === "manifest" ? S.axesFromManifest : S.axesDefaultM1),
  );
  host.appendChild(
    kvTable(Object.entries(axes.axes || {}).map(([key, value]) => [String(key), String(value)])),
  );
  host.appendChild(h("h2", {}, S.sectionContents));
  const contents = overview.contents || {};
  host.appendChild(
    kvTable([
      ["optical", formatBool(contents.optical)],
      ["optical_artifacts", joined(contents.optical_artifacts)],
      ["transforms_json", formatBool(contents.transforms_json)],
      ["path_gt", formatBool(contents.path_gt)],
      ["path_schema", formatBool(contents.path_schema)],
      ["observations", joined(contents.observations)],
      ["partials", joined(contents.partials)],
      [
        "scene",
        contents.scene === null || contents.scene === undefined
          ? S.statusUnavailable
          : String(contents.scene),
      ],
      ["placement", formatBool(contents.placement)],
      ["tomography_gt", formatBool(contents.tomography_gt)],
    ]),
  );
  applySelection(context.store.get());
}

/**
 * Load the overview derivation and render it, with retry on failure.
 * @param {string} digest bundle digest
 * @param {object} member member record
 * @param {AbortSignal} signal abort signal
 * @returns {Promise<void>} resolves after rendering
 */
async function load(digest, member, signal) {
  clear(host);
  const loading = loadingView(S.deriving);
  host.appendChild(loading);
  const onStatus = (job) => {
    clear(host);
    host.appendChild(
      loadingView(job && job.stage ? fmt(S.derivingStage, { stage: job.stage }) : S.deriving),
    );
  };
  try {
    const payload = await context.api.derive(digest, member.id, "overview", {}, { signal, onStatus });
    const overview = await context.api.fetchDerivedJson(payload, "overview.json", { signal });
    if (signal.aborted) {
      return;
    }
    render(overview);
  } catch (err) {
    if (signal.aborted) {
      return;
    }
    clear(host);
    host.appendChild(
      errorView(err, {
        onRetry: async () => {
          const payload = await context.api.retryDerive(
            digest,
            member.id,
            "overview",
            {},
            { signal },
          );
          const overview = await context.api.fetchDerivedJson(payload, "overview.json");
          if (!signal.aborted) {
            render(overview);
          }
        },
      }),
    );
  }
}

const overviewPanel = {
  id: "overview",
  title: S.overviewTitle,
  kinds: ["rf_dataset"],
  /**
   * Mount the overview panel and start the derivation.
   * @param {Element} el host element
   * @param {object} ctx panel context with store, digest and member
   * @returns {Promise<void>} resolves after the first render
   */
  async mount(el, ctx) {
    controller = new AbortController();
    context = ctx;
    host = el;
    rows = [];
    await load(ctx.digest, ctx.member, controller.signal);
  },
  /**
   * Highlight the selected pair row.
   * @param {object} state selection state
   * @returns {void}
   */
  update(state) {
    applySelection(state);
  },
  /**
   * Abort pending requests and drop panel state.
   * @returns {void}
   */
  unmount() {
    if (controller) {
      controller.abort();
    }
    controller = null;
    context = null;
    host = null;
    rows = [];
  },
};

export default overviewPanel;

import * as apiModule from "../api.js";
import { clear, emptyView, errorView, h, loadingView } from "../dom.js";
import { formatBytes, shortDigest } from "../format.js";
import { S, fmt } from "../strings.js";

let controller = null;
let context = null;
let root = null;
let listHost = null;
let messageHost = null;
let fileInput = null;
let nameInput = null;
let startButton = null;
let dropzone = null;
let fileLabel = null;
let progressEl = null;
let progressText = null;
let statusTimer = 0;
let statusDelay = 0;
let selectedFile = null;
let onDragOver = null;
let onDrop = null;

/**
 * Return the API module from the panel context.
 * @returns {object} the API module
 */
function api() {
  return (context && context.api) || apiModule;
}

/**
 * Update the upload progress display.
 * @param {{loaded: number, total: number, phase: string}} progress progress event
 * @returns {void}
 */
function showProgress(progress) {
  const phase = progress.phase === "uploading" ? S.uploading : S.validating;
  const percent = progress.total ? Math.round((progress.loaded / progress.total) * 100) : 0;
  progressEl.hidden = false;
  progressEl.value = percent;
  progressText.textContent = fmt(S.uploadProgress, { phase, percent });
}

/**
 * Hide the progress bar once an upload has finished.
 * @returns {void}
 */
function hideProgress() {
  progressEl.hidden = true;
  progressText.textContent = "";
}

/**
 * Enable or disable the upload controls.
 * @param {boolean} disabled True to disable
 * @returns {void}
 */
function setUploadEnabled(disabled) {
  fileInput.disabled = disabled;
  nameInput.disabled = disabled;
  startButton.disabled = disabled || selectedFile === null;
}

/**
 * Render the derivation status of one bundle as text.
 * @param {object|null} status bundleStatus body (or null when unavailable)
 * @returns {string} status text
 */
function statusText(status) {
  if (!status) {
    return S.statusUnavailable;
  }
  if (status.complete) {
    return S.statusComplete;
  }
  const failures = Array.isArray(status.failures) ? status.failures.length : 0;
  const eagerFailed = status.eager && status.eager.failed ? status.eager.failed : 0;
  if (failures > 0 || eagerFailed > 0) {
    return fmt(S.statusFailed, { n: failures > 0 ? failures : eagerFailed });
  }
  const done = status.eager && status.eager.done ? status.eager.done : 0;
  const total = status.eager && status.eager.total ? status.eager.total : 0;
  return fmt(S.statusProgress, { done, total });
}

/**
 * Return True when a bundle still needs status refreshes.
 * @param {object|null} status bundleStatus body (or null when unavailable)
 * @returns {boolean} True when polling should continue
 */
function needsRefresh(status) {
  if (!status || status.complete) {
    return false;
  }
  const failures = Array.isArray(status.failures) ? status.failures.length : 0;
  const eagerFailed = status.eager && status.eager.failed ? status.eager.failed : 0;
  return failures === 0 && eagerFailed === 0;
}

/**
 * Render the views/BS cell for one bundle.
 * @param {object} bundle bundle summary
 * @returns {string} cell text
 */
function viewsBsText(bundle) {
  const parts = [];
  for (const member of bundle.members || []) {
    if (member.kind === "rf_dataset" && member.validation && member.validation.summary) {
      const summary = member.validation.summary;
      parts.push(fmt(S.viewsBs, { views: summary.num_views, bs: summary.num_bs }));
    }
  }
  return parts.length > 0 ? parts.join(", ") : S.statusUnavailable;
}

/**
 * Stop the scheduled status refresh.
 * @returns {void}
 */
function stopRefresh() {
  if (statusTimer) {
    clearTimeout(statusTimer);
    statusTimer = 0;
  }
  statusDelay = 0;
}

/**
 * Refresh the statuses of incomplete bundles with backoff.
 * @param {Array<{digest: string, cell: Element}>} pending bundles needing refresh
 * @returns {void}
 */
function scheduleRefresh(pending) {
  stopRefresh();
  if (!controller || pending.length === 0) {
    return;
  }
  statusDelay = apiModule.nextPollDelay(statusDelay);
  statusTimer = setTimeout(async () => {
    statusTimer = 0;
    if (!controller) {
      return;
    }
    const signal = controller.signal;
    const still = [];
    for (const item of pending) {
      let status = null;
      try {
        status = await api().bundleStatus(item.digest, { signal });
      } catch {
        status = null;
      }
      if (!controller || signal.aborted) {
        return;
      }
      if (status === null) {
        still.push(item);
        continue;
      }
      item.cell.textContent = statusText(status);
      if (needsRefresh(status)) {
        still.push(item);
      }
    }
    if (still.length > 0) {
      scheduleRefresh(still);
    } else {
      statusDelay = 0;
    }
  }, statusDelay);
}

/**
 * Build the delete confirmation row for one bundle.
 * @param {string} digest bundle digest
 * @param {Element} anchor table row after which the confirmation appears
 * @returns {void}
 */
function showDeleteConfirm(digest, anchor) {
  const prefix = digest.slice(0, 8);
  const input = h("input", { type: "text", "aria-label": S.deletePrompt });
  const confirm = h("button", { type: "button", disabled: true }, S.confirmDelete);
  const cancel = h("button", { type: "button" }, S.cancel);
  input.addEventListener("input", () => {
    confirm.disabled = input.value !== prefix;
  });
  cancel.addEventListener("click", () => {
    confirmRow.remove();
  });
  confirm.addEventListener("click", async () => {
    confirm.disabled = true;
    try {
      await api().deleteBundle(digest);
      confirmRow.remove();
      await reloadList();
    } catch (err) {
      clear(listHost);
      listHost.appendChild(errorView(err));
    }
  });
  const confirmRow = h(
    "tr",
    { class: "delete-confirm" },
    h(
      "td",
      { colspan: "8" },
      h("span", {}, fmt(S.deletePrompt, { prefix })),
      input,
      confirm,
      cancel,
    ),
  );
  anchor.after(confirmRow);
  input.focus();
}

/**
 * Render bundle rows into a tbody, fetching each derivation status.
 * @param {Element} tbody table body element
 * @param {Array<object>} bundles bundle summaries
 * @param {AbortSignal} signal abort signal
 * @returns {Promise<void>} resolves after rendering
 */
async function renderRows(tbody, bundles, signal) {
  const pending = [];
  for (const bundle of bundles) {
    const digest = bundle.digest;
    const memberText = (bundle.members || [])
      .map((member) => fmt(S.memberLabel, { id: member.id, kind: member.kind }))
      .join(", ");
    const statusCell = h("td", {}, S.loading);
    const row = h(
      "tr",
      { dataset: { digest } },
      h("td", {}, h("a", { href: `#/b/${digest}` }, String(bundle.name))),
      h("td", {}, h("code", { title: digest }, shortDigest(digest))),
      h("td", {}, memberText),
      h("td", {}, viewsBsText(bundle)),
      h("td", {}, formatBytes(bundle.total_bytes)),
      h("td", {}, String(bundle.created_at)),
      statusCell,
      h(
        "td",
        {},
        h(
          "button",
          {
            type: "button",
            onclick: (event) => {
              showDeleteConfirm(digest, event.target.closest("tr"));
            },
          },
          S.delete,
        ),
      ),
    );
    tbody.appendChild(row);
    if (signal.aborted) {
      return;
    }
    let status = null;
    try {
      status = await api().bundleStatus(digest, { signal });
    } catch {
      status = null;
    }
    if (signal.aborted) {
      return;
    }
    statusCell.textContent = statusText(status);
    if (needsRefresh(status)) {
      pending.push({ digest, cell: statusCell });
    }
  }
  if (pending.length > 0) {
    scheduleRefresh(pending);
  }
}

/**
 * Load the bundle list into the list host.
 * @param {Element} host list container
 * @param {boolean} keepLoading True to keep the current content on failure
 * @returns {Promise<void>} resolves after rendering
 */
async function reloadListInto(host, keepLoading) {
  stopRefresh();
  if (!keepLoading) {
    clear(host);
    host.appendChild(loadingView());
  }
  const signal = controller.signal;
  try {
    const body = await api().listBundles({ signal });
    if (signal.aborted) {
      return;
    }
    clear(host);
    const bundles = (body && body.bundles) || [];
    if (bundles.length === 0) {
      host.appendChild(emptyView(S.noBundles));
      return;
    }
    const tbody = h("tbody");
    const table = h(
      "table",
      { class: "bundles" },
      h(
        "thead",
        {},
        h(
          "tr",
          {},
          h("th", {}, S.colName),
          h("th", {}, S.colDigest),
          h("th", {}, S.colMembers),
          h("th", {}, S.colViewsBs),
          h("th", {}, S.colSize),
          h("th", {}, S.colRegistered),
          h("th", {}, S.colDerivations),
          h("th", {}, S.colActions),
        ),
      ),
      tbody,
    );
    host.appendChild(table);
    await renderRows(tbody, bundles, signal);
  } catch (err) {
    if (signal.aborted) {
      return;
    }
    if (!keepLoading) {
      clear(host);
      host.appendChild(errorView(err));
    }
  }
}

/**
 * Reload the bundle list.
 * @returns {Promise<void>} resolves after rendering
 */
async function reloadList() {
  await reloadListInto(listHost, false);
}

/**
 * Start uploading the selected file.
 * @returns {Promise<void>} resolves after the upload finishes
 */
async function startUpload() {
  if (!selectedFile) {
    return;
  }
  const file = selectedFile;
  const name = nameInput.value || file.name;
  clear(messageHost);
  progressEl.hidden = true;
  setUploadEnabled(true);
  const signal = controller.signal;
  try {
    const body = await api().uploadBundle(file, {
      name,
      signal,
      onProgress: showProgress,
    });
    if (signal.aborted) {
      return;
    }
    const digest = body.digest;
    const created = body.created;
    hideProgress();
    clear(messageHost);
    const link = h("a", { href: `#/b/${digest}` }, shortDigest(digest));
    messageHost.appendChild(
      h(
        "div",
        { class: "status ok" },
        h(
          "span",
          {},
          fmt(created ? S.uploaded : S.alreadyRegistered, {
            name: String(name),
            digest: shortDigest(digest),
          }),
        ),
        link,
      ),
    );
    setUploadEnabled(false);
    await reloadList();
  } catch (err) {
    if (signal.aborted) {
      return;
    }
    hideProgress();
    clear(messageHost);
    messageHost.appendChild(errorView(err));
    setUploadEnabled(false);
  }
}

const bundlesPanel = {
  id: "bundles",
  title: S.bundlesTitle,
  /**
   * Mount the home panel with the upload box and bundle list.
   * @param {Element} el host element
   * @param {object} ctx panel context with store and api
   * @returns {Promise<void>} resolves after the first render
   */
  async mount(el, ctx) {
    controller = new AbortController();
    context = ctx;
    root = el;
    selectedFile = null;
    clear(el);
    fileInput = h("input", { type: "file", accept: ".zip,.tar,.tgz,.tar.gz" });
    nameInput = h("input", { type: "text" });
    startButton = h("button", { type: "button", disabled: true }, S.upload);
    dropzone = h("div", { class: "dropzone" }, S.dropzone);
    fileLabel = h("span", { class: "file-label" }, "");
    progressEl = h("progress", { max: "100", value: 0, hidden: true });
    progressText = h("span", { class: "progress-text" }, "");
    messageHost = h("div", { class: "upload-message" });
    listHost = h("div", { class: "bundle-list" });
    fileInput.addEventListener("change", () => {
      selectedFile = fileInput.files && fileInput.files[0] ? fileInput.files[0] : null;
      // The native input already shows the chosen name; the label is for dropped files.
      fileLabel.textContent = "";
      startButton.disabled = selectedFile === null;
    });
    dropzone.addEventListener("dragover", (event) => {
      event.preventDefault();
      dropzone.classList.add("dragover");
    });
    dropzone.addEventListener("dragleave", () => {
      dropzone.classList.remove("dragover");
    });
    dropzone.addEventListener("drop", (event) => {
      event.preventDefault();
      dropzone.classList.remove("dragover");
      if (event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0]) {
        selectedFile = event.dataTransfer.files[0];
        fileLabel.textContent = selectedFile.name;
        startButton.disabled = false;
      }
    });
    startButton.addEventListener("click", () => {
      startUpload();
    });
    onDragOver = (event) => {
      event.preventDefault();
    };
    onDrop = (event) => {
      event.preventDefault();
    };
    window.addEventListener("dragover", onDragOver);
    window.addEventListener("drop", onDrop);
    el.appendChild(h("h2", {}, S.uploadTitle));
    el.appendChild(dropzone);
    el.appendChild(
      h(
        "div",
        { class: "upload-controls" },
        fileInput,
        fileLabel,
        h("label", {}, h("span", {}, S.bundleName), nameInput),
        startButton,
      ),
    );
    el.appendChild(h("div", { class: "upload-progress" }, progressEl, progressText));
    el.appendChild(messageHost);
    el.appendChild(h("h2", {}, S.bundlesTitle));
    el.appendChild(listHost);
    await reloadList();
  },
  /**
   * Home panel ignores state changes.
   * @param {object} _state selection state
   * @returns {void}
   */
  update(_state) {},
  /**
   * Abort pending requests and remove window listeners.
   * @returns {void}
   */
  unmount() {
    stopRefresh();
    if (controller) {
      controller.abort();
    }
    if (onDragOver) {
      window.removeEventListener("dragover", onDragOver);
    }
    if (onDrop) {
      window.removeEventListener("drop", onDrop);
    }
    controller = null;
    context = null;
    root = null;
    listHost = null;
    messageHost = null;
    statusTimer = 0;
    statusDelay = 0;
    selectedFile = null;
    onDragOver = null;
    onDrop = null;
  },
};

export default bundlesPanel;

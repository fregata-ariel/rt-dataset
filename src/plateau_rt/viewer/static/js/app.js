import * as api from "./api.js";
import { clear, emptyView, errorView, h, loadingView } from "./dom.js";
import { shortDigest } from "./format.js";
import { parseNpy } from "./npy.js";
import { createStore, formatHash, parseHash } from "./state.js";
import { HOME_PANEL, panelsForKind } from "./panels/registry.js";
import { S, fmt } from "./strings.js";

let store = null;
let mainEl = null;
let mounted = null;
let controller = null;
let generation = 0;

/**
 * Build the mount key identifying the mounted panel instance.
 * @param {object} route current route
 * @returns {string} mount key
 */
function mountKey(route) {
  return JSON.stringify([route.name, route.digest || null, route.member || null, route.panel || null]);
}

/**
 * Unmount the current panel and abort its pending requests.
 * @returns {void}
 */
function unmountCurrent() {
  if (controller) {
    controller.abort();
    controller = null;
  }
  if (mounted) {
    const panel = mounted.panel;
    mounted = null;
    panel.unmount();
  }
}

/**
 * Render the application header with the home link.
 * @param {Element} app root element
 * @returns {void}
 */
function renderHeader(app) {
  app.appendChild(
    h("header", { class: "viewer-header" }, h("a", { href: "#/" }, S.appTitle)),
  );
  mainEl = h("main", { class: "viewer-main" });
  app.appendChild(mainEl);
}

/**
 * Show a small notice when the link version is not 1.
 * @param {number} version parsed link version
 * @returns {void}
 */
function maybeVersionNotice(version) {
  if (version !== 1) {
    mainEl.appendChild(h("div", { class: "notice" }, S.unsupportedLinkVersion));
  }
}

/**
 * Render the home route.
 * @param {number} version parsed link version
 * @param {number} gen render generation
 * @returns {Promise<void>} resolves after mounting
 */
async function renderHome(version, gen) {
  maybeVersionNotice(version);
  const host = h("div", { class: "panel", dataset: { panel: HOME_PANEL.id } });
  mainEl.appendChild(host);
  if (gen !== generation) {
    return;
  }
  // Register before awaiting: a newer render unmounts this panel through unmountCurrent().
  mounted = { panel: HOME_PANEL, key: mountKey({ name: "home" }) };
  await HOME_PANEL.mount(host, {
    store,
    api,
    navigate: (route, opts) => store.navigate(route, opts),
  });
}

/**
 * Render the invalid-link route.
 * @returns {void}
 */
function renderInvalid() {
  const box = errorView({ type: "invalid", member: null, message: S.invalidLink });
  box.appendChild(h("a", { href: "#/" }, S.backHome));
  mainEl.appendChild(box);
}

/**
 * Render the not-registered view for an unknown bundle digest.
 * @param {string} digest bundle digest
 * @returns {void}
 */
function renderNotRegistered(digest) {
  mainEl.appendChild(
    h(
      "div",
      { class: "status error" },
      h("span", {}, fmt(S.notRegistered, { digest })),
      h("a", { href: "#/" }, S.backHome),
    ),
  );
}

/**
 * Render the bundle header with the member selector.
 * @param {object} bundle bundle detail body
 * @param {object} route current route
 * @returns {void}
 */
function renderBundleHeader(bundle, route) {
  const header = h(
    "div",
    { class: "bundle-header" },
    h("span", { class: "bundle-name" }, String(bundle.name)),
    h("code", { title: bundle.digest }, shortDigest(bundle.digest)),
  );
  mainEl.appendChild(header);
  if (Array.isArray(bundle.members) && bundle.members.length > 1) {
    const bar = h("div", { class: "member-bar" });
    for (const member of bundle.members) {
      const active = member.id === route.member;
      const button = h(
        "button",
        {
          type: "button",
          class: active ? "member active" : "member",
          onclick: () => {
            store.navigate(
              { name: "bundle", digest: route.digest, member: member.id, panel: null },
              { state: "reset" },
            );
          },
        },
        fmt(S.memberLabel, { id: member.id, kind: member.kind }),
      );
      if (active) {
        button.setAttribute("aria-selected", "true");
      }
      bar.appendChild(button);
    }
    mainEl.appendChild(bar);
  }
}

/**
 * Render one bundle route.
 * @param {object} route current route
 * @param {number} version parsed link version
 * @param {number} gen render generation
 * @returns {Promise<void>} resolves after mounting
 */
async function renderBundle(route, version, gen) {
  const ctrl = new AbortController();
  controller = ctrl;
  mainEl.appendChild(loadingView());
  let bundle;
  try {
    bundle = await api.getBundle(route.digest, { signal: ctrl.signal });
  } catch (err) {
    if (gen !== generation || ctrl.signal.aborted) {
      return;
    }
    controller = null;
    clear(mainEl);
    if (err instanceof api.ApiError && err.type === "not_found") {
      renderNotRegistered(route.digest);
    } else {
      mainEl.appendChild(errorView(err));
    }
    return;
  }
  if (gen !== generation || ctrl.signal.aborted) {
    return;
  }
  controller = null;
  const members = (bundle && bundle.members) || [];
  let member = null;
  if (route.member) {
    member = members.find((entry) => entry.id === route.member) || null;
    if (!member) {
      clear(mainEl);
      mainEl.appendChild(
        errorView({
          type: "not_found",
          member: route.member,
          message: fmt(S.unknownMember, { member: route.member }),
        }),
      );
      return;
    }
  } else {
    member =
      members.find((entry) => panelsForKind(entry.kind).length > 0) || members[0] || null;
  }
  if (!member) {
    clear(mainEl);
    mainEl.appendChild(emptyView(S.noPanels));
    return;
  }
  const available = panelsForKind(member.kind);
  let panel = route.panel
    ? available.find((entry) => entry.id === route.panel) || null
    : null;
  if (!panel) {
    panel = available[0] || null;
  }
  const canonical = {
    name: "bundle",
    digest: route.digest,
    member: member.id,
    panel: panel ? panel.id : null,
  };
  if (
    route.member !== canonical.member ||
    (route.panel || null) !== (canonical.panel || null)
  ) {
    store.navigate(canonical, { replace: true });
    return;
  }
  clear(mainEl);
  maybeVersionNotice(version);
  renderBundleHeader(bundle, route);
  if (!panel) {
    mainEl.appendChild(emptyView(S.noPanels));
    return;
  }
  const tabs = h("div", { class: "tabs" });
  for (const entry of available) {
    const active = entry.id === panel.id;
    const tab = h(
      "button",
      {
        type: "button",
        class: active ? "tab active" : "tab",
        onclick: () => {
          store.navigate({ ...route, panel: entry.id });
        },
      },
      entry.title,
    );
    if (active) {
      tab.setAttribute("aria-selected", "true");
    }
    tabs.appendChild(tab);
  }
  mainEl.appendChild(tabs);
  const host = h("div", { class: "panel", dataset: { panel: panel.id } });
  mainEl.appendChild(host);
  // Register before awaiting: a newer render unmounts this panel through unmountCurrent().
  mounted = { panel, key: mountKey(route) };
  await panel.mount(host, {
    store,
    api,
    digest: route.digest,
    bundle,
    member,
    navigate: (next, opts) => store.navigate(next, opts),
  });
}

/**
 * Render the current route, remounting only when the panel changes.
 * @returns {Promise<void>} resolves after rendering
 */
async function render() {
  const gen = ++generation;
  if (!store || !mainEl) {
    return;
  }
  const route = store.route();
  const parsed = parseHash(window.location.hash);
  unmountCurrent();
  clear(mainEl);
  if (route.name === "home") {
    await renderHome(parsed.version, gen);
  } else if (route.name === "invalid") {
    renderInvalid();
  } else {
    await renderBundle(route, parsed.version, gen);
  }
}

/**
 * Initialize the router once the DOM is ready.
 * @returns {void}
 */
function init() {
  const app = document.getElementById("app");
  renderHeader(app);
  store = createStore({ onRouteChange: () => void render() });
  store.bindWindow();
  store.subscribe((state) => {
    if (mounted) {
      mounted.panel.update(state);
    }
  });
  window.__viewer = Object.freeze({ store, api, parseHash, formatHash, parseNpy });
  store.syncFromLocation();
  void render();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}

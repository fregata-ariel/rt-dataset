const DIGEST_RE = /^[0-9a-f]{64}$/;
const MEMBER_RE = /^[A-Za-z0-9_.-]{1,64}$/;
const PANEL_RE = /^[a-z][a-z0-9_-]{0,63}$/;
const ID_RE = /^[^\x00-\x1f\x7f]{1,200}$/;
const INT_RE = /^[0-9]+$/;
const PIXEL_RE = /^([0-9]+),([0-9]+)$/;

/** URL format version written as `v`. */
export const URL_VERSION = 1;

/** Selection fields in canonical URL order. */
export const FIELDS = [
  { key: "view", param: "view", type: "id", def: null },
  { key: "bs", param: "bs", type: "id", def: null },
  { key: "hemisphere", param: "hemi", type: "hemisphere", def: "front" },
  { key: "orientation", param: "orient", type: "orientation", def: "rf" },
  { key: "path", param: "path", type: "int", def: null },
  { key: "pixel", param: "px", type: "pixel", def: null },
  { key: "delayBin", param: "dbin", type: "int", def: null },
  { key: "variant", param: "variant", type: "id", def: null },
];

/**
 * Return the default selection state.
 * @returns {Object<string, unknown>} the default state
 */
export function defaultState() {
  return {
    view: null,
    bs: null,
    hemisphere: "front",
    orientation: "rf",
    path: null,
    pixel: null,
    delayBin: null,
    variant: null,
  };
}

/**
 * Validate one id field (1..200 chars, no control characters).
 * @param {unknown} value candidate value
 * @returns {boolean} True when valid
 */
function isId(value) {
  return typeof value === "string" && ID_RE.test(value);
}

/**
 * Validate one state value without throwing.
 * @param {string} key field key
 * @param {unknown} value candidate value
 * @returns {unknown} the validated value or undefined when invalid
 */
function validValue(key, value) {
  if (value === null || value === undefined) {
    return null;
  }
  if (key === "view" || key === "bs" || key === "variant") {
    return isId(value) ? value : undefined;
  }
  if (key === "hemisphere") {
    return value === "front" || value === "back" ? value : undefined;
  }
  if (key === "orientation") {
    return value === "rf" || value === "photo" ? value : undefined;
  }
  if (key === "path" || key === "delayBin") {
    if (typeof value === "number" && Number.isInteger(value) && value >= 0) {
      return value;
    }
    if (typeof value === "string" && INT_RE.test(value)) {
      return Number(value);
    }
    return undefined;
  }
  if (key === "pixel") {
    if (typeof value === "string") {
      const match = value.match(PIXEL_RE);
      return match ? [Number(match[1]), Number(match[2])] : undefined;
    }
    if (
      Array.isArray(value) &&
      value.length === 2 &&
      validValue("path", value[0]) !== undefined &&
      validValue("path", value[1]) !== undefined
    ) {
      return [Number(value[0]), Number(value[1])];
    }
    return undefined;
  }
  return undefined;
}

/**
 * Validate one state value, throwing on invalid input.
 * @param {string} key field key
 * @param {unknown} value candidate value (`null` resets to the default)
 * @returns {unknown} the validated value
 */
export function coerce(key, value) {
  const field = FIELDS.find((entry) => entry.key === key);
  if (!field) {
    throw new TypeError(`unknown state key ${key}`);
  }
  if (value === null) {
    return field.def === null ? null : field.def;
  }
  const checked = validValue(key, value);
  if (checked === undefined) {
    throw new TypeError(`invalid value for ${key}`);
  }
  if (field.def !== null && checked === null) {
    return field.def;
  }
  return checked;
}

/**
 * Decode one path segment, returning null on failure.
 * @param {string} segment raw path segment
 * @returns {string|null} the decoded segment
 */
function decodeSegment(segment) {
  try {
    return decodeURIComponent(segment);
  } catch {
    return null;
  }
}

/**
 * Parse a location hash into a route, state and URL version.
 * @param {string} hash the location hash (e.g. "#/b/<digest>")
 * @returns {{route: object, state: object, version: number}} parsed result
 */
export function parseHash(hash) {
  const state = defaultState();
  let version = 1;
  const text = typeof hash === "string" && hash.startsWith("#") ? hash.slice(1) : hash || "";
  if (text === "" || text === "/") {
    return { route: { name: "home" }, state, version };
  }
  const [pathPart, queryPart] = text.split("?", 2);
  const segments = pathPart.split("/");
  if (segments.length < 3 || segments[0] !== "" || segments[1] !== "b") {
    return { route: { name: "home" }, state, version };
  }
  if (segments.length > 5) {
    return { route: { name: "invalid", hash }, state, version };
  }
  const digest = decodeSegment(segments[2]);
  const member = segments.length > 3 ? decodeSegment(segments[3]) : null;
  const panel = segments.length > 4 ? decodeSegment(segments[4]) : null;
  if (
    digest === null ||
    !DIGEST_RE.test(digest) ||
    (member !== null && (member === "" || !MEMBER_RE.test(member))) ||
    (panel !== null && (panel === "" || !PANEL_RE.test(panel)))
  ) {
    return { route: { name: "invalid", hash }, state, version };
  }
  const params = new URLSearchParams(queryPart || "");
  const rawVersion = params.get("v");
  if (rawVersion !== null && INT_RE.test(rawVersion)) {
    version = Number(rawVersion);
  }
  for (const field of FIELDS) {
    const raw = params.get(field.param);
    if (raw === null) {
      continue;
    }
    if (field.key === "pixel") {
      const checked = validValue("pixel", raw);
      if (checked !== undefined) {
        state.pixel = checked;
      }
      continue;
    }
    const checked = validValue(field.key, raw);
    if (checked !== undefined && checked !== null) {
      state[field.key] = checked;
    } else if (checked === null && field.def === null) {
      state[field.key] = null;
    }
  }
  return { route: { name: "bundle", digest, member, panel }, state, version };
}

/**
 * Format one state value for the URL, or null when it equals the default.
 * @param {object} field FIELDS entry
 * @param {unknown} value current value
 * @returns {string|null} the encoded value or null
 */
function formatValue(field, value) {
  if (field.key === "pixel") {
    if (value === null) {
      return null;
    }
    return `${value[0]},${value[1]}`;
  }
  if (value === field.def) {
    return null;
  }
  if (value === null || value === undefined) {
    return null;
  }
  return String(value);
}

/**
 * Format a route and state as a shareable location hash.
 * @param {{name: string, digest?: string, member?: string|null, panel?: string|null}} route route
 * @param {object} state selection state
 * @returns {string} the location hash
 */
export function formatHash(route, state) {
  if (!route || route.name === "home") {
    return "#/";
  }
  let hash = `#/b/${route.digest}`;
  if (route.member) {
    hash += `/${encodeURIComponent(route.member)}`;
    if (route.panel) {
      hash += `/${encodeURIComponent(route.panel)}`;
    }
  }
  const query = [`v=${URL_VERSION}`];
  for (const field of FIELDS) {
    const text = formatValue(field, state[field.key]);
    if (text !== null) {
      query.push(`${field.param}=${encodeURIComponent(text)}`);
    }
  }
  return `${hash}?${query.join("&")}`;
}

/**
 * Compare two routes for equality.
 * @param {object} a first route
 * @param {object} b second route
 * @returns {boolean} True when equal
 */
function sameRoute(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

/**
 * Compare two state values for equality.
 * @param {unknown} a first value
 * @param {unknown} b second value
 * @returns {boolean} True when equal
 */
function sameValue(a, b) {
  if (Array.isArray(a) || Array.isArray(b)) {
    return (
      Array.isArray(a) && Array.isArray(b) && a.length === b.length && a[0] === b[0] && a[1] === b[1]
    );
  }
  return a === b;
}

/**
 * Copy a state object, freezing the result.
 * @param {object} state state to copy
 * @returns {object} the frozen copy
 */
function copyState(state) {
  const copy = { ...state };
  if (Array.isArray(copy.pixel)) {
    copy.pixel = [...copy.pixel];
  }
  return Object.freeze(copy);
}

/**
 * Create a hash-backed router store.
 * @param {{location?: object, history?: object, onRouteChange?: (route: object) => void}} [options] fakes and route listener
 * @returns {object} the store object
 */
export function createStore({ location, history, onRouteChange } = {}) {
  const loc =
    location || (typeof window !== "undefined" ? window.location : undefined);
  const hist =
    history || (typeof window !== "undefined" ? window.history : undefined);
  let route = { name: "home" };
  let state = defaultState();
  const listeners = new Set();

  /**
   * Write the current hash without adding a history entry.
   * @returns {void}
   */
  function writeReplace() {
    const hash = formatHash(route, state);
    if (hist && typeof hist.replaceState === "function") {
      hist.replaceState(null, "", hash);
    } else if (loc) {
      loc.hash = hash;
    }
  }

  /**
   * Notify subscribers about a state change.
   * @param {object} snapshot frozen state snapshot
   * @param {string[]} changed changed keys
   * @returns {void}
   */
  function notify(snapshot, changed) {
    for (const fn of [...listeners]) {
      fn(snapshot, changed);
    }
  }

  const store = {
    /**
     * Return a frozen copy of the current state.
     * @returns {object} the state snapshot
     */
    get() {
      return copyState(state);
    },
    /**
     * Return the current route.
     * @returns {object} the route
     */
    route() {
      return { ...route };
    },
    /**
     * Merge a patch into the state and rewrite the hash.
     * @param {object} patch partial state
     * @returns {void}
     */
    set(patch) {
      const changed = [];
      const next = { ...state };
      for (const key of Object.keys(patch)) {
        if (!FIELDS.some((entry) => entry.key === key)) {
          throw new TypeError(`unknown state key ${key}`);
        }
        const value = coerce(key, patch[key]);
        if (!sameValue(next[key], value)) {
          next[key] = value;
          changed.push(key);
        }
      }
      if (changed.length === 0) {
        return;
      }
      state = next;
      writeReplace();
      notify(copyState(state), changed);
    },
    /**
     * Subscribe to state changes.
     * @param {(state: object, changed: string[]) => void} fn listener
     * @returns {() => void} unsubscribe function
     */
    subscribe(fn) {
      listeners.add(fn);
      return () => {
        listeners.delete(fn);
      };
    },
    /**
     * Navigate to a route, optionally replacing the state.
     * @param {object} nextRoute the new route
     * @param {{state?: object|null|string, replace?: boolean}} [options] state handling
     * @returns {void}
     */
    navigate(nextRoute, { state: nextState = null, replace = false } = {}) {
      let next = { ...state };
      let changed = [];
      if (nextState === "reset") {
        const fresh = defaultState();
        changed = FIELDS.map((entry) => entry.key).filter(
          (key) => !sameValue(next[key], fresh[key]),
        );
        next = fresh;
      } else if (nextState !== null) {
        for (const key of Object.keys(nextState)) {
          if (!FIELDS.some((entry) => entry.key === key)) {
            throw new TypeError(`unknown state key ${key}`);
          }
          const value = coerce(key, nextState[key]);
          if (!sameValue(next[key], value)) {
            next[key] = value;
            changed.push(key);
          }
        }
      }
      const routeChanged = !sameRoute(route, nextRoute);
      if (!routeChanged && changed.length === 0) {
        return;
      }
      route = { ...nextRoute };
      state = next;
      const hash = formatHash(route, state);
      if (replace && hist && typeof hist.replaceState === "function") {
        hist.replaceState(null, "", hash);
      } else if (loc) {
        loc.hash = hash;
      }
      notify(copyState(state), changed);
      if (routeChanged && typeof onRouteChange === "function") {
        onRouteChange(store.route());
      }
    },
    /**
     * Re-read the route and state from the location hash.
     * @returns {{route: object, state: object, version: number}} parsed result
     */
    syncFromLocation() {
      const parsed = parseHash(loc ? loc.hash : "");
      const routeChanged = !sameRoute(route, parsed.route);
      const changed = FIELDS.map((entry) => entry.key).filter(
        (key) => !sameValue(state[key], parsed.state[key]),
      );
      route = { ...parsed.route };
      state = { ...parsed.state };
      if (changed.length > 0) {
        notify(copyState(state), changed);
      }
      if (routeChanged && typeof onRouteChange === "function") {
        onRouteChange(store.route());
      }
      return parsed;
    },
    /**
     * Listen to hash changes on a window.
     * @param {object} [win] window (defaults to the global one)
     * @returns {object} the store
     */
    bindWindow(win = typeof window !== "undefined" ? window : undefined) {
      if (win) {
        win.addEventListener("hashchange", () => store.syncFromLocation());
      }
      return store;
    },
  };
  return store;
}

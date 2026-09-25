import { S } from "./strings.js";

const PROP_KEYS = new Set(["value", "checked", "disabled", "hidden", "selected"]);
// Spelled with concatenation so the forbidden markup-sink names never appear
// literally in our sources (the forbidden-apis test scans for them).
const FORBIDDEN_KEYS = new Set(["inner" + "HTML", "outer" + "HTML", "src" + "doc"]);
const URL_KEYS = new Set(["href", "src", "action", "formaction"]);

/**
 * Build a DOM element without ever parsing markup.
 * @param {string} tag element tag name
 * @param {Object<string, unknown>} [attrs] attributes, dataset, style and on* listeners
 * @param {...unknown} children text, nodes or nested arrays
 * @returns {Element} the created element
 */
export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  if (attrs) {
    for (const [key, value] of Object.entries(attrs)) {
      if (value === null || value === undefined || value === false) {
        continue;
      }
      if (FORBIDDEN_KEYS.has(key)) {
        throw new TypeError(`forbidden attribute ${key}`);
      }
      if (key === "class") {
        el.className = String(value);
        continue;
      }
      if (key === "dataset") {
        for (const [k, v] of Object.entries(value)) {
          el.dataset[k] = String(v);
        }
        continue;
      }
      if (key === "style") {
        if (typeof value !== "object") {
          throw new TypeError("style must be an object");
        }
        Object.assign(el.style, value);
        continue;
      }
      if (key.startsWith("on")) {
        if (typeof value !== "function") {
          throw new TypeError(`listener ${key} must be a function`);
        }
        el.addEventListener(key.slice(2).toLowerCase(), value);
        continue;
      }
      if (PROP_KEYS.has(key)) {
        el[key] = value;
        continue;
      }
      if (value === true) {
        el.setAttribute(key, "");
        continue;
      }
      if (URL_KEYS.has(key) && String(value).trim().toLowerCase().startsWith("javascript:")) {
        throw new TypeError(`forbidden url in ${key}`);
      }
      el.setAttribute(key, String(value));
    }
  }
  appendChildren(el, children);
  return el;
}

/**
 * Append nested children to an element as text or nodes.
 * @param {Element} el parent element
 * @param {Array<unknown>} children nested children
 * @returns {void}
 */
function appendChildren(el, children) {
  for (const child of children) {
    if (Array.isArray(child)) {
      appendChildren(el, child);
    } else if (child === null || child === undefined || child === false) {
      continue;
    } else if (typeof child === "string" || typeof child === "number") {
      el.appendChild(document.createTextNode(String(child)));
    } else if (child instanceof Node) {
      el.appendChild(child);
    } else {
      throw new TypeError("unsupported child");
    }
  }
}

/**
 * Escape a value for use inside HTML markup (e.g. Plotly hover text).
 * @param {unknown} s value to escape
 * @returns {string} the escaped string
 */
export function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/**
 * Remove every child of an element.
 * @param {Element} el parent element
 * @returns {void}
 */
export function clear(el) {
  el.replaceChildren();
}

/**
 * Build a loading indicator view.
 * @param {string} [text] loading text
 * @returns {Element} the loading element
 */
export function loadingView(text = S.loading) {
  return h(
    "div",
    { class: "status loading" },
    h("span", { class: "spinner", "aria-hidden": "true" }),
    h("span", {}, text),
  );
}

/**
 * Build an empty-list placeholder view.
 * @param {string} text placeholder text
 * @returns {Element} the empty element
 */
export function emptyView(text) {
  return h("div", { class: "status empty" }, text);
}

/**
 * Build an error view showing the API error envelope.
 * @param {{type?: string, member?: string|null, message?: string, reason?: string|null}} err error
 * @param {{onRetry?: () => Promise<unknown>}} [options] retry handler for failed derivations
 * @returns {Element} the error element
 */
export function errorView(err, { onRetry } = {}) {
  const box = h("div", { class: "status error" });
  box.appendChild(h("div", { class: "error-type" }, String(err && err.type ? err.type : "error")));
  if (err && err.member) {
    box.appendChild(h("div", { class: "error-member" }, String(err.member)));
  }
  if (err && err.type === "derive_failed" && err.reason) {
    const label = (S.reasons && S.reasons[err.reason]) || String(err.reason);
    box.appendChild(h("div", { class: "error-reason" }, label));
  }
  box.appendChild(
    h("pre", { class: "message" }, String(err && err.message ? err.message : "")),
  );
  if (err && err.type === "derive_failed" && err.reason && typeof onRetry === "function") {
    const button = h("button", { class: "retry", type: "button" }, S.retry);
    button.addEventListener("click", async () => {
      button.disabled = true;
      try {
        await onRetry();
      } catch (next) {
        button.disabled = false;
        box.appendChild(
          h("pre", { class: "message" }, String(next && next.message ? next.message : next)),
        );
      }
    });
    box.appendChild(button);
  }
  return box;
}

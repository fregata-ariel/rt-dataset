/**
 * Address a vendored third-party file relative to this module.
 * @param {string} relpath path below static/vendor/
 * @returns {string} absolute URL of the vendored file
 */
export function vendorUrl(relpath) {
  return new URL(`../vendor/${relpath}`, import.meta.url).href;
}

let threePromise = null;
let plotlyPromise = null;

/**
 * Load the vendored three.js module and its orbit controls.
 * @returns {Promise<{THREE: object, OrbitControls: object}>} the three.js namespace and controls
 */
export async function loadThree() {
  if (!threePromise) {
    threePromise = (async () => {
      const THREE = await import(vendorUrl("three/three.module.js"));
      const { OrbitControls } = await import(vendorUrl("three/OrbitControls.js"));
      return { THREE, OrbitControls };
    })();
  }
  return threePromise;
}

/**
 * Load the vendored Plotly bundle exactly once.
 * @returns {Promise<object>} window.Plotly once the script loaded
 */
export function loadPlotly() {
  if (!plotlyPromise) {
    plotlyPromise = new Promise((resolve, reject) => {
      if (window.Plotly) {
        resolve(window.Plotly);
        return;
      }
      const script = document.createElement("script");
      script.src = vendorUrl("plotly/plotly-strict.min.js");
      script.addEventListener("load", () => resolve(window.Plotly));
      script.addEventListener("error", () => reject(new Error("plotly failed to load")));
      document.head.appendChild(script);
    });
  }
  return plotlyPromise;
}

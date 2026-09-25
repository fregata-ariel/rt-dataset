import { S } from "./strings.js";

/**
 * Format a byte count with binary units.
 * @param {number} n byte count
 * @returns {string} like "512 B" or "1.2 MiB"
 */
export function formatBytes(n) {
  if (n < 1024) {
    return `${n} B`;
  }
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let value = n / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(1)} ${units[unit]}`;
}

/**
 * Format a duration in seconds as nanoseconds.
 * @param {number|null} s seconds (or null when unknown)
 * @returns {string} like "12.5 ns" or "—"
 */
export function formatSecondsAsNs(s) {
  if (s === null || s === undefined) {
    return "—";
  }
  return `${formatNumber(s * 1e9)} ns`;
}

/**
 * Format a frequency in hertz with an SI prefix.
 * @param {number|null} hz frequency in hertz (or null when unknown)
 * @returns {string} like "3.5 GHz" or "—"
 */
export function formatHz(hz) {
  if (hz === null || hz === undefined) {
    return "—";
  }
  if (hz >= 1e9) {
    return `${formatNumber(hz / 1e9)} GHz`;
  }
  if (hz >= 1e6) {
    return `${formatNumber(hz / 1e6)} MHz`;
  }
  if (hz >= 1e3) {
    return `${formatNumber(hz / 1e3)} kHz`;
  }
  return `${formatNumber(hz)} Hz`;
}

/**
 * Format a number with significant digits, trimming trailing zeros.
 * @param {number|null|undefined} x value to format
 * @param {number} [digits] significant digits
 * @returns {string} the formatted number or "—"
 */
export function formatNumber(x, digits = 4) {
  if (x === null || x === undefined) {
    return "—";
  }
  if (!Number.isFinite(x)) {
    return String(x);
  }
  return String(Number(x.toPrecision(digits)));
}

/**
 * Format an energy value in exponential notation.
 * @param {number|null} x energy value (or null when unknown)
 * @returns {string} like "1.234e+5" or "—"
 */
export function formatEnergy(x) {
  if (x === null || x === undefined) {
    return "—";
  }
  return Number(x).toExponential(3);
}

/**
 * Format a 3D vector with fixed decimals.
 * @param {Array<number>|null} arr [x, y, z] (or null when unknown)
 * @param {number} [digits] decimals per component
 * @returns {string} like "[1.000, 2.000, 3.000]" or "—"
 */
export function formatVector(arr, digits = 3) {
  if (arr === null || arr === undefined) {
    return "—";
  }
  return `[${Array.from(arr, (v) => Number(v).toFixed(digits)).join(", ")}]`;
}

/**
 * Format a boolean with the shared yes/no strings.
 * @param {boolean|null} b value to format
 * @returns {string} S.yes, S.no or "—"
 */
export function formatBool(b) {
  if (b === true) {
    return S.yes;
  }
  if (b === false) {
    return S.no;
  }
  return "—";
}

/**
 * Shorten a digest for display.
 * @param {string} d full digest
 * @param {number} [n] characters to keep
 * @returns {string} the first n characters
 */
export function shortDigest(d, n = 12) {
  return String(d).slice(0, n);
}

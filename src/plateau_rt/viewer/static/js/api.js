import { parseNpy } from "./npy.js";

/**
 * Error thrown for failed API calls, network issues and polling aborts.
 */
export class ApiError extends Error {
  /**
   * Build an API error from its fields.
   * @param {{type: string, member?: string|null, message: string, status?: number, reason?: string|null, jobId?: string|null}} fields error fields
   */
  constructor({ type, member = null, message, status = 0, reason = null, jobId = null }) {
    super(message);
    this.name = "ApiError";
    this.type = type;
    this.member = member;
    this.message = message;
    this.status = status;
    this.reason = reason;
    this.jobId = jobId;
  }
}

/**
 * Read a response body as JSON when it has one.
 * @param {Response} response fetch response
 * @returns {Promise<unknown>} parsed body or null
 */
async function readJsonBody(response) {
  const text = await response.text();
  if (!text) {
    return null;
  }
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

/**
 * Send one API request and parse its JSON body.
 * @param {string} method HTTP method
 * @param {string} url request URL
 * @param {{body?: unknown, headers?: Object<string, string>, signal?: AbortSignal}} [options] body, headers and abort signal
 * @returns {Promise<{status: number, body: unknown}>} status and parsed body
 */
export async function request(method, url, { body, headers = {}, signal } = {}) {
  const init = { method, headers: { ...headers }, signal };
  if (method !== "GET" && method !== "HEAD") {
    init.headers["X-Viewer-Request"] = "1";
  }
  if (body !== undefined) {
    init.body = body;
  }
  let response;
  try {
    response = await fetch(url, init);
  } catch (err) {
    if (err && err.name === "AbortError") {
      throw new ApiError({ type: "aborted", message: String((err && err.message) || "aborted") });
    }
    throw new ApiError({ type: "network", message: String((err && err.message) || "network") });
  }
  const parsed = await readJsonBody(response);
  if (!response.ok) {
    if (parsed && typeof parsed === "object" && parsed.error && typeof parsed.error === "object") {
      throw new ApiError({
        type: parsed.error.type || "http",
        member: parsed.error.member ?? null,
        message: parsed.error.message || `HTTP ${response.status}`,
        status: response.status,
      });
    }
    throw new ApiError({
      type: "http",
      message: `HTTP ${response.status}`,
      status: response.status,
    });
  }
  return { status: response.status, body: parsed };
}

/**
 * GET one JSON body.
 * @param {string} url request URL
 * @param {{signal?: AbortSignal}} [opts] abort signal
 * @returns {Promise<unknown>} parsed body
 */
export async function getJson(url, opts) {
  const { body } = await request("GET", url, opts);
  return body;
}

/**
 * Build the URL of one bundle.
 * @param {string} digest bundle digest
 * @returns {string} bundle URL
 */
export function bundleUrl(digest) {
  return `/api/bundles/${encodeURIComponent(digest)}`;
}

/**
 * Build the derivation URL for one member, deriver and parameter set.
 * @param {string} digest bundle digest
 * @param {string} member member id
 * @param {string} deriver deriver name
 * @param {Object<string, unknown>} [params] query parameters
 * @returns {string} derivation URL
 */
export function derivedUrl(digest, member, deriver, params = {}) {
  const base = `/api/bundles/${encodeURIComponent(digest)}/members/${encodeURIComponent(
    member,
  )}/derived/${encodeURIComponent(deriver)}`;
  const keys = Object.keys(params).sort();
  if (keys.length === 0) {
    return base;
  }
  const query = new URLSearchParams();
  for (const key of keys) {
    query.append(key, String(params[key]));
  }
  return `${base}?${query.toString()}`;
}

/**
 * Build the retry URL for one failed derivation.
 * @param {string} digest bundle digest
 * @param {string} member member id
 * @param {string} deriver deriver name
 * @param {Object<string, unknown>} [params] query parameters
 * @returns {string} retry URL
 */
export function retryUrl(digest, member, deriver, params = {}) {
  return `${derivedUrl(digest, member, deriver, params)}/retry`;
}

/**
 * List every stored bundle.
 * @param {{signal?: AbortSignal}} [opts] abort signal
 * @returns {Promise<unknown>} the bundles body
 */
export async function listBundles(opts) {
  return getJson("/api/bundles", opts);
}

/**
 * Fetch one bundle's detail body.
 * @param {string} digest bundle digest
 * @param {{signal?: AbortSignal}} [opts] abort signal
 * @returns {Promise<unknown>} the bundle body
 */
export async function getBundle(digest, opts) {
  return getJson(bundleUrl(digest), opts);
}

/**
 * Fetch one bundle's derivation status.
 * @param {string} digest bundle digest
 * @param {{signal?: AbortSignal}} [opts] abort signal
 * @returns {Promise<unknown>} the status body
 */
export async function bundleStatus(digest, opts) {
  return getJson(`${bundleUrl(digest)}/status`, opts);
}

/**
 * Delete one bundle after confirming its digest.
 * @param {string} digest bundle digest
 * @returns {Promise<unknown>} the deletion body
 */
export async function deleteBundle(digest) {
  const { body } = await request("DELETE", bundleUrl(digest), {
    body: JSON.stringify({ confirm: digest }),
    headers: { "Content-Type": "application/json" },
  });
  return body;
}

/**
 * Upload an archive file with progress reporting.
 * @param {File} file archive file to upload
 * @param {{name?: string, onProgress?: (progress: object) => void, signal?: AbortSignal}} [options] name override, progress callback and abort signal
 * @returns {Promise<unknown>} the parsed upload response
 */
export function uploadBundle(file, { name = file.name, onProgress, signal } = {}) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    let done = false;
    const fail = (err) => {
      if (!done) {
        done = true;
        reject(err);
      }
    };
    const succeed = (value) => {
      if (!done) {
        done = true;
        resolve(value);
      }
    };
    xhr.open("PUT", `/api/bundles/upload?name=${encodeURIComponent(String(name).slice(0, 200))}`);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.setRequestHeader("X-Viewer-Request", "1");
    if (signal) {
      if (signal.aborted) {
        fail(new ApiError({ type: "aborted", message: "aborted" }));
        return;
      }
      signal.addEventListener("abort", () => xhr.abort(), { once: true });
    }
    xhr.upload.onprogress = (event) => {
      if (onProgress) {
        onProgress({ loaded: event.loaded, total: event.total, phase: "uploading" });
      }
    };
    xhr.upload.onload = (event) => {
      if (onProgress) {
        onProgress({ loaded: event.total, total: event.total, phase: "validating" });
      }
    };
    xhr.onload = () => {
      let parsed = null;
      try {
        parsed = xhr.responseText ? JSON.parse(xhr.responseText) : null;
      } catch {
        parsed = null;
      }
      if (xhr.status === 200 || xhr.status === 201) {
        succeed(parsed);
      } else if (parsed && typeof parsed === "object" && parsed.error) {
        fail(
          new ApiError({
            type: parsed.error.type || "http",
            member: parsed.error.member ?? null,
            message: parsed.error.message || `HTTP ${xhr.status}`,
            status: xhr.status,
          }),
        );
      } else {
        fail(new ApiError({ type: "http", message: `HTTP ${xhr.status}`, status: xhr.status }));
      }
    };
    xhr.onerror = () => {
      fail(new ApiError({ type: "network", message: "network" }));
    };
    xhr.onabort = () => {
      fail(new ApiError({ type: "aborted", message: "aborted" }));
    };
    xhr.send(file);
  });
}

/**
 * Compute the next derivation polling delay.
 * @param {number} previousMs previous delay in milliseconds
 * @returns {number} next delay, 0.5 s growing to a 5 s cap
 */
export function nextPollDelay(previousMs) {
  if (!previousMs) {
    return 500;
  }
  return Math.min(5000, Math.round(previousMs * 1.5));
}

/**
 * Sleep, rejecting early when the signal aborts.
 * @param {number} ms milliseconds to wait
 * @param {AbortSignal} [signal] abort signal
 * @returns {Promise<void>} resolves after the delay
 */
function defaultSleep(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal && signal.aborted) {
      reject(new ApiError({ type: "aborted", message: "aborted" }));
      return;
    }
    const timer = setTimeout(() => {
      if (signal) {
        signal.removeEventListener("abort", onAbort);
      }
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(new ApiError({ type: "aborted", message: "aborted" }));
    };
    if (signal) {
      signal.addEventListener("abort", onAbort, { once: true });
    }
  });
}

/**
 * Poll a derivation job until it is done or failed.
 * @param {string} statusUrl job status URL
 * @param {{signal?: AbortSignal, onStatus?: (job: object) => void, sleep?: function}} [options] signal, status callback and sleep hook
 * @returns {Promise<object>} the finished job
 */
export async function pollJob(statusUrl, { signal, onStatus, sleep = defaultSleep } = {}) {
  let delay = 0;
  for (;;) {
    if (signal && signal.aborted) {
      throw new ApiError({ type: "aborted", message: "aborted" });
    }
    const job = await getJson(statusUrl, { signal });
    if (onStatus) {
      onStatus(job);
    }
    if (job.status === "done") {
      return job;
    }
    if (job.status === "failed") {
      throw new ApiError({
        type: "derive_failed",
        member: job.member ?? null,
        message: job.error || job.reason || "failed",
        status: 0,
        reason: job.reason ?? null,
        jobId: job.job_id ?? null,
      });
    }
    delay = nextPollDelay(delay);
    await sleep(delay, signal);
  }
}

/**
 * Fetch a derivation, polling when the backend queues a job.
 * @param {string} digest bundle digest
 * @param {string} member member id
 * @param {string} deriver deriver name
 * @param {Object<string, unknown>} [params] query parameters
 * @param {{signal?: AbortSignal, onStatus?: (job: object) => void}} [opts] signal and status callback
 * @returns {Promise<object>} the ready derived payload
 */
export async function derive(digest, member, deriver, params = {}, { signal, onStatus } = {}) {
  const url = derivedUrl(digest, member, deriver, params);
  for (let round = 0; round < 3; round += 1) {
    const { status, body } = await request("GET", url, { signal });
    if (status !== 202) {
      return body;
    }
    const job = await pollJob(body.status_url, { signal, onStatus });
    if (job.result && typeof job.result === "object") {
      return job.result;
    }
  }
  throw new ApiError({ type: "http", message: "derivation did not become ready" });
}

/**
 * Retry a failed derivation, then poll like derive does.
 * @param {string} digest bundle digest
 * @param {string} member member id
 * @param {string} deriver deriver name
 * @param {Object<string, unknown>} [params] query parameters
 * @param {{signal?: AbortSignal, onStatus?: (job: object) => void}} [opts] signal and status callback
 * @returns {Promise<object>} the ready derived payload
 */
export async function retryDerive(digest, member, deriver, params = {}, opts = {}) {
  const { signal, onStatus } = opts;
  const url = retryUrl(digest, member, deriver, params);
  try {
    const { body } = await request("POST", url, { signal });
    const job = await pollJob(body.status_url, { signal, onStatus });
    if (job.result && typeof job.result === "object") {
      return job.result;
    }
    return derive(digest, member, deriver, params, opts);
  } catch (err) {
    if (err instanceof ApiError && err.type === "conflict") {
      return derive(digest, member, deriver, params, opts);
    }
    throw err;
  }
}

/**
 * Find the URL of one file inside a derived payload.
 * @param {object} payload derived payload with a files array
 * @param {string} name file name to find
 * @returns {string} the file URL
 */
export function fileUrl(payload, name) {
  for (const entry of payload.files || []) {
    if (entry.name === name) {
      return entry.url;
    }
  }
  throw new ApiError({ type: "not_found", message: `missing derived file ${name}` });
}

/**
 * Fetch one JSON file from a derived payload.
 * @param {object} payload derived payload with a files array
 * @param {string} name file name to fetch
 * @param {{signal?: AbortSignal}} [opts] abort signal
 * @returns {Promise<unknown>} the parsed JSON
 */
export async function fetchDerivedJson(payload, name, opts) {
  return getJson(fileUrl(payload, name), opts);
}

/**
 * Fetch one .npy file from a derived payload.
 * @param {object} payload derived payload with a files array
 * @param {string} name file name to fetch
 * @param {{signal?: AbortSignal}} [opts] abort signal
 * @returns {Promise<{shape: number[], dtype: string, data: object}>} the parsed array
 */
export async function fetchDerivedNpy(payload, name, { signal } = {}) {
  let response;
  try {
    response = await fetch(fileUrl(payload, name), { signal });
  } catch (err) {
    const type = err && err.name === "AbortError" ? "aborted" : "network";
    throw new ApiError({ type, message: String((err && err.message) || type) });
  }
  if (!response.ok) {
    throw new ApiError({ type: "http", message: `HTTP ${response.status}`, status: response.status });
  }
  return parseNpy(await response.arrayBuffer());
}

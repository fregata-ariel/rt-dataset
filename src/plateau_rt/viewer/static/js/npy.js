const DESCR_RE = /'descr'\s*:\s*'([^']*)'/;
const ORDER_RE = /'fortran_order'\s*:\s*(True|False)/;
const SHAPE_RE = /'shape'\s*:\s*\(([^)]*)\)/;

const ITEMSIZES = {
  "<f2": 2,
  "<f4": 4,
  "<i4": 4,
  "<u4": 4,
  "|u1": 1,
  "|b1": 1,
};

let f16Table = null;

/**
 * Decode one 16-bit half-precision value to 32-bit float.
 * @param {number} bits the 16-bit pattern as an integer
 * @returns {number} the decoded float
 */
export function float16ToFloat32(bits) {
  const sign = (bits >> 15) & 1;
  const exp = (bits >> 10) & 31;
  const frac = bits & 1023;
  let value;
  if (exp === 0) {
    value = frac === 0 ? 0 : frac * 2 ** -24;
  } else if (exp === 31) {
    value = frac === 0 ? Infinity : NaN;
  } else {
    value = (1 + frac / 1024) * 2 ** (exp - 15);
  }
  return sign ? -value : value;
}

/**
 * Return the lazily built 65536-entry half-to-float lookup table.
 * @returns {Float32Array} the lookup table
 */
function lookupTable() {
  if (f16Table === null) {
    f16Table = new Float32Array(65536);
    for (let i = 0; i < 65536; i += 1) {
      f16Table[i] = float16ToFloat32(i);
    }
  }
  return f16Table;
}

/**
 * Read an ASCII header slice from raw bytes.
 * @param {Uint8Array} bytes the whole file bytes
 * @param {number} start header start offset
 * @param {number} length header length
 * @returns {string} the latin1-decoded header
 */
function headerText(bytes, start, length) {
  let text = "";
  for (let i = 0; i < length; i += 1) {
    text += String.fromCharCode(bytes[start + i]);
  }
  return text;
}

/**
 * Parse the shape tuple body of a .npy header.
 * @param {string} body text between the shape parentheses
 * @returns {number[]} the parsed shape
 */
function parseShape(body) {
  const trimmed = body.trim();
  if (trimmed === "") {
    return [];
  }
  return trimmed.split(",").reduce((shape, part) => {
    const token = part.trim();
    if (token === "") {
      return shape;
    }
    if (!/^[0-9]+$/.test(token)) {
      throw new Error(`invalid .npy shape ${body}`);
    }
    shape.push(Number(token));
    return shape;
  }, []);
}

/**
 * Parse a .npy buffer into shape, dtype and a typed array.
 * @param {ArrayBuffer|Uint8Array} buffer the .npy file bytes
 * @returns {{shape: number[], dtype: string, data: TypedArray}} parsed array
 */
export function parseNpy(buffer) {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  if (bytes.length < 10) {
    throw new Error("invalid .npy file");
  }
  if (
    bytes[0] !== 0x93 ||
    bytes[1] !== 0x4e ||
    bytes[2] !== 0x55 ||
    bytes[3] !== 0x4d ||
    bytes[4] !== 0x50 ||
    bytes[5] !== 0x59
  ) {
    throw new Error("invalid .npy magic");
  }
  const major = bytes[6];
  const minor = bytes[7];
  let headerLength;
  let dataOffset;
  if (major === 1 && minor === 0) {
    headerLength = bytes[8] | (bytes[9] << 8);
    dataOffset = 10 + headerLength;
  } else if (major === 2 && minor === 0) {
    headerLength =
      bytes[8] | (bytes[9] << 8) | (bytes[10] << 16) | (bytes[11] * 2 ** 24);
    dataOffset = 12 + headerLength;
  } else {
    throw new Error(`unsupported .npy version ${major}.${minor}`);
  }
  if (bytes.length < dataOffset) {
    throw new Error("invalid .npy header length");
  }
  const header = headerText(bytes, dataOffset - headerLength, headerLength);
  const descr = header.match(DESCR_RE);
  const order = header.match(ORDER_RE);
  const shapeMatch = header.match(SHAPE_RE);
  if (!descr || !order || !shapeMatch) {
    throw new Error("invalid .npy header");
  }
  if (!(descr[1] in ITEMSIZES)) {
    throw new Error(`unsupported .npy dtype ${descr[1]}`);
  }
  if (order[1] !== "False") {
    throw new Error("unsupported .npy fortran order");
  }
  const shape = parseShape(shapeMatch[1]);
  const itemsize = ITEMSIZES[descr[1]];
  const count = shape.reduce((acc, dim) => acc * dim, 1);
  if (bytes.length - dataOffset !== count * itemsize) {
    throw new Error("invalid .npy data length");
  }
  if (new Uint8Array(new Uint16Array([1]).buffer)[0] !== 1) {
    throw new Error("big-endian platforms are not supported");
  }
  const raw = bytes.slice(dataOffset, dataOffset + count * itemsize);
  const code = descr[1];
  if (code === "<f4") {
    return { shape, dtype: "float32", data: new Float32Array(raw.buffer, raw.byteOffset, count) };
  }
  if (code === "<f2") {
    const halves = new Uint16Array(raw.buffer, raw.byteOffset, count);
    const table = lookupTable();
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) {
      out[i] = table[halves[i]];
    }
    return { shape, dtype: "float16", data: out };
  }
  if (code === "<i4") {
    return { shape, dtype: "int32", data: new Int32Array(raw.buffer, raw.byteOffset, count) };
  }
  if (code === "<u4") {
    return { shape, dtype: "uint32", data: new Uint32Array(raw.buffer, raw.byteOffset, count) };
  }
  if (code === "|u1" || code === "|b1") {
    return { shape, dtype: code === "|u1" ? "uint8" : "bool", data: raw.slice() };
  }
  throw new Error(`unsupported .npy dtype ${code}`);
}

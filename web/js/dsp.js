// Small numeric toolbox used by features.js / model.js. Pure JS (no WebAudio),
// so it runs under node for the parity tests.
//
//   fft512(re)            real input of length N (power of 2) → power spectrum bins 0..N/2
//   butter2(cutoffHz, rateHz)   scipy.signal.butter(2, cutoff / (rate/2)) coefficients
//   filtfilt(b, a, x, padlen)   scipy.signal.filtfilt(method='pad', padtype='odd')
//   float16ToFloat32(u16)       IEEE half → single
//   seededRng(seed)             deterministic uniform generator (sfc32)

// ---------------------------------------------------------------------------
// FFT: iterative radix-2, complex, in place; power spectrum helper
// ---------------------------------------------------------------------------
const _fftCache = new Map();

function fftTables(n) {
  let t = _fftCache.get(n);
  if (t) return t;
  const bits = Math.log2(n);
  if (!Number.isInteger(bits)) throw new Error(`fft size ${n} is not a power of two`);
  const rev = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    let r = 0;
    for (let b = 0; b < bits; b++) r = (r << 1) | ((i >> b) & 1);
    rev[i] = r;
  }
  const cos = new Float64Array(n / 2);
  const sin = new Float64Array(n / 2);
  for (let i = 0; i < n / 2; i++) {
    cos[i] = Math.cos((-2 * Math.PI * i) / n);
    sin[i] = Math.sin((-2 * Math.PI * i) / n);
  }
  t = { rev, cos, sin };
  _fftCache.set(n, t);
  return t;
}

/**
 * Power spectrum |FFT(x)|² for bins 0..n/2 of a real signal zero-padded to `n`.
 * @param {Float64Array|Float32Array|number[]} x  length ≤ n
 * @param {number} n  FFT size (power of two)
 * @param {Float64Array} [out]  length n/2+1
 */
export function powerSpectrum(x, n, out = null) {
  const { rev, cos, sin } = fftTables(n);
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  for (let i = 0; i < x.length && i < n; i++) re[rev[i]] = x[i];
  // note: bit-reversal applied on load; zero padding stays zero
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1;
    const step = n / size;
    for (let start = 0; start < n; start += size) {
      for (let k = 0; k < half; k++) {
        const wr = cos[k * step], wi = sin[k * step];
        const i = start + k, j = i + half;
        const tr = re[j] * wr - im[j] * wi;
        const ti = re[j] * wi + im[j] * wr;
        re[j] = re[i] - tr; im[j] = im[i] - ti;
        re[i] += tr; im[i] += ti;
      }
    }
  }
  const nb = n / 2 + 1;
  out = out || new Float64Array(nb);
  for (let k = 0; k < nb; k++) out[k] = re[k] * re[k] + im[k] * im[k];
  return out;
}

// ---------------------------------------------------------------------------
// Butterworth order 2 (bilinear, prewarped) == scipy.signal.butter(2, wn)
// ---------------------------------------------------------------------------
export function butter2(cutoffHz, rateHz) {
  const wn = Math.min(cutoffHz / (0.5 * rateHz), 0.99);
  const K = Math.tan((Math.PI * wn) / 2);
  const K2 = K * K;
  const norm = 1 + Math.SQRT2 * K + K2;
  const b0 = K2 / norm;
  return { b: [b0, 2 * b0, b0], a: [1, (2 * (K2 - 1)) / norm, (1 - Math.SQRT2 * K + K2) / norm] };
}

/** scipy.signal.lfilter_zi for a 2nd-order section. */
function lfilterZi(b, a) {
  // solve (I - companion(a).T) zi = b[1:] - a[1:]*b[0]
  const m00 = 1 + a[1], m01 = -1, m10 = a[2], m11 = 1;
  const r0 = b[1] - a[1] * b[0], r1 = b[2] - a[2] * b[0];
  const det = m00 * m11 - m01 * m10;
  return [(r0 * m11 - m01 * r1) / det, (m00 * r1 - m10 * r0) / det];
}

/** Direct form II transposed, order 2, with initial state (scipy.signal.lfilter). */
function lfilter(b, a, x, zi) {
  const y = new Float64Array(x.length);
  let z0 = zi[0], z1 = zi[1];
  for (let n = 0; n < x.length; n++) {
    const xn = x[n];
    const yn = b[0] * xn + z0;
    z0 = b[1] * xn - a[1] * yn + z1;
    z1 = b[2] * xn - a[2] * yn;
    y[n] = yn;
  }
  return y;
}

/**
 * Zero-phase filtering, identical to scipy.signal.filtfilt(b, a, x, padlen=padlen)
 * (method 'pad', padtype 'odd'). Returns a Float64Array of x.length.
 */
export function filtfilt(b, a, x, padlen) {
  const n = x.length;
  if (n === 0) return new Float64Array(0);
  padlen = Math.min(padlen, n - 1);
  if (padlen < 0) padlen = 0;
  const ext = new Float64Array(n + 2 * padlen);
  for (let i = 0; i < padlen; i++) ext[i] = 2 * x[0] - x[padlen - i];
  for (let i = 0; i < n; i++) ext[padlen + i] = x[i];
  for (let i = 0; i < padlen; i++) ext[padlen + n + i] = 2 * x[n - 1] - x[n - 2 - i];
  const zi = lfilterZi(b, a);
  let y = lfilter(b, a, ext, [zi[0] * ext[0], zi[1] * ext[0]]);
  y.reverse();
  y = lfilter(b, a, y, [zi[0] * y[0], zi[1] * y[0]]);
  y.reverse();
  return y.slice(padlen, padlen + n);
}

/**
 * Zero-phase Butterworth per column of a [T, C] row-major array; mirrors
 * animacy.model.infer.smooth_motion (untouched when T < 12 or no cutoff).
 */
export function smoothColumns(x, T, C, rateHz, cutoffHz) {
  if (!cutoffHz || T < 12) return Float32Array.from(x);
  const { b, a } = butter2(cutoffHz, rateHz);
  const padlen = Math.min(9, T - 1);
  const out = new Float32Array(T * C);
  const col = new Float64Array(T);
  for (let c = 0; c < C; c++) {
    for (let t = 0; t < T; t++) col[t] = x[t * C + c];
    const y = filtfilt(b, a, col, padlen);
    for (let t = 0; t < T; t++) out[t * C + c] = y[t];
  }
  return out;
}

// ---------------------------------------------------------------------------
// misc
// ---------------------------------------------------------------------------
/** IEEE 754 half-precision → Float32Array. */
export function float16ToFloat32(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i];
    const s = (h & 0x8000) ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const f = h & 0x3ff;
    let v;
    if (e === 0) v = s * Math.pow(2, -14) * (f / 1024);
    else if (e === 31) v = f ? NaN : s * Infinity;
    else v = s * Math.pow(2, e - 15) * (1 + f / 1024);
    out[i] = v;
  }
  return out;
}

/** Deterministic uniform [0,1) generator (sfc32 seeded from a 32-bit integer). */
export function seededRng(seed) {
  let a = 0x9e3779b9 ^ (seed >>> 0), b = 0x243f6a88, c = 0xb7e15162, d = (seed * 0x85ebca6b) >>> 0 || 1;
  const next = () => {
    a >>>= 0; b >>>= 0; c >>>= 0; d >>>= 0;
    let t = (a + b) | 0;
    a = b ^ (b >>> 9);
    b = (c + (c << 3)) | 0;
    c = (c << 21) | (c >>> 11);
    d = (d + 1) | 0;
    t = (t + d) | 0;
    c = (c + t) | 0;
    return (t >>> 0) / 4294967296;
  };
  for (let i = 0; i < 12; i++) next();
  return next;
}

/** np.hanning(n) (symmetric). */
export function hanning(n) {
  const w = new Float64Array(n);
  if (n === 1) { w[0] = 1; return w; }
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1));
  return w;
}

/** np.convolve(x, k, mode='same'). */
export function convolveSame(x, k) {
  const n = x.length, m = k.length;
  const full = new Float64Array(n + m - 1);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) full[i + j] += x[i] * k[j];
  const start = Math.floor((m - 1) / 2);
  return full.slice(start, start + n);
}

// Audio features for the motion model — the browser side of the
// `animacy.features.v1` contract. Line-for-line port of animacy/features.py:
//
//   16 kHz mono float32 → [T, 66] float32 on the 30 Hz motion grid
//   64 log-mel bands (win 400 = 25 ms, hop 160 = 10 ms, periodic Hann, rfft 512,
//   HTK mel 50–7600 Hz, area norm off) averaged per 33.3 ms tick, + log energy
//   + delta log energy, then per-utterance mean/variance normalisation.
//
// tests/test_web_features_parity.py runs this under node against the Python
// output for the same waveform (tolerance 1e-4 after normalisation).

import { powerSpectrum } from './dsp.js';

export const SR = 16000;
export const N_FFT = 512;
export const WIN = 400;
export const HOP = 160;
export const N_MELS = 64;
export const FMIN = 50.0;
export const FMAX = 7600.0;
export const RATE_HZ = 30.0;
export const N_FEATS = N_MELS + 2;
const EPS = 1e-6;

const hzToMel = (f) => 2595.0 * Math.log10(1.0 + f / 700.0);
const melToHz = (m) => 700.0 * (Math.pow(10.0, m / 2595.0) - 1.0);

/** [n_mels][n_fft/2+1] triangular filters (HTK mel, Slaney-style area norm off). */
export function melFilterbank(sr = SR, nFft = N_FFT, nMels = N_MELS, fmin = FMIN, fmax = FMAX) {
  const nBins = nFft / 2 + 1;
  const freqs = new Float64Array(nBins);
  for (let i = 0; i < nBins; i++) freqs[i] = (i * (sr / 2)) / (nBins - 1); // np.linspace(0, sr/2, nBins)
  const mlo = hzToMel(fmin), mhi = hzToMel(fmax);
  const edges = new Float64Array(nMels + 2);
  for (let i = 0; i < nMels + 2; i++) edges[i] = melToHz(mlo + ((mhi - mlo) * i) / (nMels + 1));
  const fb = [];
  for (let i = 0; i < nMels; i++) {
    const lo = edges[i], c = edges[i + 1], hi = edges[i + 2];
    const row = new Float32Array(nBins);
    for (let k = 0; k < nBins; k++) {
      const up = (freqs[k] - lo) / Math.max(c - lo, EPS);
      const down = (hi - freqs[k]) / Math.max(hi - c, EPS);
      row[k] = Math.max(Math.min(up, down), 0.0);
    }
    fb.push(row);
  }
  return fb;
}

let _FB = null;
let _WINDOW = null;

function periodicHann(n) {
  // np.hanning(n + 1)[:-1]
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
  return w;
}

/**
 * [N][64] log-mel frames at 100 Hz (frame k centred at k*10 ms); reflect-padded like numpy.
 * @param {Float32Array} wav
 * @returns {Float32Array[]} rows
 */
export function logMel100hz(wav) {
  if (!_FB) _FB = melFilterbank();
  if (!_WINDOW) _WINDOW = periodicHann(WIN);
  const pad = WIN / 2;
  const n0 = wav.length;
  let x;
  if (n0 > pad) {
    // np.pad(x, (pad, pad), mode='reflect')
    x = new Float32Array(n0 + 2 * pad);
    for (let i = 0; i < pad; i++) x[i] = wav[pad - i];
    x.set(wav, pad);
    for (let i = 0; i < pad; i++) x[pad + n0 + i] = wav[n0 - 2 - i];
  } else {
    // np.pad(x, (pad, pad + WIN)) zeros
    x = new Float32Array(n0 + 2 * pad + WIN);
    x.set(wav, pad);
  }
  const nFrames = 1 + Math.floor((x.length - WIN) / HOP);
  const rows = [];
  if (nFrames <= 0) return rows;
  const frame = new Float64Array(WIN);
  const spec = new Float64Array(N_FFT / 2 + 1);
  for (let f = 0; f < nFrames; f++) {
    const off = f * HOP;
    for (let i = 0; i < WIN; i++) frame[i] = x[off + i] * _WINDOW[i];
    powerSpectrum(frame, N_FFT, spec);
    const row = new Float32Array(N_MELS);
    for (let m = 0; m < N_MELS; m++) {
      const fbm = _FB[m];
      let s = 0.0;
      for (let k = 0; k < spec.length; k++) s += spec[k] * fbm[k];
      row[m] = Math.log(s + EPS);
    }
    rows.push(row);
  }
  return rows;
}

/** Average 100 Hz rows into each 1/rate tick (tick i spans [i, i+1)/rate). Returns rows [nTicks][dim]. */
export function toMotionGrid(rows100, nTicks, dim, rateHz = RATE_HZ) {
  const out = [];
  for (let i = 0; i < nTicks; i++) {
    const a = Math.round((i * 100.0) / rateHz);
    const b = Math.round(((i + 1) * 100.0) / rateHz);
    const row = new Float32Array(dim);
    const lo = a, hi = b > a ? b : a + 1;
    let cnt = 0;
    for (let k = lo; k < hi && k < rows100.length; k++) {
      const r = rows100[k];
      for (let d = 0; d < dim; d++) row[d] += r[d];
      cnt++;
    }
    if (cnt > 0) for (let d = 0; d < dim; d++) row[d] /= cnt;
    else if (i > 0) row.set(out[i - 1]);
    out.push(row);
  }
  return out;
}

/** log RMS energy per 10 ms frame (no padding; indices clipped to the signal), [N][1]. */
export function energy100hz(wav) {
  const n = wav.length;
  const nFrames = n >= WIN ? Math.max(1, 1 + Math.floor((n - WIN) / HOP)) : 1;
  const rows = [];
  for (let f = 0; f < nFrames; f++) {
    let s = 0.0;
    for (let i = 0; i < WIN; i++) {
      const idx = Math.min(Math.max(f * HOP + i, 0), Math.max(n - 1, 0));
      const v = n ? wav[idx] : 0;
      s += v * v;
    }
    rows.push(Float32Array.of(Math.log(Math.sqrt(s / WIN) + EPS)));
  }
  return rows;
}

/** Per-utterance mean/variance normalisation over rows (population std + 1e-3), in place. */
export function normaliseRows(rows) {
  if (!rows.length) return rows;
  const dim = rows[0].length;
  const mu = new Float64Array(dim), sd = new Float64Array(dim);
  for (const r of rows) for (let d = 0; d < dim; d++) mu[d] += r[d];
  for (let d = 0; d < dim; d++) mu[d] /= rows.length;
  for (const r of rows) for (let d = 0; d < dim; d++) { const v = r[d] - mu[d]; sd[d] += v * v; }
  for (let d = 0; d < dim; d++) sd[d] = Math.sqrt(sd[d] / rows.length) + 1e-3;
  for (const r of rows) for (let d = 0; d < dim; d++) r[d] = (r[d] - mu[d]) / sd[d];
  return rows;
}

/**
 * The full contract: [nTicks][66] normalised features on the motion grid.
 * @param {Float32Array} wav  16 kHz mono
 * @param {number} [nTicks]   default ceil(len / sr * 30)
 * @returns {Float32Array[]}
 */
export function audioFeatures(wav, nTicks = null, rateHz = RATE_HZ) {
  if (nTicks === null || nTicks === undefined) nTicks = Math.ceil((wav.length / SR) * rateHz);
  const mel = toMotionGrid(logMel100hz(wav), nTicks, N_MELS, rateHz);
  const en = toMotionGrid(energy100hz(wav), nTicks, 1, rateHz);
  const rows = [];
  for (let i = 0; i < nTicks; i++) {
    const r = new Float32Array(N_FEATS);
    r.set(mel[i], 0);
    r[N_MELS] = en[i][0];
    r[N_MELS + 1] = i === 0 ? 0.0 : en[i][0] - en[i - 1][0]; // np.diff(prepend=en[:1])
    rows.push(r);
  }
  return normaliseRows(rows);
}

/** rows [T][66] → flat Float32Array [T*66] (row-major) for an ONNX tensor. */
export function flatten(rows, dim = N_FEATS) {
  const out = new Float32Array(rows.length * dim);
  for (let i = 0; i < rows.length; i++) out.set(rows[i], i * dim);
  return out;
}

// Node harness for tests/test_web_features_parity.py.
//   stdin:  {"wav": [...float samples at 16 kHz...], "n_ticks": int|null,
//            "filt": {"x": [...], "cutoff_hz": f, "rate_hz": f, "padlen": int} (optional)}
//   stdout: {"features": [[66 floats] x T], "filt": [...]}
import { audioFeatures } from '../js/features.js';
import { butter2, filtfilt } from '../js/dsp.js';

let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (c) => { input += c; });
process.stdin.on('end', () => {
  const job = JSON.parse(input);
  const wav = Float32Array.from(job.wav);
  const rows = audioFeatures(wav, job.n_ticks ?? null);
  const out = { features: rows.map((r) => Array.from(r)) };
  if (job.filt) {
    const { b, a } = butter2(job.filt.cutoff_hz, job.filt.rate_hz);
    out.filt = Array.from(filtfilt(b, a, Float64Array.from(job.filt.x), job.filt.padlen));
    out.coeffs = { b, a };
  }
  process.stdout.write(JSON.stringify(out));
});

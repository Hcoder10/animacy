// Node harness for tests/test_web_retarget_parity.py: run the browser
// retargeter (web/js/retarget.js) on a JSON job from stdin and print the
// joint tables it produces, so Python can diff them against
// animacy.retarget.LiveRetargeter / to_urdf_values.
//
//   stdin:  {"profile": <web/robots/x.json>, "mode": "default", "dt": 0.0333,
//            "frames": [{channel: value, ...}, ...]}
//   stdout: {"joints": [{joint: value}, ...], "urdf": [{urdf_joint: value}, ...]}
import { LiveRetargeter, toUrdfValues } from '../js/retarget.js';

let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (c) => { input += c; });
process.stdin.on('end', () => {
  const job = JSON.parse(input);
  const rt = new LiveRetargeter(job.profile, job.mode || 'default', job.default_smooth_hz ?? 6.0);
  const joints = [];
  const urdf = [];
  for (const f of job.frames) {
    const j = rt.step(f, job.dt);
    joints.push(j);
    urdf.push(toUrdfValues(j, job.profile));
  }
  process.stdout.write(JSON.stringify({ joints, urdf }));
});

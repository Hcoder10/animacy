# Retargeting: from canonical human motion to a body that looks alive

`ROBOT.md` (`docs/ROBOT_MD_SPEC.md`) declares a robot; this document is the
exact arithmetic that turns a canonical frame (`docs/CANONICAL.md`) into a
joint frame, the data behind the numbers in the two shipped mappings, and the
evaluation that scores them. There are two implementations of the per-frame
update — `animacy/retarget.py:LiveRetargeter.step` (reference) and
`web/js/retarget.js:LiveRetargeter.step` (browser) — and one offline path,
`retarget_clip`, that shares every function with the live one. Everything a
mapping can do is **data in `ROBOT.md`**; nothing is robot-specific code.

## 1. Per-frame update (`animacy.robot.v1`, mapping keys v1.1)

Per joint `j` (declared `rest`, `min`, `max`, `max_speed`) with mapping `m`
(`terms[] = (channel, gain)`, `offset`, `deadband`, bounds `lo`/`hi` = mapping
`min`/`max` if given else the joint's, `smooth_hz`, and the optional v1.1 keys
`soft_limit`, `idle{amp, hz, still}`, `spring{hz, zeta}`). Per-joint state:
`y` (last output, init `rest`), `v` (velocity, init 0), `e` (activity
envelope, init 0), `p` (previous pre-idle target, init `rest`), `t` (idle
clock, init 0). `dt` is the seconds since the previous frame. Channels that
are missing, `null` or NaN count as 0.

```
1. x  = Σ_k gain_k · channel_k
2. x  = 0                     if |x| < deadband
      = x − sign(x)·deadband  otherwise
3. u  = rest + offset + x
4. soft limit (only if soft_limit s > 0):   k = s · (hi − lo)
      u > hi − k :  u = (hi − k) + k · tanh((u − (hi − k)) / k)
      u < lo + k :  u = (lo + k) − k · tanh(((lo + k) − u) / k)
5. u  = clamp(u, lo, hi)
6. idle (only if idle):        a = |u − p| / dt ;  p = u
      e = max(a, e · exp(−dt / 0.5))
      g = clamp(1 − e / still, 0, 1)           still defaults to 10·amp·hz
      t = t + dt
      n = amp · Σ_{k=0..2} w_k · sin(2π · r_k · hz · t + φ_{j,k})
      u = clamp(u + g · n, lo, hi)
7. tracker:
      spring:  [y_new − u, v] = M(hz, zeta, dt) · [y − u, v]      (§1.2, exact)
      else:    α = 1 − exp(−2π · cutoff · dt)  (α = 1 if cutoff is 0/absent)
               y_new = y + α · (u − y)          cutoff = smooth_hz or the runtime default
8. d = y_new − y ;  clipped = |d| > max_speed·dt
   y_new = y + clamp(d, −max_speed·dt, +max_speed·dt)                ← rate limit, always here
9. if y_new < min_j or y_new > max_j:  clipped = true ;  y_new = clamp(y_new, min_j, max_j)   ← hard clamp, always last
10. v = v_spring                 if the joint has a spring and not clipped
      = (y_new − y) / dt         otherwise            ;   y = y_new
```

Step 10 keeps the spring's own velocity while it runs free (so the response
is the exact damped oscillator) and re-derives it from what was actually
output only when the rate limit or clamp engaged, so a spring pushed against
a limit does not wind up. Joints with no mapping in the mode behave as
`u = rest` with the one-pole tracker.

### 1.1 Idle sway generator (step 6)

Deterministic, no RNG: three sines around `hz` with fixed frequency ratios,
weights and per-joint phases. `j` is the joint's **0-based index in
`profile.joints`** (servo order), the same in Python and JS.

```
r = (1.0, 1.31, 0.67)          frequency multipliers
w = (0.5, 0.3, 0.2)            weights, Σ = 1 ⇒ |n| ≤ amp
φ_{j,k} = fmod(2.39996322972865332 · (3·j + k + 1), 2π)     golden angle
```

The gate `g` uses an activity envelope with instant attack and a 0.5 s
release (`e = max(a, e·exp(−dt/0.5))`), so the sway vanishes the frame the
person moves and fades back in over ~1 s of stillness. The idle clock `t` is
the joint's accumulated `dt`, so with a fixed frame rate it equals
`frame_index / rate_hz` and the sway is reproducible frame for frame.

### 1.2 Spring tracker (step 7)

`spring: {hz, zeta}` replaces the one-pole smoother with the damped
oscillator `y'' = ω²(u − y) − 2ζω·y'`, `ω = 2π·hz`. The discrete step is the
**exact zero-order-hold solution** (the 2×2 matrix exponential of the state
matrix), written in closed form per damping regime so any `hz`/`dt` is
stable and the overshoot is the analytic `exp(−πζ/√(1−ζ²))`:

```
ζ = 1 (critical):  e = exp(−ω dt); te = dt·e; tef = te·ω
                   pp = tef + e ;  pv = te ;  vp = −ω·tef ;  vv = −tef + e
ζ < 1 (under):     ωz = ω ζ ;  a = ω √(1−ζ²) ;  e = exp(−ωz dt) ;  c = cos(a dt) ;  s = sin(a dt)
                   es = e s ;  ec = e c ;  ewzs = e ωz s / a
                   pp = ec + ewzs ;  pv = es / a ;  vp = −es·a − ωz·ewzs ;  vv = ec − ewzs
ζ > 1 (over):      za = −ω ζ ;  zb = ω √(ζ²−1) ;  z1 = za − zb ;  z2 = za + zb
                   e1 = exp(z1 dt) ;  e2 = exp(z2 dt) ;  e1i = e1/(2 zb) ;  e2i = e2/(2 zb)
                   pp = e1i·z2 − z2·e2i + e2 ;  pv = −e1i + e2i
                   vp = (z1·e1i − z2·e2i + e2)·z2 ;  vv = −z1·e1i + z2·e2i
step:              p = y − u ;  y_new = p·pp + v·pv + u ;  v_new = p·vp + v·vv
```

(`animacy.retarget.spring_coefficients`, checked against
`scipy.linalg.expm` to 1e-11 in `tests/test_retarget_features.py`.) Why not
semi-implicit Euler: with explicit damping it diverges at `hz = 7.5` and
distorts ζ at 4 Hz (overshoot 0 % instead of 4.6 %). `animacy check` still
rejects `spring.hz > rate_hz/4` — above that the 30 Hz grid under-samples the
overshoot, not a stability issue.

Overshoot by ζ: 1.0 → 0 %, 0.9 → 0.15 %, 0.7 → 4.6 %, 0.6 → 9.5 %,
0.45 → 20.5 %. Low-frequency lag (group delay) = `2ζ/ω = ζ/(π·hz)` seconds:
4 Hz ζ 0.7 → 56 ms; 2 Hz ζ 1 → 159 ms.

### 1.3 Soft limit (step 4)

A `tanh` knee over the last `soft_limit` fraction of the mapping range at
each end: identity inside, slope 1 at the knee (C1), asymptotic to the bound.
The hard clamp in step 5/9 remains the guarantee. For a joint whose `rest`
sits close to one bound (lamp `wrist_pitch`: rest −62.4, `min` −85) use a
smaller fraction, or the knee eats the short side.

### 1.4 Offline path (`retarget_clip`)

Same functions, on the robot's grid: `raw_joint_targets` (steps 1–5 on the
clip) → `stretch_timeline` (widen only the frame gaps that would exceed
`max_speed`, margin 0.92) → `resample` to `rate_hz` → per joint: if `spring`,
advance the target by the spring's lag `ζ/(π·hz)` (linear interpolation, end
held) so the output stays on the audio clock; step 6 (idle) → step 7 (spring
with steps 8–10 inside the loop, **or** zero-phase 2nd-order Butterworth at
`smooth_hz` for one-pole joints) → `rate_limit` → clamp. Offline and live
therefore differ only by the zero-phase filter (one-pole joints) and the lag
advance (spring joints); `tests/test_retarget_features.py::test_offline_and_live_idle_use_the_same_generator`
checks a constant input gives the same sway on both.

### 1.5 Numeric example (reproduced by `tests/test_retarget_features.py::test_numeric_example_in_docs`)

Joint `a`: `rest 0, min −60, max 60, max_speed 150`; mapping
`{from: head_yaw, gain: 1, soft_limit: 0.2, spring: {hz: 2, zeta: 0.6}}`;
`dt = 1/30`; `head_yaw = 50` held.

```
step 3   u = 0 + 0 + 50 = 50
step 4   k = 0.2·120 = 24 ; hi − k = 36 ; u = 36 + 24·tanh(14/24) = 48.602015
step 7   M(2 Hz, 0.6, 1/30) = (pp, pv, vp, vv) = (0.926342, 0.025443, −4.017812, 0.542669)
frame 1  p = 0 − 48.602 = −48.602 ; y = −48.602·0.926342 + 0 + 48.602 = 3.57994 ; v = −48.602·(−4.017812) = 195.2738
         step 8: |Δ| = 3.58 ≤ 150/30 = 5, not clipped ; step 10: v stays 195.2738
frame 2  p = 3.580 − 48.602 = −45.022 ; y = −45.022·0.926342 + 195.274·0.025443 + 48.602 = 11.864
         step 8: Δ = 8.28 > 5 → clipped, y = 3.57994 + 5 = 8.57994 ; step 10: v = 5/(1/30) = 150.0
frame 3  spring from (8.580, 150) wants 15.26 → clipped again → 13.57994, v = 150
```

Idle example: joint index 1, `idle: {amp: 3, hz: 0.4}` (still = 12 units/s):
`φ_{1,k} = (3.316668, 5.716631, 1.833409)`; at `t = 1/30`,
`n = 3·(0.5·sin(2π·0.4/30 + 3.316668) + 0.3·sin(2π·0.524/30 + 5.716631) + 0.2·sin(2π·0.268/30 + 1.833409)) = −0.211154`;
at `t = 1.0`, `n = −0.513490`.

### 1.6 What the web port must implement (checklist)

- Profile JSON per mapping (`animacy profile export`): `terms[]`, `offset`,
  `min`, `max` (resolved), `deadband`, `smooth_hz`, plus `spring {hz, zeta} | null`,
  `idle {amp, hz, still} | null` (`still` already resolved), `soft_limit | null`.
- State per joint: `y`, `v`, `e`, `p`, `t`; `reset()` sets `y = rest`, the
  rest 0 (`p = rest`).
- Steps 1–10 in that order; the idle phase uses the joint's index in
  `profile.joints`; `spring` uses the closed-form coefficients of §1.2; the
  velocity carried into the next frame is the spring's own unless step 8 or
  9 clipped (Python: `animacy.retarget.clip_step`).
- Parity: `tests/test_web_retarget_parity.py` runs both on 240 random frames
  per robot/mode (1e-6) and on a synthetic v1.1 profile with a still-then-move
  stream (idle must engage).

## 2. Data → numbers: how the two mappings were fitted

`scripts/retarget_fit.py --robot <name> [--write]` (library:
`animacy/retarget_fit.py`). Human corpus: `data/clips/*` minus the two
tracker-broken captures `checkpoints/v1/REPORT.md` excludes
(`sd_rapper_interview`, `cbp_vlog_day2`), face-valid frames only, frames with
|head_x/y/z| ≥ 140 mm (the ±150 sanity clamp) dropped. Vendor envelope: the
robot's `native_clips` (lamp: 31 Autonomous OS recordings; Reachy: 16
Pollen emotion clips), values taken around the **library median** per joint
(the lamp's `wrist_pitch` median is `rest + 29`: the library looks further
down than `idle.csv`).

### 2.1 Envelope fit

Per joint: `mult = vendor |.|p95 / retargeted |.|p95` (retargeted = the human
corpus through the linear part of the mapping), clipped to `[0.5, 3]`, to the
mapping's headroom (`0.9 ×` distance from `rest` to each bound must hold the
scaled p99/p1), and to a **velocity cap**: the scaled velocity p95 may not
exceed 1.25 × the vendor's — human heads are brisker than authored robot
clips, and matching amplitude alone made the lamp whip (wrist_roll velocity
p95 97 °/s vs the vendor's 44). Then three fixed-point passes re-measure the
actual pipeline (soft limit, spring) and correct. Every term gain of the joint
is multiplied by `mult`, except derived terms (§2.2). Antennas are skipped
(§2.3). Changed lines are stamped `# fitted by scripts/retarget_fit.py <date>`.

### 2.2 Gaze preservation (lamp) — URDF FK

Elevation/azimuth of the head's look axis (`[0.70, 0, −0.71]` in the `head`
frame, the vector `tests/test_lamp_urdf.py` uses) from `yourdfpy` FK of
`robots/lamp/urdf/lamp.urdf`, central difference ±1° at `rest`:

| joint | ∂elevation/∂joint | ∂azimuth/∂joint |
|---|---|---|
| base_pitch | **−1.000** | 0 |
| elbow_pitch | **+1.000** | 0 |
| wrist_pitch | **−1.000** | 0 |
| base_yaw | 0 | −1.000 |
| wrist_roll | 0 | −1.008 |

The pitch chain is planar, so this is exact, not linearised: over the working
ranges (base 0..45, elbow −25..34, wrist −20..60) the secant slope equals the
derivative to 0.00°. Hence `elevation = e_rest − Δbase_pitch + Δelbow_pitch −
Δwrist_pitch`, and for every channel `c` feeding a body joint the cancelling
`wrist_pitch` term is

```
gain_wrist(c) = −(J_base·g_base(c) + J_elbow·g_elbow(c)) / J_wrist = −g_base(c) + g_elbow(c)
```

written into `ROBOT.md` as ordinary `mix` terms (`torso_lean_fwd`, `head_x`,
`head_z`, `mouth_open`). The old mapping had `torso_lean_fwd → wrist_pitch
+0.5` — the wrong sign for the vendor's `+wrist_pitch = head down`; a 20°
lean dropped the gaze 22°. `tests/test_retarget_features.py::test_lamp_compensation_terms_are_consistent_with_the_planar_chain`
keeps the file self-consistent. Residual error comes only from the soft
limit/clamp on `wrist_pitch`'s short upward range (rest −62.4, min −85):
dropping the head 50 mm (`head_z = −50`) asks the wrist for +28° of look-up
and gets ~22°.

### 2.3 Reachy antennas — mirror hinges, anchors from Pollen's library

On the vendor URDF/daemon `+antenna_right` swings outward and
`+antenna_left` swings inward (both toward −y), so a symmetric "ears out" is
`left = −right`. Define splay `s = (right − left)/2` and tilt
`c = (right + left)/2`. Over the 16 native clips (deg):

| clip | splay mean | splay p95 | tilt sd | reading |
|---|---|---|---|---|
| attentive1 / amazed1 / curious1 | 31 / 49 / 16 | 45 / 55 / 49 | 4 / 9 / 19 | perked, alert |
| surprised1 / welcoming1 | 38 / 22 | 109 / 67 | 2 / 3 | a flap |
| laughing1 / yes1 / cheerful1 / thoughtful1 | 14 / 16 / 21 / 12 | 18 / 22 / 21 / 12 | ≤ 1 | slightly perked |
| sad1 / confused1 / boredom1 | 87 / 81 / 74 | 130 / 112 / 131 | 9 / 4 / 10 | drooped past horizontal |
| no1 | −27 | 6 | 1 | ears crossed inward |
| pooled | | | | tilt ~ head_roll slope **−0.67** (r −0.45); splay ~ head_pitch slope **−2.2** (r −0.47) |

Mapping (both joints, opposite signs on every expressive term, same sign on
the roll term): brow raise → 55·splay (a full raise = the alert clips' p95),
brow furrow → 85 (the sad/confused/bored mean), mouth open → 15 (the
talking/laughing perk), head pitch → −0.8 splay/deg (the pooled −2.2 damped so
ordinary nods only flutter the ears), head roll → −0.67 common-mode tilt.
Under the human corpus the antennas' |.|p95 is 38–40°, i.e. the attentive
band; the vendor's 90–127° p95 is its droop clips, which no human channel
should hit in conversation. The old mapping sent `+90·brow` to both joints —
on hardware a tilt toward the robot's right, not a splay. The hardware array
order (`[left, right]` on this unit vs the SDK's `[right, left]`) is the
sink's concern; the mapping is per named joint.

### 2.4 Trackers, idle and soft limits chosen

| robot | joint(s) | spring | idle | soft_limit |
|---|---|---|---|---|
| lamp | base_yaw / base_pitch | 2.5 Hz ζ 1.0 / 2.0 Hz ζ 1.0 | – | 0.15 |
| lamp | elbow_pitch | 2.5 Hz ζ 0.9 | 1.5° @ 0.25 Hz (breathing) | 0.15 |
| lamp | wrist_roll / wrist_pitch | 4 Hz ζ 0.7 | 5° @ 0.2 Hz / 3° @ 0.2 Hz | 0.15 / 0.08 |
| reachy | head_yaw / pitch / roll | 4 Hz ζ 0.7 | – / 0.5° @ 0.22 / 0.4° @ 0.17 | 0.15 |
| reachy | head_x / y / z | 2.5 Hz ζ 1.0 | – / – / 1 mm @ 0.22 Hz | 0.15 |
| reachy | body_yaw | 2 Hz ζ 1.0 | – | 0.15 |
| reachy | antennas | 3.5 Hz ζ 0.45 | 3° @ 0.3 Hz | 0.15 |

Lamp idle amplitudes are ≈ 0.8 × the per-joint std of the vendor's own
`idle.csv` (0.6 / 2.3 / 2.1 / 6.2 / 4.4 for yaw / base / elbow / roll / wrist)
and the frequencies its dominant FFT peaks (0.1–0.3 Hz); base joints get no
idle so the gaze compensation is not disturbed.

## 3. Evaluation (`scripts/retarget_eval.py --robot <name> --before HEAD`)

(a) envelope match — per-joint |.|p95 ratio retargeted/vendor, score
`exp(−mean |log ratio|)`; (b) gaze error under lean (lamp, FK); (c) legality
— speed-cap and limit violations over the whole corpus, offline and live;
(d) stillness (frames < 5 units/s) and velocity-histogram W1 vs the vendor;
(e) JS-parity readiness. Results below are the working tree against `HEAD`,
on the corpus at the time of the run.

RESULTS_PLACEHOLDER

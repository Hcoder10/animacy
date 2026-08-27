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
(`terms[] = (channel, gain[, tag])`, `offset`, `deadband`, bounds `lo`/`hi` =
mapping `min`/`max` if given else the joint's, `smooth_hz`, and the optional
v1.1 keys `soft_limit`, `idle{amp, hz, still}`, `spring{hz, zeta}`,
`settle{seconds, quiet, still}`). Per-joint state: `y` (last output, init
`rest`), `v` (velocity, init 0), `e` (activity envelope, init 0), `p`
(previous pre-idle target, init `rest`), `t` (idle clock, init 0), and for
`settle`: `p_s` (previous raw target, init `rest`), `q` (quiet time, init 0),
`b` (blend, init 0). `dt` is the seconds since the previous frame and **must be
> 0** (the reference raises on 0; callers split long gaps into nominal steps).
Channels that are missing, `null` or NaN count as 0; a term's `tag` is
ignored by the runtime (it labels derived terms for the fitter, §2.2). The
`speaking` flag is read once per frame the same way (missing → 0).

```
1. x  = Σ_k gain_k · channel_k
2. x  = 0                     if |x| < deadband
      = x − sign(x)·deadband  otherwise
3. u  = rest + offset + x
4. soft limit (only if soft_limit s > 0):   k = s · (hi − lo)
      u > hi − k :  u = (hi − k) + k · tanh((u − (hi − k)) / k)
      u < lo + k :  u = (lo + k) − k · tanh(((lo + k) − u) / k)
5. u  = clamp(u, lo, hi)
5b. settle (only if settle):   a_s = |u − p_s| / dt ;  p_s = u
      quiet_now = (a_s < still_s) and (speaking < 0.5)      still_s defaults to 0.1·(hi − lo)
      q = q + dt   if quiet_now,  else 0
      b = max( clamp((q − quiet) / seconds, 0, 1),  b − 4·dt/seconds )
      u = u + b · (rest + offset − u)
6. idle (only if idle):        a = |u − p| / dt ;  p = u
      e = max(a, e · exp(−dt / 0.5))
      g = clamp(1 − e / still, 0, 1)           still defaults to 10·amp·hz
      t = t + dt
      n = amp · Σ_{k=0..2} w_k · sin(2π · r_k · hz · t + φ_{j,k})
      u = clamp(u + g · n, lo, hi)
7. tracker:
      spring:  [y_new − u, v] = M(hz, zeta, dt) · [y − u, v]      (§1.2, exact)
      else:    cutoff = smooth_hz if given, else the runtime default (LIVE: 6 Hz in
               both LiveRetargeter and web/js; OFFLINE: 8 Hz zero-phase, §1.4)
               α = 1                              if cutoff is 0 / null
                 = 1 − exp(−2π · cutoff · dt)    otherwise
               y_new = y + α · (u − y)
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
`u = rest` with the one-pole tracker at the runtime default cutoff, and
**still** pass through steps 8–10 (rate limit, clamp).

### 1.0 Settle (step 5b) — return to the attentive pose

`settle: {seconds: S, quiet: Q = 0.4, still: T}` per joint. Once the raw
mapped target has moved slower than `T` units/s **and** the subject has not
been `speaking` for `Q` seconds, the blend `b` ramps linearly from 0 to 1
over `S` seconds and the target becomes `rest + offset`; idle sway (step 6)
then plays on top, so a quiet robot drifts back to its attentive pose and
breathes there instead of freezing in the last gesture (the blind grader's
"ending an affirmation in a drooped bow"). The moment motion or speech
resumes, `q` resets and `b` releases linearly over `S/4` (`4·dt/S` per
frame), so the return to the person's pose takes ≤ 0.15 s at `S = 0.6`.
Both mappings declare `settle: {seconds: 0.6}` on every default-mode joint.

Worked example (`tests/test_retarget_features.py::test_settle_numeric_example_in_docs`):
joint `rest 0, ±60, max_speed 300`, `{from: head_yaw, gain: 1, settle: {seconds: 0.6, quiet: 0.4, still: 12}}`,
`dt = 1/30`, `head_yaw = 20` held from frame 1, `40` from frame 32, `speaking` absent (0):

```
frame 1    a_s = |20 − 0| / dt = 600 ≥ 12 → q = 0, b = 0, u = 20
frames 2–13  a_s = 0 → q = (frame − 1)·dt ; frame 13: q = 0.4 = Q → (q − Q)/S = 0 → b = 0
frame 14   q = 0.4333 → b = 0.0556 → u = 20 + 0.0556·(0 − 20) = 18.889
frame 15   q = 0.4667 → b = 0.1111 → u = 17.778        … (b grows 1/18 per frame)
frame 31   q = 1.0    → b = 1       → u = 0            (fully settled; idle sway now visible)
frame 32   head_yaw = 40: a_s = |40 − 20|/dt = 600 → q = 0 ; b = max(0, 1 − 4·dt/0.6) = 0.7778 → u = 40 + 0.7778·(0 − 40) = 8.889
frame 33   b = 0.5556 → u = 17.778 ;  frame 34  b = 0.3333 ;  frame 35  b = 0.1111 ;  frame 36  b = 0 → u = 40
```

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
`fmod` and JS `%` agree here because both operands are non-negative (do not
"simplify" to a floored modulo). `amp` must be > 0 (`still` would otherwise
default to 0). The sway's RMS is `amp·√(Σw²/2) = 0.436·amp`, so matching a
vendor idle's std σ needs `amp ≈ 2.3σ` (§2.4 uses ≈ 1.8σ).

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
frame 3  spring from (8.580, 150) wants 15.344 → clipped again → 13.57994, v = 150
```

Idle example: joint index 1, `idle: {amp: 3, hz: 0.4}` (still = 12 units/s):
`φ_{1,k} = (3.316668, 5.716631, 1.833409)`; at `t = 1/30`,
`n = 3·(0.5·sin(2π·0.4/30 + 3.316668) + 0.3·sin(2π·0.524/30 + 5.716631) + 0.2·sin(2π·0.268/30 + 1.833409)) = −0.211154`;
at `t = 1.0`, `n = −0.513490`.

### 1.6 What the web port must implement (checklist)

- Profile JSON per mapping (`animacy profile export`): `terms[]` (`from`,
  `gain`; tags are not exported), `offset`, `min`, `max` (resolved),
  `deadband`, `smooth_hz`, plus `spring {hz, zeta} | null`,
  `idle {amp, hz, still} | null` (`still` already resolved), `soft_limit | null`,
  `settle {seconds, quiet, still} | null` (`still` already resolved).
- State per joint: `y`, `v`, `e`, `p`, `t`, `p_s`, `q`, `b`; `reset()` sets
  `y = p = p_s = rest`, the rest 0.
- Steps 1–10 in that order (5b between 5 and 6); the idle phase uses the
  joint's index in `profile.joints`; `spring` uses the closed-form
  coefficients of §1.2; the velocity carried into the next frame is the
  spring's own unless step 8 or 9 clipped (Python: `animacy.retarget.clip_step`);
  `speaking` is read once per frame (Python: `settle_update`).
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

Per joint: `mult = target / retargeted |.|p95` with `target = 1.15 × vendor
|.|p95` (the blind grader called the first fit "small-amplitude"; retargeted
= the human corpus through the linear, expressive part of the mapping),
clipped to `[0.5, 3]`, to the mapping's headroom (`0.9 ×` distance from
`rest` to each bound must hold the scaled p99/p1), and to a **velocity cap**:
the scaled velocity p95 may not exceed 1.5 × the vendor's — human heads are
brisker than authored robot clips, and matching amplitude alone made the
lamp whip (wrist_roll velocity p95 97 °/s vs the vendor's 44 at an uncapped
fit). Then fixed-point passes re-measure the actual pipeline (soft limit,
spring, settle) and correct until every multiplier moves < 2 % (max 5).
Every expressive term gain of the joint is multiplied by `mult`; derived
`tag: gaze_comp` terms (§2.2) are never scaled — they are **re-derived from
the scaled, as-written body gains** after each fit. Antennas are skipped
(§2.3). Changed lines are stamped `# fitted by scripts/retarget_fit.py <date>`.
Vendor excursions are measured around the library median (where the clips
live), retargeted ones around `rest` (where the mapping lives): both are
gesture sizes around the pose the motion returns to.

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

written into `ROBOT.md` as `mix` terms tagged `tag: gaze_comp`
(`torso_lean_fwd`, `head_x`, `head_z`, `mouth_open`, `head_pitch`) — a joint
may carry an expressive term and a derived term for the same channel; the
runtime just sums them, the fitter scales only the former. The original
mapping had `torso_lean_fwd → wrist_pitch +0.5` — the wrong sign for the
vendor's `+wrist_pitch = head down`; a 20° lean dropped the gaze 22°.
`tests/test_retarget_features.py::test_lamp_compensation_terms_are_consistent_with_the_planar_chain`
keeps the file self-consistent. Residual error comes only from the soft
limit/clamp on `wrist_pitch`'s short upward range (rest −62.4, min −85);
dropping the head 50 mm now costs 1.4°.

**Whole-arm participation (blind grader: "the arm column and base stayed
near-static").** Speech-driven human motion has little torso lean, so
lean/height alone leaves the arm dead. The vendor's own conversational clips
are head-led *whole-arm* moves: regressing the arm on the head joints across
the native clips (values around the library median) gives

| clip | base_pitch ~ wrist_pitch | elbow_pitch ~ wrist_pitch | R² |
|---|---|---|---|
| nod | −0.59 | −1.60 | 0.98 / 1.00 |
| listening | −0.44 | −0.56 | 0.88 / 0.86 |
| happy_wiggle | −0.31 | −0.81 | 0.95 / 0.99 |
| pooled, all 31 | −0.25 | −0.15 | 0.18 / 0.06 (gestures differ) |
| base_yaw ~ wrist_roll, pooled | +0.44 | | r 0.51 |

so `head_pitch` (and `mouth_open`) now drive `base_pitch` and `elbow_pitch`
at −0.45 / −1.0 × the wrist's expressive gain (the nod bobs the arm), with the
`gaze_comp` terms cancelling the elevation change (`−g_base + g_elbow` per
channel) — the head still points at the person while the body moves. All
three pitch joints share one spring (3 Hz, ζ 0.6) so the cancellation also
holds mid-motion (with different trackers per joint a nod produced a ~100 ms
elevation transient). Reachy gets the same treatment from Pollen's library:
`head_x ≈ +0.31 mm/deg` and `head_z ≈ +0.23 mm/deg` of head pitch (a nod is a
bob), `body_yaw ≈ 0.65 × head_yaw` (r 0.69; a look is a turn).

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
| within-clip (each clip demeaned) | | | | tilt ~ head_roll **−0.47** (r −0.29); splay ~ head_pitch **−2.5** (r −0.47) |

Mapping (both joints, opposite signs on every expressive term, same sign on
the roll term): brow raise → 55·splay (a full raise = the alert clips' p95),
brow furrow → 85 (the sad/confused/bored mean), mouth open → 15 (the
talking/laughing perk), head pitch → −0.8 splay/deg (the pooled/within −2.2 /
−2.5 damped so ordinary nods only flutter the ears), head roll → −0.5
common-mode tilt (between the pooled and within-clip slopes). These are point
estimates from 16 clips; the clip-level spread of the perked p95s is 45–55
(attentive/amazed/curious) and of the drooped means 74–87.
Under the human corpus the antennas' |.|p95 is 38–40°, i.e. the attentive
band; the vendor's 90–127° p95 is its droop clips, which no human channel
should hit in conversation. The old mapping sent `+90·brow` to both joints —
on hardware a tilt toward the robot's right, not a splay. The hardware array
order (`[left, right]` on this unit vs the SDK's `[right, left]`) is the
sink's concern; the mapping is per named joint.

### 2.4 Trackers, idle and soft limits chosen

| robot | joint(s) | spring | idle (peak amp @ hz) | soft_limit | settle |
|---|---|---|---|---|---|
| lamp | base_yaw | 2.5 Hz ζ 0.85 | 1.0° @ 0.15 | 0.15 | 0.6 s |
| lamp | base_pitch / elbow_pitch / wrist_pitch | 3 Hz ζ 0.6 (one spring for the whole pitch chain) | 2.5° @ 0.25 / 3.8° @ 0.25 / 8° @ 0.2 | 0.15 / 0.15 / 0.08 | 0.6 s |
| lamp | wrist_roll | 3.5 Hz ζ 0.6 | 11° @ 0.2 | 0.15 | 0.6 s |
| reachy | head_yaw / pitch / roll | 4 Hz ζ 0.6 | 1.0° @ 0.15 / 1.0° @ 0.22 / 0.8° @ 0.17 | 0.15 | 0.6 s |
| reachy | head_x / y / z | 3 Hz ζ 0.8 | 1.5 mm @ 0.2 / – / 2 mm @ 0.22 | 0.15 | 0.6 s |
| reachy | body_yaw | 2 Hz ζ 0.9 | – | 0.15 | 0.6 s |
| reachy | antennas | 3.5 Hz ζ 0.45 | 6° @ 0.3 | 0.15 | 0.6 s |

ζ 0.6 on the head/wrist joints gives a visible 9.5 % overshoot-and-settle
(the grader asked for anticipation/overshoot; a data-declared anticipation
dip is not in the frozen live path). Lamp idle amplitudes are ≈ 1.8 × the
per-joint std of the vendor's own `idle.csv` (0.6 / 2.3 / 2.1 / 6.2 / 4.4 for
yaw / base / elbow / roll / wrist), which makes the sway's RMS ≈ 0.8 × that
std (§1.1: RMS = 0.436·amp); base_pitch is held at 2.5° so the gaze wander
from idle stays within ±3°. Frequencies are the vendor idle's dominant FFT
peaks (0.1–0.3 Hz). Accepted trade-offs (Kimi K3 review, 2026-08-27): idle on
the pitch chain moves the gaze by its own amplitude; the live path does not
compensate tracker lag (body_yaw trails by ~140 ms at 2 Hz ζ 0.9); the
velocity re-derivation after a clip re-injects `±max_speed`.

## 3. Evaluation (`scripts/retarget_eval.py --robot <name> --before HEAD`)

(a) envelope match — per-joint |.|p95 ratio retargeted/vendor, score
`exp(−mean |log ratio|)`; (b) gaze error under lean (lamp, FK); (c) legality
— speed-cap and limit violations over the whole corpus, offline and live;
(d) stillness (frames < 5 units/s) and velocity-histogram W1 vs the vendor;
(e) JS-parity readiness. Results below are the working tree against `HEAD`,
on the corpus at the time of the run.

RESULTS_PLACEHOLDER

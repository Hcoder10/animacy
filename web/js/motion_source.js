// Motion sources: everything that can put a frame on the robots.
//
// Interface (duck-typed):
//   await source.start()          acquire resources (camera, model, ...)
//   source.stop()                 release them
//   source.onFrame(cb)            cb({t, dt, channels?, joints?}) — optional push API
//   source.update(realDt) → frame|null   pull API, called once per animation frame
//
// A frame is one of
//   { t, dt, channels: {<canonical channel>: number} }     → goes through each
//        robot's LiveRetargeter (web/js/retarget.js)
//   { t, dt, joints: { <robot name>: {<joint>: value} } }   → applied directly
//        (vendor-native clips, already in the robot's own joint space)
//
// `dt` is the time the retargeter should integrate over: clip time for clips
// (so speed × 2 plays a faster robot, like the real thing would), wall time
// for the webcam. `seek: true` marks a timeline discontinuity (scrub); the
// consumer settles the retargeter there rather than integrating across it.

export class MotionSource {
  constructor() { this._cb = null; }
  async start() {}
  stop() {}
  onFrame(cb) { this._cb = cb; }
  _emit(frame) { if (this._cb) this._cb(frame); }
  update(_realDt) { return null; }
}

/** Shared transport for anything with a timeline. */
export class ClipSource extends MotionSource {
  /**
   * @param {import('./clips.js').Track} track
   * @param {object} [o]
   * @param {boolean} [o.loop]
   * @param {number} [o.speed]
   */
  constructor(track, { loop = true, speed = 1.0 } = {}) {
    super();
    this.track = track;
    this.loop = loop;
    this.speed = speed;
    this.playing = true;
    this.time = 0;
    this.finished = false;
    this._pendingSeek = false;
  }

  get duration() { return this.track.duration; }

  seek(t) {
    this.time = Math.min(Math.max(t, 0), this.duration);
    this.finished = false;
    this._pendingSeek = true;
  }

  play() { if (this.finished && !this.loop) this.seek(0); this.playing = true; }
  pause() { this.playing = false; }
  toggle() { this.playing ? this.pause() : this.play(); }

  /** Advance by wall-clock dt; returns the frame at the new time (always, so scrubbing while paused re-samples). */
  update(realDt) {
    let dtClip = 0;
    if (this.playing && !this.finished) {
      dtClip = realDt * this.speed;
      this.time += dtClip;
      if (this.time >= this.duration) {
        if (this.loop) this.time = this.duration > 0 ? this.time % this.duration : 0;
        else { this.time = this.duration; this.finished = true; this.playing = false; }
      }
    }
    // A seek is a discontinuity: flag it so the consumer settles the pose
    // (repeated nominal steps) instead of slewing from wherever it was.
    const seek = this._pendingSeek;
    const dt = Math.max(dtClip, 1e-4);
    this._pendingSeek = false;
    const frame = this._frameAt(this.time, dt, seek);
    this._emit(frame);
    return frame;
  }

  _frameAt(t, dt, seek = false) {
    const s = this.track.sample(t);
    if (this.track.kind === 'canonical') return { t, dt, seek, channels: s };
    return { t, dt, seek, joints: { [this.track.robot]: s } };
  }
}

/** A clip synthesised in JS (calibration clips) — same transport, different origin. */
export class SyntheticSource extends ClipSource {}

/**
 * Live webcam puppeteering via MediaPipe (FaceLandmarker + PoseLandmarker),
 * producing canonical channels with the signs in docs/CANONICAL.md.
 * The MediaPipe bundle (~10 MB with models) is imported lazily on start().
 */
export class WebcamSource extends MotionSource {
  /**
   * @param {object} o
   * @param {HTMLVideoElement} o.video
   * @param {HTMLCanvasElement} [o.overlay]  face-mesh / pose overlay (same size as the thumbnail)
   * @param {'right'|'left'|'none'} [o.arm]
   * @param {(msg:string)=>void} [o.onStatus]
   * @param {object} o.cdn  {vision: module url, wasm: wasm dir, faceModel, poseModel}
   */
  constructor({ video, overlay = null, arm = 'right', onStatus = null, cdn }) {
    super();
    this.video = video;
    this.overlay = overlay;
    this.arm = arm;
    this.onStatus = onStatus || (() => {});
    this.cdn = cdn;
    this.stream = null;
    this.face = null;
    this.pose = null;
    this.vision = null;
    this.calibrator = null;
    this.latest = null;         // newest canonical frame not yet consumed by update()
    this.lastTs = -1;
    this.lastVideoTime = -1;
    this.running = false;
    this.fps = 0;
    this._fpsAcc = { n: 0, t: 0 };
    this._rvfcHandle = null;
    this._rafHandle = null;
    this.autoNeutralAt = null;  // timestamp (s) after which the first neutral is auto-captured
    this.hasFace = false;
    this.hasPose = false;
    this.speaking = 0;
    this._audio = null;
    this.drawUtils = null;
    this.stats = { faceMs: 0, poseMs: 0 };
    this._canonical = null;
  }

  async start() {
    if (this.running) return;
    this.onStatus('loading MediaPipe…');
    const [{ FilesetResolver, FaceLandmarker, PoseLandmarker, DrawingUtils }, canonical] = await Promise.all([
      import(this.cdn.vision),
      import('./canonical.js'),
    ]);
    this._canonical = canonical;
    this.FaceLandmarker = FaceLandmarker;
    this.PoseLandmarker = PoseLandmarker;
    this.DrawingUtils = DrawingUtils;
    this.vision = await FilesetResolver.forVisionTasks(this.cdn.wasm);
    this.calibrator = new canonical.NeutralCalibrator(1.0);

    const mk = async (Cls, opts) => {
      try {
        return await Cls.createFromOptions(this.vision, { ...opts, baseOptions: { ...opts.baseOptions, delegate: 'GPU' } });
      } catch (e) {
        console.warn('[webcam] GPU delegate failed, falling back to CPU:', e && e.message);
        return await Cls.createFromOptions(this.vision, { ...opts, baseOptions: { ...opts.baseOptions, delegate: 'CPU' } });
      }
    };
    this.onStatus('loading face + pose models…');
    [this.face, this.pose] = await Promise.all([
      mk(FaceLandmarker, {
        baseOptions: { modelAssetPath: this.cdn.faceModel },
        runningMode: 'VIDEO',
        numFaces: 1,
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
      }),
      mk(PoseLandmarker, {
        baseOptions: { modelAssetPath: this.cdn.poseModel },
        runningMode: 'VIDEO',
        numPoses: 1,
      }),
    ]);

    this.onStatus('opening camera…');
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user', frameRate: { ideal: 30 } },
      audio: false,
    });
    this.video.srcObject = this.stream;
    this.video.muted = true;
    this.video.playsInline = true;
    await this.video.play();
    this._startVad();

    if (this.overlay) {
      this.overlay.width = 320;
      this.overlay.height = Math.round((320 * (this.video.videoHeight || 480)) / (this.video.videoWidth || 640));
      this.drawUtils = new DrawingUtils(this.overlay.getContext('2d'));
    }
    this.running = true;
    this.autoNeutralAt = performance.now() / 1000 + 1.8;
    this.onStatus('tracking — hold still for neutral…');
    this._scheduleFrame();
  }

  /** Optional VAD for the `speaking` flag: RMS energy on the mic, if we can get one. Never throws. */
  async _startVad() {
    try {
      const mic = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const src = ctx.createMediaStreamSource(mic);
      const an = ctx.createAnalyser();
      an.fftSize = 512;
      src.connect(an);
      this._audio = { ctx, mic, an, buf: new Float32Array(an.fftSize), level: 0, noise: 0.01 };
    } catch (e) {
      this._audio = null; // no mic → speaking stays 0
    }
  }

  _readVad() {
    const a = this._audio;
    if (!a) return 0;
    a.an.getFloatTimeDomainData(a.buf);
    let s = 0;
    for (let i = 0; i < a.buf.length; i++) s += a.buf[i] * a.buf[i];
    const rms = Math.sqrt(s / a.buf.length);
    a.noise = Math.min(a.noise * 1.001 + 1e-5, Math.max(rms, 1e-4)) ; // slow noise-floor tracker
    a.level = 0.7 * a.level + 0.3 * rms;
    return a.level > Math.max(0.02, a.noise * 3.5) ? 1 : 0;
  }

  _scheduleFrame() {
    if (!this.running) return;
    const cb = () => { this._onVideoFrame(); this._scheduleFrame(); };
    if (this.video.requestVideoFrameCallback) this._rvfcHandle = this.video.requestVideoFrameCallback(cb);
    else this._rafHandle = requestAnimationFrame(cb);
  }

  _onVideoFrame() {
    const v = this.video;
    if (!this.running || v.readyState < 2 || v.videoWidth === 0) return;
    if (v.currentTime === this.lastVideoTime) return;
    this.lastVideoTime = v.currentTime;
    const nowMs = performance.now();
    const ts = Math.max(Math.floor(nowMs), this.lastTs + 1); // MediaPipe VIDEO mode needs strictly increasing ms
    const now = nowMs / 1000;
    let faceRes = null, poseRes = null;
    try {
      const t0 = performance.now();
      faceRes = this.face.detectForVideo(v, ts);
      const t1 = performance.now();
      poseRes = this.pose.detectForVideo(v, ts);
      const t2 = performance.now();
      this.stats.faceMs = 0.9 * this.stats.faceMs + 0.1 * (t1 - t0);
      this.stats.poseMs = 0.9 * this.stats.poseMs + 0.1 * (t2 - t1);
    } catch (e) {
      console.warn('[webcam] detect failed:', e && e.message);
      return;
    }
    const C = this._canonical;
    const fr = C.faceToRaw(faceRes);
    const pr = C.poseToRaw(poseRes, this.arm);
    this.hasFace = !!fr;
    this.hasPose = !!pr;
    const raw = { ...(pr || {}), ...(fr || {}) };
    raw.face_valid = fr ? 1 : 0;
    raw.arm_valid = pr && pr.arm_valid ? 1 : 0;
    if (fr) this.calibrator.push(raw, now);
    if (!this.calibrator.hasNeutral && fr && this.autoNeutralAt !== null && now >= this.autoNeutralAt && this.calibrator.bufferSeconds >= 0.8) {
      this.calibrator.setNeutral();
      this.onStatus('neutral set automatically — press "Set neutral" to redo');
    }
    this.speaking = this._readVad();
    const channels = C.toCanonical(raw, this.calibrator.neutral, now, this.speaking);
    const dt = this.lastTs < 0 ? 1 / 30 : Math.min(0.25, (ts - this.lastTs) / 1000);
    this.lastTs = ts;
    this.latest = { t: now, dt, channels, raw };
    this._fpsAcc.n++;
    if (now - this._fpsAcc.t >= 1) { this.fps = this._fpsAcc.n / (now - this._fpsAcc.t); this._fpsAcc = { n: 0, t: now }; }
    this._drawOverlay(faceRes, poseRes);
    this._emit(this.latest);
  }

  _drawOverlay(faceRes, poseRes) {
    if (!this.overlay || !this.drawUtils) return;
    const ctx = this.overlay.getContext('2d');
    ctx.clearRect(0, 0, this.overlay.width, this.overlay.height);
    ctx.save();
    ctx.lineWidth = 0.6;
    if (faceRes && faceRes.faceLandmarks && faceRes.faceLandmarks[0]) {
      const lm = faceRes.faceLandmarks[0];
      this.drawUtils.drawConnectors(lm, this.FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: 'rgba(120,200,255,0.35)', lineWidth: 0.5 });
      this.drawUtils.drawConnectors(lm, this.FaceLandmarker.FACE_LANDMARKS_LIPS, { color: '#ff7a59', lineWidth: 1.2 });
      this.drawUtils.drawConnectors(lm, this.FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, { color: '#ffd166', lineWidth: 1.2 });
      this.drawUtils.drawConnectors(lm, this.FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, { color: '#ffd166', lineWidth: 1.2 });
    }
    if (poseRes && poseRes.landmarks && poseRes.landmarks[0]) {
      this.drawUtils.drawConnectors(poseRes.landmarks[0], this.PoseLandmarker.POSE_CONNECTIONS, { color: 'rgba(140,255,180,0.7)', lineWidth: 1.5 });
    }
    ctx.restore();
  }

  /** "Set neutral": median of the last second of raw frames. */
  setNeutral() {
    if (!this.calibrator) return false;
    const n = this.calibrator.setNeutral();
    if (n) this.onStatus('neutral set');
    return !!n;
  }

  update(_realDt) {
    const f = this.latest;
    this.latest = null; // consume once; between camera frames the robots hold (retargeter keeps state)
    return f;
  }

  stop() {
    this.running = false;
    if (this._rvfcHandle && this.video.cancelVideoFrameCallback) this.video.cancelVideoFrameCallback(this._rvfcHandle);
    if (this._rafHandle) cancelAnimationFrame(this._rafHandle);
    if (this.stream) for (const tr of this.stream.getTracks()) tr.stop();
    this.stream = null;
    this.video.srcObject = null;
    if (this._audio) {
      try { for (const tr of this._audio.mic.getTracks()) tr.stop(); this._audio.ctx.close(); } catch (e) { /* ignore */ }
      this._audio = null;
    }
    try { if (this.face) this.face.close(); } catch (e) { /* ignore */ }
    try { if (this.pose) this.pose.close(); } catch (e) { /* ignore */ }
    this.face = this.pose = null;
    if (this.overlay) this.overlay.getContext('2d').clearRect(0, 0, this.overlay.width, this.overlay.height);
    this.onStatus('stopped');
  }
}

// The learned motion model lives in model.js (ONNX runner, retrieval index,
// envelope heuristic) and is driven by TalkSource / ListenSource in talk.js:
// text → Kokoro TTS → features.js → model.js → canonical frames → the same
// {t, dt, channels} frames as a canonical clip.

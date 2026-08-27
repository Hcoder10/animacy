// One three.js viewport showing one URDF robot.
//
// URDFs are z-up (ROS); three.js is y-up. The robot is parented under a group
// rotated −90° about x so +z(URDF) → +y(three) and +x(URDF, forward) stays +x.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import URDFLoader from 'urdf-loader';

const tmpM = new THREE.Matrix4();
const tmpM2 = new THREE.Matrix4();

export class RobotViewer {
  /**
   * @param {HTMLElement} container  element to fill (position: relative)
   * @param {object} [opts]
   * @param {string} [opts.label]
   */
  constructor(container, opts = {}) {
    this.container = container;
    this.robot = null;
    this.jointNames = [];
    this.anchorLink = null;
    this.ready = false;
    this.error = null;

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.renderer.domElement.className = 'viewer-canvas';
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0e1117);
    this.scene.fog = new THREE.Fog(0x0e1117, 2.5, 6);

    this.camera = new THREE.PerspectiveCamera(38, 1, 0.01, 50);
    this.camera.position.set(0.9, 0.6, 0.9);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.maxPolarAngle = Math.PI * 0.52;
    this.controls.minDistance = 0.08;
    this.controls.maxDistance = 6;
    this.controls.target.set(0, 0.15, 0);

    // lighting: soft hemisphere + warm key + cool rim (kept low: the shells are near-white)
    this.scene.add(new THREE.HemisphereLight(0xdfe7ff, 0x1a1d24, 0.55));
    const key = new THREE.DirectionalLight(0xfff1e0, 1.25);
    key.position.set(1.5, 2.5, 1.2);
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0x8fb8ff, 0.6);
    rim.position.set(-1.5, 1.5, -1.5);
    this.scene.add(rim);
    const fill = new THREE.DirectionalLight(0xffffff, 0.35);
    fill.position.set(-1, 0.8, 2);
    this.scene.add(fill);

    // ground
    const grid = new THREE.GridHelper(4, 40, 0x3a4256, 0x232938);
    grid.position.y = -0.0005;
    this.scene.add(grid);
    this.grid = grid;
    const disc = new THREE.Mesh(
      new THREE.CircleGeometry(1.6, 64),
      new THREE.MeshStandardMaterial({ color: 0x151a24, roughness: 0.95, metalness: 0.0 }),
    );
    disc.rotation.x = -Math.PI / 2;
    disc.position.y = -0.002;
    this.scene.add(disc);

    // z-up → y-up
    this.root = new THREE.Group();
    this.root.rotation.x = -Math.PI / 2;
    this.scene.add(this.root);
    this.pivot = new THREE.Group();
    this.root.add(this.pivot);
    this.scene.updateMatrixWorld(true); // root/pivot world matrices must be valid before the first frame()/anchor pass

    this._ro = new ResizeObserver(() => this.resize());
    this._ro.observe(container);
    this.resize();
  }

  resize() {
    const w = Math.max(1, this.container.clientWidth);
    const h = Math.max(1, this.container.clientHeight);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /**
   * Load a URDF. Resolves when the URDF and all its meshes are in.
   * @param {object} o
   * @param {string} o.urdfUrl
   * @param {Object<string,string>} [o.packages]  package://<name>/ → url
   * @param {number} [o.meshScale]
   * @param {number} [o.cameraDistance]
   * @param {string|null} [o.anchorLink]  dev: link to pin at the origin (re-roots a badly rooted URDF)
   */
  async loadRobot({ urdfUrl, packages = {}, meshScale = 1.0, cameraDistance = null, anchorLink = null }) {
    this.clearRobot();
    const manager = new THREE.LoadingManager();
    const loader = new URDFLoader(manager);
    loader.packages = packages;
    loader.parseCollision = false;
    loader.parseVisual = true;
    const missing = [];
    manager.onError = (url) => missing.push(url);

    const robot = await new Promise((resolve, reject) => {
      let parsed = null;
      let allLoaded = false;
      const done = () => { if (parsed && allLoaded) resolve(parsed); };
      manager.onLoad = () => { allLoaded = true; done(); };
      loader.load(urdfUrl, (r) => { parsed = r; done(); }, undefined, (err) => reject(err instanceof Error ? err : new Error(`URDF load failed: ${urdfUrl}`)));
      // meshes may all be cached/instant → manager.onLoad may already have fired
      setTimeout(() => { if (parsed && !allLoaded) { allLoaded = true; done(); } }, 15000);
    });
    if (missing.length) console.warn(`[viewer] ${missing.length} mesh(es) failed to load for ${urdfUrl}:`, missing.slice(0, 5));

    // nicer materials, keep URDF colours
    robot.traverse((o) => {
      if (o.isMesh) {
        const src = o.material;
        const color = src && src.color ? src.color.clone() : new THREE.Color(0xb8bec9);
        // STL files may carry zero/garbage facet normals (they render black); rebuild from the winding.
        if (o.geometry && o.geometry.attributes && o.geometry.attributes.position) o.geometry.computeVertexNormals();
        // DoubleSide: exported STLs often mix windings; three flips the normal per fragment so both sides shade right.
        o.material = new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0.12, flatShading: false, side: THREE.DoubleSide });
        o.castShadow = false;
        o.receiveShadow = false;
        o.frustumCulled = false;
      }
    });
    robot.scale.setScalar(meshScale);
    this.pivot.add(robot);
    this.robot = robot;
    this.jointNames = Object.keys(robot.joints).filter((n) => robot.joints[n].jointType !== 'fixed');
    this.anchorLink = anchorLink && robot.links[anchorLink] ? robot.links[anchorLink] : null;
    if (anchorLink && !this.anchorLink) console.warn(`[viewer] anchor link '${anchorLink}' not in URDF`);
    if (this.anchorLink) { robot.matrixAutoUpdate = false; this._applyAnchor(); }
    this.frame(cameraDistance);
    this.ready = true;
    return robot;
  }

  clearRobot() {
    if (this.robot) {
      this.pivot.remove(this.robot);
      this.robot.traverse((o) => { if (o.isMesh) { o.geometry.dispose(); if (o.material.dispose) o.material.dispose(); } });
    }
    this.robot = null;
    this.jointNames = [];
    this.anchorLink = null;
    this.ready = false;
  }

  /** Pin `anchorLink` at the pivot origin: robot.matrix = inverse(anchor in robot frame). */
  _applyAnchor() {
    const robot = this.robot;
    robot.matrix.identity();
    robot.updateMatrixWorld(true);
    tmpM.copy(this.pivot.matrixWorld).invert().multiply(this.anchorLink.matrixWorld); // anchor in pivot frame
    robot.matrix.copy(tmpM2.copy(tmpM).invert());
    robot.updateMatrixWorld(true);
  }

  /** Place the camera `distance` from the robot's bounding-box centre, slightly in front (+x) and to the side. */
  frame(distance = null) {
    const box = new THREE.Box3();
    if (this.robot) { this.scene.updateMatrixWorld(true); box.setFromObject(this.robot); } // world space = y-up
    let center = new THREE.Vector3(0, 0.15, 0);
    let size = 0.4;
    if (!box.isEmpty() && Number.isFinite(box.min.x)) {
      box.getCenter(center);
      size = box.getSize(new THREE.Vector3()).length();
    }
    // the profile's camera_distance is a hint; never closer than what fits the whole robot
    const fit = (0.5 * size) / Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)) * 1.05;
    const d = Math.max(distance || 0, fit, 0.3);
    this.bounds = { center: center.clone(), size, box, distance: d };
    this.controls.target.copy(center);
    this.setView('iso');
  }

  /**
   * Named camera positions around the current target. 'front' looks at the
   * robot from where a person would stand (+x): the robot's LEFT is on the
   * viewer's RIGHT — use it to check signs.
   * @param {'iso'|'front'|'left'|'right'|'top'} kind
   */
  setView(kind = 'iso') {
    const c = this.controls.target;
    const d = (this.bounds && this.bounds.distance) || 1;
    const az = THREE.MathUtils.degToRad(38);
    const p = {
      iso: [c.x + d * Math.cos(az), c.y + d * 0.42, c.z + d * Math.sin(az)],
      front: [c.x + d, c.y + d * 0.12, c.z],
      left: [c.x, c.y + d * 0.12, c.z - d], // robot's left side (three −z = URDF +y)
      right: [c.x, c.y + d * 0.12, c.z + d],
      top: [c.x + 0.01, c.y + d, c.z],
    }[kind] || null;
    if (!p) return;
    this.camera.position.set(p[0], p[1], p[2]);
    this.camera.near = Math.max(0.005, d / 200);
    this.camera.far = Math.max(20, d * 40);
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  /**
   * Set URDF joint values (radians / metres) by URDF joint name.
   * Unknown names are ignored (counted in this.unknownJoints for the readout).
   */
  setJoints(values) {
    if (!this.robot) return;
    for (const name in values) {
      const j = this.robot.joints[name];
      if (!j) continue;
      const v = values[name];
      if (v === null || v === undefined || Number.isNaN(v)) continue;
      j.setJointValue(v);
    }
    if (this.anchorLink) this._applyAnchor();
  }

  /** Current URDF joint values by URDF joint name. */
  getJoints() {
    const out = {};
    if (!this.robot) return out;
    for (const n of this.jointNames) out[n] = this.robot.joints[n].angle;
    return out;
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    this._ro.disconnect();
    this.clearRobot();
    this.renderer.dispose();
    if (this.renderer.domElement.parentNode) this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
  }
}

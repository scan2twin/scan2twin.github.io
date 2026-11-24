// Interactive viewer for the reconstructed, articulated plant model.
// Loads res/web_model/plant.glb (part meshes in rest pose, world metres, Y-up)
// and res/web_model/plant.json (kinematic tree with estimated joint axes),
// then animates every joint gently within its estimated range.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const BASE = "res/web_model/";
const root = document.getElementById("viewer");
const canvasHost = root.querySelector(".viewer-canvas");
const status = root.querySelector(".viewer-status");
const btnPlay = root.querySelector("#v-play");
const chkAxes = root.querySelector("#v-axes");
const rngAmp = root.querySelector("#v-amp");

let started = false;
const io = new IntersectionObserver((es) => {
  if (es.some((e) => e.isIntersecting) && !started) { started = true; io.disconnect(); init(); }
}, { rootMargin: "200px" });
io.observe(root);

async function init() {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  canvasHost.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf3f3f1);
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 50);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8a80, 2.4));
  const sun = new THREE.DirectionalLight(0xffffff, 1.1);
  sun.position.set(-2, 4, 3);
  scene.add(sun);

  const resize = () => {
    const w = canvasHost.clientWidth, h = canvasHost.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(canvasHost);
  resize();

  let tree, gltf;
  try {
    const loader = new GLTFLoader();
    [tree, gltf] = await Promise.all([
      fetch(BASE + "plant.json").then((r) => r.json()),
      loader.loadAsync(BASE + "plant.glb", (e) => {
        if (e.total) status.textContent = `Loading model… ${Math.round((100 * e.loaded) / e.total)}%`;
      }),
    ]);
  } catch (err) {
    status.textContent = "Could not load the 3D model.";
    console.error(err);
    return;
  }

  // One group per node. A joint group's matrix is its motion, expressed in the
  // rest-pose world frame; nesting the groups composes motions along the chain.
  const model = new THREE.Group();
  const groups = {};
  const joints = [];
  for (const n of tree.nodes) {
    const g = new THREE.Group();
    g.name = n.name;
    g.matrixAutoUpdate = false;
    groups[n.name] = g;
  }
  const axesGroup = [];
  for (const n of tree.nodes) {
    const g = groups[n.name];
    (n.parent ? groups[n.parent] : model).add(g);
    if (!n.axis) continue;
    const r = n.axis.rotation, t = n.axis.translation;
    const hasR = Math.hypot(...r.axis) > 0.5 && r.range[1] - r.range[0] > 1e-4;
    const hasT = Math.hypot(...t.axis) > 0.5 && t.range[1] - t.range[0] > 1e-5;
    if (!hasR && !hasT) continue;
    const j = {
      g,
      c: new THREE.Vector3(...r.center),
      ra: new THREE.Vector3(...r.axis).normalize(), rr: hasR ? r.range : [0, 0],
      ta: new THREE.Vector3(...t.axis).normalize(), tr: hasT ? t.range : [0, 0],
      phase: joints.length * 1.7,
      speed: 0.35 + ((joints.length * 0.137) % 0.25),
    };
    joints.push(j);
    // Axis glyph, attached to the parent so it follows upstream motion.
    const parent = n.parent ? groups[n.parent] : model;
    if (hasR) {
      const a = j.ra.clone();
      const line = new THREE.ArrowHelper(a, j.c.clone().addScaledVector(a, -0.08), 0.22, 0x2fbf3a, 0.04, 0.025);
      parent.add(line); axesGroup.push(line);
    }
    if (hasT) {
      const line = new THREE.ArrowHelper(j.ta, j.c.clone(), Math.max(0.08, (j.tr[1] - j.tr[0]) * 1.2), 0xe0312b, 0.03, 0.02);
      parent.add(line); axesGroup.push(line);
    }
  }

  const mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9, metalness: 0, side: THREE.DoubleSide });
  const meshes = [];
  gltf.scene.traverse((o) => { if (o.isMesh) meshes.push(o); });
  for (const m of meshes) {
    m.geometry.computeVertexNormals();
    m.material = mat;
    const node = m.parent && m.parent !== gltf.scene && m.parent.name === m.name ? m.parent : m;
    const g = groups[node.name] || groups[m.name];
    (g || model).add(node);
  }
  scene.add(model);

  // Frame the camera on the model.
  model.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(model);
  const ctr = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3()).length();
  controls.target.copy(ctr);
  camera.position.copy(ctr).add(new THREE.Vector3(1.0, 0.75, 1.15).normalize().multiplyScalar(size * 1.55));
  camera.near = size / 200; camera.far = size * 10;
  camera.updateProjectionMatrix();
  controls.update();

  const setAxes = () => axesGroup.forEach((a) => (a.visible = chkAxes.checked));
  chkAxes.addEventListener("change", setAxes);
  setAxes();

  let playing = true;
  const syncBtn = () => (btnPlay.textContent = playing ? "Pause" : "Play");
  btnPlay.addEventListener("click", () => { playing = !playing; syncBtn(); });
  syncBtn();

  const mR = new THREE.Matrix4(), mT = new THREE.Matrix4(), tmp = new THREE.Matrix4();
  let t = 0, last = performance.now();
  status.textContent = "";
  root.classList.add("ready");

  renderer.setAnimationLoop((now) => {
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    if (playing) t += dt;
    const amp = rngAmp.value / 100;
    for (const j of joints) {
      // s in [0,1]; interpolate between amp*min and amp*max (both ranges contain the rest pose 0)
      const s = 0.5 - 0.5 * Math.cos(t * j.speed * Math.PI + j.phase);
      const th = amp * (j.rr[0] + s * (j.rr[1] - j.rr[0]));
      const d = amp * (j.tr[0] + s * (j.tr[1] - j.tr[0]));
      mR.makeTranslation(j.c.x, j.c.y, j.c.z)
        .multiply(tmp.makeRotationAxis(j.ra, th))
        .multiply(tmp.makeTranslation(-j.c.x, -j.c.y, -j.c.z));
      mT.makeTranslation(j.ta.x * d, j.ta.y * d, j.ta.z * d);
      j.g.matrix.multiplyMatrices(mR, mT);
    }
    controls.update();
    renderer.render(scene, camera);
  });
}

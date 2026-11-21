"""Zero-dependency interactive Three.js PBR material inspector (`hypbr preview`)."""
from __future__ import annotations

import functools
import http.server
import socketserver
import webbrowser
from pathlib import Path

VIEWER_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>hypbr — PBR Material Inspector ({name})</title>
<style>
  :root {{ --bg:#0c0f14; --panel:#141923; --border:#252d3d; --text:#eef2f8; --dim:#9aa6ba; --acc:#5eead4; }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text); font:13px/1.45 -apple-system,BlinkMacSystemFont,Inter,sans-serif; overflow:hidden; }}
  #canvas {{ width:100vw; height:100vh; display:block; }}
  .hud {{ position:fixed; top:16px; left:16px; width:276px; background:rgba(20,25,35,0.88); backdrop-filter:blur(12px);
          border:1px solid var(--border); border-radius:12px; padding:14px; display:flex; flex-direction:column; gap:10px; }}
  .title {{ font-size:15px; font-weight:600; letter-spacing:0.01em; display:flex; justify-content:space-between; align-items:center; }}
  .badge {{ font:11px/1 ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--acc); background:rgba(94,234,212,0.12);
            border:1px solid rgba(94,234,212,0.3); padding:3px 7px; border-radius:99px; }}
  .row {{ display:flex; flex-direction:column; gap:4px; }}
  label {{ color:var(--dim); font-size:11px; text-transform:uppercase; letter-spacing:0.06em; display:flex; justify-content:space-between; }}
  select, input[type=range] {{ width:100%; accent-color:var(--acc); }}
  select {{ background:#0e121a; color:var(--text); border:1px solid var(--border); border-radius:7px; padding:6px 8px; }}
  .hint {{ color:var(--dim); font-size:11px; border-top:1px solid var(--border); padding-top:8px; }}
</style>
</head>
<body>
<canvas id="canvas"></canvas>
<div class="hud">
  <div class="title"><span>{name}</span><span class="badge">hypbr</span></div>
  <div class="row">
    <label>Channel View</label>
    <select id="channel">
      <option value="pbr">Final PBR (Lit)</option>
      <option value="albedo_delit.png">Albedo (De-lit)</option>
      <option value="normal.png">Normal (OpenGL +Y)</option>
      <option value="bent_normal.png">Bent Normal</option>
      <option value="ao.png">8-Azimuth HBAO</option>
      <option value="roughness.png">Roughness</option>
      <option value="metallic.png">Metallic</option>
      <option value="orm.png">glTF ORM</option>
      <option value="height.png">Height</option>
      <option value="curvature.png">Curvature</option>
      <option value="opacity.png">Opacity (Cutout)</option>
    </select>
  </div>
  <div class="row">
    <label>Geometry</label>
    <select id="geom">
      <option value="sphere">Sphere</option>
      <option value="knot">Torus Knot</option>
      <option value="box">Rounded Box</option>
      <option value="plane">Flat Tile Plane</option>
    </select>
  </div>
  <div class="row">
    <label><span>UV Tiling</span><span id="tileVal">2.0x</span></label>
    <input id="tile" type="range" min="1" max="4" step="0.5" value="2" />
  </div>
  <div class="row">
    <label><span>Displacement</span><span id="dispVal">0.03</span></label>
    <input id="disp" type="range" min="0" max="0.12" step="0.005" value="0.03" />
  </div>
  <div class="row">
    <label><span>Studio Light Angle</span><span id="lightVal">0°</span></label>
    <input id="light" type="range" min="0" max="360" step="1" value="35" />
  </div>
  <div class="hint">Drag to orbit · Scroll to zoom · Exact seamless tiling</div>
</div>
<script type="importmap">
{{"imports":{{"three":"https://unpkg.com/three@0.170.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.170.0/examples/jsm/"}}}}
</script>
<script type="module">
import * as THREE from "three";
import {{ OrbitControls }} from "three/addons/controls/OrbitControls.js";
import {{ RoomEnvironment }} from "three/addons/environments/RoomEnvironment.js";

const canvas = document.getElementById("canvas");
const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true }});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0c0f14);
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

const camera = new THREE.PerspectiveCamera(38, innerWidth / innerHeight, 0.1, 100);
camera.position.set(0, 0.35, 3.1);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;

const keyLight = new THREE.DirectionalLight(0xfff4e6, 2.2);
const rimLight = new THREE.DirectionalLight(0x88b4ff, 0.8);
rimLight.position.set(-3, 2, -2);
scene.add(keyLight, rimLight);

const loader = new THREE.TextureLoader();
const texCache = new Map();
function loadTex(file, srgb = false) {{
  const k = `${{file}}:${{srgb}}`;
  if (texCache.has(k)) return texCache.get(k);
  const t = loader.load(`./${{file}}`);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(2, 2);
  t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  t.anisotropy = renderer.capabilities.getMaxAnisotropy();
  texCache.set(k, t);
  return t;
}}

const pbrMat = new THREE.MeshPhysicalMaterial({{
  map: loadTex("albedo_delit.png", true),
  normalMap: loadTex("normal.png", false),
  roughnessMap: loadTex("roughness.png", false),
  metalnessMap: loadTex("metallic.png", false),
  aoMap: loadTex("ao.png", false),
  displacementMap: loadTex("height.png", false),
  displacementScale: 0.03,
  roughness: 1.0,
  metalness: 1.0,
}});
const flatMat = new THREE.MeshBasicMaterial({{ map: loadTex("albedo_delit.png", true) }});
fetch("./opacity.png").then(r => r.ok ? r.blob() : null).then(blob => {{
  if (!blob) return;
  const img = new Image();
  img.onload = () => {{
    const c = document.createElement("canvas");
    c.width = Math.min(img.width, 128); c.height = Math.min(img.height, 128);
    const ctx = c.getContext("2d");
    ctx.drawImage(img, 0, 0, c.width, c.height);
    const d = ctx.getImageData(0, 0, c.width, c.height).data;
    let holeCount = 0;
    const totalPx = d.length / 4;
    for (let i = 0; i < d.length; i += 4) {{ if (d[i] < 128) holeCount++; }}
    if (totalPx > 0 && (holeCount / totalPx) > 0.005) {{
      const opTex = loadTex("opacity.png", false);
      pbrMat.alphaMap = opTex;
      pbrMat.alphaTest = 0.5;
      pbrMat.side = THREE.DoubleSide;
      pbrMat.needsUpdate = true;
    }}
  }};
  img.src = URL.createObjectURL(blob);
}});

const geoms = {{
  sphere: new THREE.SphereGeometry(1, 128, 128),
  knot: new THREE.TorusKnotGeometry(0.68, 0.24, 220, 48),
  box: new THREE.BoxGeometry(1.5, 1.5, 1.5, 64, 64, 64),
  plane: new THREE.PlaneGeometry(2.4, 2.4, 128, 128),
}};
for (const g of Object.values(geoms)) g.setAttribute("uv2", g.attributes.uv);

const mesh = new THREE.Mesh(geoms.sphere, pbrMat);
scene.add(mesh);

function updateLight(deg) {{
  const rad = (deg * Math.PI) / 180;
  keyLight.position.set(Math.cos(rad) * 4, 3.0, Math.sin(rad) * 4);
  document.getElementById("lightVal").textContent = `${{deg}}°`;
}}
updateLight(35);

document.getElementById("channel").addEventListener("change", (e) => {{
  const v = e.target.value;
  if (v === "pbr") {{ mesh.material = pbrMat; }}
  else {{
    flatMat.map = loadTex(v, v === "albedo_delit.png");
    flatMat.needsUpdate = true;
    mesh.material = flatMat;
  }}
}});
document.getElementById("geom").addEventListener("change", (e) => {{
  mesh.geometry = geoms[e.target.value];
}});
document.getElementById("tile").addEventListener("input", (e) => {{
  const v = parseFloat(e.target.value);
  document.getElementById("tileVal").textContent = `${{v.toFixed(1)}}x`;
  for (const t of texCache.values()) t.repeat.set(v, v);
}});
document.getElementById("disp").addEventListener("input", (e) => {{
  const v = parseFloat(e.target.value);
  document.getElementById("dispVal").textContent = v.toFixed(3);
  pbrMat.displacementScale = v;
}});
document.getElementById("light").addEventListener("input", (e) => updateLight(parseFloat(e.target.value)));

addEventListener("resize", () => {{
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
}});

(function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}})();
</script>
</body>
</html>
"""


def render_preview_html(material_name: str) -> str:
    """Return the self-contained Three.js PBR material inspector HTML string."""
    return VIEWER_HTML.format(name=material_name)


def serve_preview(mat_dir: Path | str, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    """Serve an interactive Three.js PBR preview for `mat_dir` on a local HTTP server."""
    d = Path(mat_dir).resolve()
    if not (d / "albedo_delit.png").exists():
        raise FileNotFoundError(f"Expected albedo_delit.png in {d}")
    html_bytes = render_preview_html(d.name).encode("utf-8")

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html_bytes)))
                self.end_headers()
                self.wfile.write(html_bytes)
                return
            super().do_GET()

        def log_message(self, _fmt: str, *_args) -> None:
            return

    handler = functools.partial(_Handler, directory=str(d))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((host, port), handler) as httpd:
        url = f"http://{host}:{port}/"
        print(f"Serving hypbr PBR preview for '{d.name}' at {url} (Ctrl+C to stop)")
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped preview server.")

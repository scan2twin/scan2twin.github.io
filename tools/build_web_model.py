"""Convert the raw part meshes in res/model into one compact GLB for the web viewer.

- Streams the (very large) OBJ files, keeps positions + vertex colours.
- Decimates each part by vertex clustering to a size-dependent triangle budget.
- Bakes each part's rest transform (model.json, metres) into the vertices.
- Writes a quantized GLB (KHR_mesh_quantization: int16 positions, uint8 colours)
  plus a small joints JSON with the kinematic tree for animation.

Usage:  python tools/build_web_model.py   (from the repository root)
Only numpy is required.
"""
import json, os, struct, sys, math
import numpy as np

SRC = "res/model"
OUT_GLB = "res/web_model/plant.glb"
OUT_JSON = "res/web_model/plant.json"
BUDGET_SCALE = float(os.environ.get("BUDGET_SCALE", "1.0"))


def parse_obj(path):
    vs, fs = [], []
    with open(path, "r") as fh:
        while True:
            lines = fh.readlines(64 << 20)
            if not lines:
                break
            vtxt = [l[2:] for l in lines if l.startswith("v ")]
            ftxt = [l[2:] for l in lines if l.startswith("f ")]
            if vtxt:
                a = np.array(" ".join(vtxt).split(), dtype=np.float32)
                vs.append(a.reshape(len(vtxt), -1))
            if ftxt:
                toks = " ".join(ftxt).replace("/", " ").split()
                a = np.array(toks, dtype=np.int64)
                per = a.size // len(ftxt)
                a = a.reshape(len(ftxt), per)
                step = per // 3
                fs.append(a[:, ::step][:, :3] - 1)
    v = np.concatenate(vs)
    f = np.concatenate(fs)
    pos = v[:, :3]
    col = v[:, 3:6] if v.shape[1] >= 6 else np.full((len(v), 3), 0.6, np.float32)
    return pos, col, f


def cluster(pos, col, f, cell):
    q = np.floor(pos / cell).astype(np.int64)
    q -= q.min(0)
    key = (q[:, 0] * 1_000_003 + q[:, 1]) * 1_000_003 + q[:, 2]
    uniq, inv = np.unique(key, return_inverse=True)
    n = len(uniq)
    cnt = np.bincount(inv, minlength=n).astype(np.float64)
    P = np.stack([np.bincount(inv, pos[:, i], n) for i in range(3)], 1) / cnt[:, None]
    C = np.stack([np.bincount(inv, col[:, i], n) for i in range(3)], 1) / cnt[:, None]
    F = inv[f]
    ok = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
    F = F[ok]
    s = np.sort(F, 1)
    _, idx = np.unique(s, axis=0, return_index=True)
    F = F[np.sort(idx)]
    used = np.unique(F)
    remap = np.full(n, -1, np.int64)
    remap[used] = np.arange(len(used))
    return P[used], C[used], remap[F]


def decimate(pos, col, f, target):
    if len(f) <= target:
        return pos.astype(np.float64), col.astype(np.float64), f
    ext = np.linalg.norm(pos.max(0) - pos.min(0))
    lo, hi = ext / 5000, ext / 20
    best = None
    for _ in range(10):
        mid = math.sqrt(lo * hi)
        P, C, F = cluster(pos, col, f, mid)
        if len(F) > target:
            lo = mid
        else:
            hi = mid
            best = (P, C, F)
        if best is not None and 0.8 * target <= len(F) <= target:
            break
    if best is None:
        best = cluster(pos, col, f, hi)
    return best


def srgb_to_linear(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def walk(nodes, parent=None, out=None):
    for n in nodes:
        out.append((n, parent))
        walk(n.get("children", []), n["name"], out)
    return out


def main():
    model = json.load(open(os.path.join(SRC, "model.json")))
    flat = walk(model["roots"], None, [])

    gltf = {"asset": {"version": "2.0", "generator": "scan2twin build_web_model.py"},
            "extensionsUsed": ["KHR_mesh_quantization"],
            "extensionsRequired": ["KHR_mesh_quantization"],
            "scene": 0, "scenes": [{"nodes": []}], "nodes": [], "meshes": [],
            "accessors": [], "bufferViews": [], "materials": [
                {"name": "vc", "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1],
                 "metallicFactor": 0.0, "roughnessFactor": 0.85}}]}
    blob = bytearray()

    def add_view(data, target, stride=None):
        while len(blob) % 4:
            blob.append(0)
        bv = {"buffer": 0, "byteOffset": len(blob), "byteLength": len(data), "target": target}
        if stride:
            bv["byteStride"] = stride
        blob.extend(data)
        gltf["bufferViews"].append(bv)
        return len(gltf["bufferViews"]) - 1

    joints = []
    total_in = total_out = 0
    for n, parent in flat:
        entry = {"name": n["name"], "parent": parent, "mesh": False}
        kind = n.get("kind")
        if isinstance(kind, dict) and "Axis" in kind:
            entry["axis"] = kind["Axis"]
        joints.append(entry)
        mp = n.get("mesh")
        if not mp or not os.path.exists(os.path.join(SRC, mp)):
            continue
        pos, col, f = parse_obj(os.path.join(SRC, mp))
        target = int(min(len(f), (4000 + 55 * math.sqrt(len(f))) * BUDGET_SCALE))
        P, C, F = decimate(pos, col, f, target)
        total_in += len(f); total_out += len(F)
        print(f"{n['name']:24s} {len(f):>9d} -> {len(F):>7d} tris", flush=True)

        # bake rest transform (column-major 4x4), mm -> m
        M = np.array(n["transform"], np.float64).reshape(4, 4).T
        W = (P * 0.001) @ M[:3, :3].T + M[:3, 3]
        mn, mx = W.min(0), W.max(0)
        ctr = (mn + mx) / 2
        half = np.maximum((mx - mn) / 2, 1e-6)
        qp = np.round((W - ctr) / half * 32767).astype(np.int16)
        qp4 = np.zeros((len(qp), 4), np.int16); qp4[:, :3] = qp
        lc = np.round(srgb_to_linear(C) * 255).astype(np.uint8)
        c4 = np.full((len(lc), 4), 255, np.uint8); c4[:, :3] = lc
        idx = F.astype(np.uint16 if len(P) < 65535 else np.uint32).ravel()

        a0 = len(gltf["accessors"])
        gltf["accessors"] += [
            {"bufferView": add_view(qp4.tobytes(), 34962, 8), "componentType": 5122, "normalized": True,
             "count": len(qp), "type": "VEC3", "min": [-1, -1, -1], "max": [1, 1, 1]},
            {"bufferView": add_view(c4.tobytes(), 34962, 4), "componentType": 5121, "normalized": True,
             "count": len(c4), "type": "VEC4"},
            {"bufferView": add_view(idx.tobytes(), 34963), "componentType": 5123 if idx.dtype == np.uint16 else 5125,
             "count": int(idx.size), "type": "SCALAR"},
        ]
        gltf["meshes"].append({"name": n["name"], "primitives": [
            {"attributes": {"POSITION": a0, "COLOR_0": a0 + 1}, "indices": a0 + 2, "material": 0}]})
        gltf["nodes"].append({"name": n["name"], "mesh": len(gltf["meshes"]) - 1,
                              "translation": ctr.tolist(), "scale": half.tolist()})
        gltf["scenes"][0]["nodes"].append(len(gltf["nodes"]) - 1)
        entry["mesh"] = True

    while len(blob) % 4:
        blob.append(0)
    gltf["buffers"] = [{"byteLength": len(blob)}]
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    os.makedirs(os.path.dirname(OUT_GLB), exist_ok=True)
    with open(OUT_GLB, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(blob)))
        fh.write(struct.pack("<II", len(js), 0x4E4F534A)); fh.write(js)
        fh.write(struct.pack("<II", len(blob), 0x004E4942)); fh.write(blob)
    json.dump({"nodes": joints}, open(OUT_JSON, "w"), indent=1)
    print(f"total {total_in} -> {total_out} tris, {os.path.getsize(OUT_GLB)/1e6:.1f} MB")


if __name__ == "__main__":
    main()

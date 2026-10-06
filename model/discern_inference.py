"""discern_inference.py - reference backend for the Discern open-set re-ID model (copy into your backend).
pip install onnxruntime-gpu (or onnxruntime) numpy pillow
"""
import json, os
import numpy as np
from PIL import Image
import onnxruntime as ort


class Discern:
    def __init__(self, model_dir=".", providers=None):
        self.cfg = json.load(open(os.path.join(model_dir, "decision_config.json")))
        avail = ort.get_available_providers()
        prov = providers or [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        self.sess = ort.InferenceSession(os.path.join(model_dir, self.cfg["model"]["onnx"]), providers=prov)
        self.H, self.W = self.cfg["preprocess"]["input_size_hw"]
        self.rows = {}          # identity -> list of row indices into self.emb
        self.emb = np.zeros((0, self.cfg["model"]["embedding_dim"]), np.float32)

    # ---------- embedding ----------
    def _prep(self, img):
        a = np.asarray(img.convert("RGB").resize((self.W, self.H), Image.BILINEAR), dtype=np.float32) / 255.0
        return a.transpose(2, 0, 1)

    def embed(self, imgs):
        """imgs: list of PIL person crops -> (N, 2048) L2-normalised, flip-TTA."""
        x = np.stack([self._prep(i) for i in imgs])
        e1 = self.sess.run(None, {"images": x})[0]
        e2 = self.sess.run(None, {"images": np.ascontiguousarray(x[..., ::-1])})[0]
        e = e1 + e2
        return (e / np.linalg.norm(e, axis=1, keepdims=True)).astype(np.float32)

    # ---------- gallery ----------
    def add_embeddings(self, identity, emb):
        start = len(self.emb)
        self.emb = np.concatenate([self.emb, emb.astype(np.float32)])
        self.rows.setdefault(identity, []).extend(range(start, start + len(emb)))

    def enroll(self, identity, imgs):
        self.add_embeddings(identity, self.embed(imgs))

    def remove(self, identity):          # rebuild after delete
        keep = {k: v for k, v in self.rows.items() if k != identity}
        old = self.emb; self.emb = np.zeros((0, old.shape[1]), np.float32); self.rows = {}
        for k, v in keep.items(): self.add_embeddings(k, old[v])

    # ---------- open-set decision ----------
    def identify(self, img_or_emb, op=None):
        q = self.embed([img_or_emb])[0] if isinstance(img_or_emb, Image.Image) else img_or_emb
        names = list(self.rows)
        if not names:
            return dict(decision="reject", reason="empty gallery", identity=None, confidence=0.0)
        agg = self.cfg["matching"]["aggregation"]; k = 1 if agg == "max" else int(agg[3:])
        sims = self.emb @ q
        S = np.array([np.sort(sims[self.rows[n]])[::-1][:k].mean() for n in names])
        order = np.argsort(-S); s1 = float(S[order[0]]); n = len(names)
        s2 = float(S[order[1]]) if n > 1 else 0.0
        m = s1 - s2; z = (s1 - S.mean()) / (S.std() + 1e-6)
        cal_name = "s1_margin_z" if n >= self.cfg["matching"]["min_identities_full_calibrator"] else ("s1_margin" if n >= 2 else "s1")
        cal = self.cfg["calibrators"][cal_name]
        vals = {"s1": s1, "m": m, "z": z}
        x = (np.array([vals[f] for f in cal["features"]]) - np.array(cal["mean"])) / np.array(cal["scale"])
        conf = float(1 / (1 + np.exp(-(np.dot(cal["coef"], x) + cal["intercept"]))))
        op = op or self.cfg["default_operating_point"]; thr = cal["operating_points"][op]["threshold"]
        top = [dict(identity=names[i], score=float(S[i])) for i in order[:5]]
        accept = conf > thr
        return dict(decision="accept" if accept else "reject", identity=names[order[0]] if accept else None,
                    best_candidate=names[order[0]], confidence=conf, threshold=thr, calibrator=cal_name,
                    s1=s1, margin=m, z=float(z), operating_point=op, top5=top)


if __name__ == "__main__":      # replay the bundled demo
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    D = Discern(d); man = json.load(open(os.path.join(d, "demo", "demo_manifest.json")))
    for g in man["gallery"]:
        D.enroll(g["identity"], [Image.open(os.path.join(d, "demo", g["file"]))])
    ok = tot = fa = unk = 0
    for p in man["probes"]:
        r = D.identify(Image.open(os.path.join(d, "demo", p["file"])))
        if p["truth"] == "enrolled": tot += 1; ok += (r["identity"] == p["expected"])
        else: unk += 1; fa += (r["decision"] == "accept")
    print(f"known probes correctly accepted: {ok}/{tot} | look-alike strangers falsely accepted: {fa}/{unk}")

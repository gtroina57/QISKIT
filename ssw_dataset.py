"""SSW v1.0 recreated from the supplied 11 October 2026 specification.

The original companion generator was not found. This implementation explicitly
uses rng.integers for uniform configuration selection and sequential noise draws.
"""
import itertools
import json
from pathlib import Path
import numpy as np

SEED = 20261011
HELD_OUT = [(0, 0, 1, 1), (0, 1, 0, 1), (1, 0, 1, 0), (1, 1, 0, 0)]
SIZES = {"train": 2000, "val_iid": 400, "test_iid": 400, "test_ood": 400}


def clean_features(bits):
    a, b, c, d = np.asarray(bits, dtype=np.int8).T
    return np.stack([a, b, c, d, a ^ b, b ^ c, a & d,
                     np.where(c == 0, a ^ d, b & d)], axis=1).astype(np.float32)


def generate():
    rng = np.random.default_rng(SEED)
    iid = np.array([v for v in itertools.product((0, 1), repeat=4) if v not in HELD_OUT], dtype=np.int8)
    ood = np.array(HELD_OUT, dtype=np.int8)
    arrays = {}
    for split, count in SIZES.items():
        pool = ood if split == "test_ood" else iid
        bits = pool[rng.integers(0, len(pool), size=count)]
        clean = clean_features(bits)
        x = (clean + rng.normal(0, 0.05, size=clean.shape)).astype(np.float32)
        arrays.update({f"{split}_x": x, f"{split}_hidden_bits": bits,
                       f"{split}_x_clean": clean})
    return arrays


def validate(arrays):
    for split, count in SIZES.items():
        x, bits, clean = (arrays[f"{split}_{k}"] for k in ("x", "hidden_bits", "x_clean"))
        assert x.shape == clean.shape == (count, 8) and bits.shape == (count, 4)
        assert x.dtype == clean.dtype == np.float32 and bits.dtype == np.int8
        assert np.isfinite(x).all() and np.isin(bits, [0, 1]).all()
        # Independent scalar reference for every feature and contextual branch.
        expected = np.array([[a,b,c,d,int(a!=b),int(b!=c),int(a and d),
                              int(a!=d) if c==0 else int(b and d)]
                             for a,b,c,d in bits], dtype=np.float32)
        assert np.array_equal(clean, expected)
        patterns = set(map(tuple, bits.tolist()))
        assert patterns == set(HELD_OUT) if split == "test_ood" else len(patterns)==12 and patterns.isdisjoint(HELD_OUT)
        assert abs(float((x-clean).mean())) < 0.004
        assert abs(float((x-clean).std()) - 0.05) < 0.004
    regenerated = generate()
    assert all(np.array_equal(v, regenerated[k]) for k,v in arrays.items())
    return {"status": "passed", "checks": ["formulas", "dtypes/shapes", "configuration pools",
            "OOD separation", "noise statistics", "deterministic regeneration"]}


def main():
    root = Path(__file__).resolve().parent
    arrays = generate()
    checks = validate(arrays)
    np.savez_compressed(root / "ssw_v1.npz", **arrays)
    metadata = {"version": "1.0", "seed": SEED, "noise_std": 0.05, "split_sizes": SIZES,
                "held_out": HELD_OUT, "scaling": "none", "clipping": "none",
                "split_order": list(SIZES), "selection": "rng.integers; lexicographic IID pool; listed OOD pool",
                "provenance": "Recreated from supplied specification; original companion code not found",
                "validation": checks}
    (root / "ssw_v1_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

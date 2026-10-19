"""Stream one contiguous SKIPP'D train shard (~70k 1-minute frames, Mar-Sep 2017)
and cache it as small arrays: 32x32 RGB uint8 frames + PV + timestamps.

Cut on purpose: full dataset is 2.3 GB; one time-ordered shard is enough for a
CPU build and keeps the series contiguous (no random subsampling).
"""
import sys, io
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
from PIL import Image
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"; CACHE.mkdir(exist_ok=True)
SIZE = 32

def main():
    p = hf_hub_download("winkqw/SKIPPD", "data/train-00000-of-00005.parquet",
                        repo_type="dataset", local_dir=ROOT / "data_raw")
    f = pq.ParquetFile(p)
    imgs, pv, ts = [], [], []
    for b in f.iter_batches(batch_size=2000, columns=["image", "time", "pv"]):
        d = b.to_pydict()
        for im, t, v in zip(d["image"], d["time"], d["pv"]):
            a = Image.open(io.BytesIO(im["bytes"])).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
            imgs.append(np.asarray(a, dtype=np.uint8)); pv.append(v)
            ts.append(np.datetime64(t.replace(tzinfo=None), "s"))  # local Pacific wall-clock
        print(len(imgs), end="\r", file=sys.stderr)
    np.savez_compressed(CACHE / "skippd_slice.npz", img=np.stack(imgs),
                        pv=np.asarray(pv, np.float32), ts=np.asarray(ts))
    print("\nsaved", len(imgs), "frames")

if __name__ == "__main__":
    main()

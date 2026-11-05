"""Summarize measured near-view response; never infers a high-view Jacobian."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def analyze(directory):
    root=Path(directory)
    metadata=json.loads((root/"metadata.json").read_text(encoding="utf-8"))
    if metadata.get("source")!="hardware":
        raise ValueError("hardware logs required")
    pairs=[]
    samples=[]
    pose=None
    heights=[]
    with (root/"samples.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["event"]=="feedback" and row["tcp_x_mm"]:
                pose=np.array([float(row["tcp_x_mm"]),float(row["tcp_y_mm"])])
                z=float(row["tcp_z_mm"])
            if row["event"]=="vision" and row["phase"]=="servo_align" and pose is not None:
                samples.append((pose.copy(),np.array([float(row["error_u_px"]),float(row["error_v_px"])])))
                heights.append(z)
    for (p,e),(q,f) in zip(samples,samples[1:]):
        travel=float(np.linalg.norm(q-p))
        if travel>.2:
            pairs.append(float(np.linalg.norm(f-e))/travel)
    return {"run":root.name,"source":"hardware","phase":"servo_align",
        "height_range_mm":[min(heights),max(heights)] if heights else None,
        "pairs":len(pairs),"median_response_px_per_mm":float(np.median(pairs)) if pairs else None,
        "range_response_px_per_mm":[min(pairs),max(pairs)] if pairs else None,
        "note":"Near-view directional response only; not a high-view or full 2D J calibration"}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs",nargs="+")
    args=parser.parse_args()
    print(json.dumps([analyze(root) for root in args.runs],ensure_ascii=False,indent=2))

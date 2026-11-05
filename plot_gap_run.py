"""Offline, one-axis-per-file exports; never queries robot SDK or opens cameras."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def export(log_path, output=None):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    log_path = Path(log_path)
    raw = log_path.read_bytes()
    events = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    output = Path(output) if output else log_path.parent/"figures"
    output.mkdir(parents=True, exist_ok=True)
    samples = [e for e in events if e["event"] == "measurement"]
    if not samples:
        raise ValueError("no accepted observations to plot; inspect abort events")
    source = events[0]["source"]
    rows = [dict(sequence=e["sequence"], stage=e["stage"], error_px=e["error_px"],
                 estimated_mm=e["estimated_mm"], **{f"J{i+1}{j+1}": e["J"][i][j]
                                                     for i in range(2) for j in range(2)}) for e in samples]
    with (output/"samples.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paths = []
    for metric, unit in (("error_px", "px"), ("estimated_mm", "model-estimated mm"),
                         ("J11", "px/mm"), ("J12", "px/mm"), ("J21", "px/mm"), ("J22", "px/mm")):
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        for stage in ("coarse", "fine"):
            subset = [row for row in rows if row["stage"] == stage]
            if subset:
                ax.plot([row["sequence"] for row in subset], [row[metric] for row in subset], label=stage)
        ax.set(xlabel="Audit event sequence", ylabel=unit, title=f"{metric} | {source.upper()}")
        ax.legend()
        for extension in ("png", "svg", "pdf"):
            destination = output/f"{metric}.{extension}"
            fig.savefig(destination, dpi=180)
            paths.append(destination.name)
        plt.close(fig)
    manifest = dict(source=source, log_sha256=hashlib.sha256(raw).hexdigest(), files=paths,
                    note="Estimated mm is a model output, not measured assembly accuracy")
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log")
    parser.add_argument("--output")
    args = parser.parse_args()
    print(export(args.log, args.output).resolve())

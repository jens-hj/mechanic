"""Build a standalone viewer from the actual benchmark JSONL captures."""
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).parent
data = {}
colors = dict(sand="#dbc188", soil="#83624a", rock="#9aa0a5", iron="#946f63", graphite="#596370", cover="#78874f")
for name in ["clumps", *colors]:
    source = ROOT / f"{name}.jsonl.gz"
    if not source.exists():
        continue
    rows = [json.loads(line) for line in gzip.open(source, "rt")]
    frames, meshes, mesh_index = [], [], None
    for row in rows[:-1]:
        if "terrain" in row:
            meshes.append(row.pop("terrain"))
            mesh_index = len(meshes) - 1
        if "poses" not in row or row["tick"] % 3 != 0:
            continue
        row["terrain"] = mesh_index
        frames.append(row)
    data[name] = dict(frames=frames, meshes=meshes, summary=rows[-1], mining=name != "clumps", color=colors.get(name))
template = (ROOT / "replay-template.html").read_text()
(ROOT / "replay.html").write_text(template.replace("/*REPLAY_DATA*/{}", json.dumps(data, separators=(",", ":"))))
print("Built", ROOT / "replay.html")

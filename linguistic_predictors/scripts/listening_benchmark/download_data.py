"""Download selected files of OSF project 4m2st (Gehmacher et al. 2026), keeping the folder layout."""
import json, sys, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = "https://api.osf.io/v2/nodes/4m2st/files/osfstorage/"
OUT = Path(sys.argv[1])
MEG = set(sys.argv[2].split(","))          # participants to fetch, e.g. P3,P4

def ls(url):
    out, nxt = [], url + "?page%5Bsize%5D=100"
    while nxt:
        d = json.load(urllib.request.urlopen(nxt)); out += d["data"]; nxt = d["links"].get("next")
    return out

jobs = []
def walk(url, rel):
    for f in ls(url):
        a = f["attributes"]; path = rel / a["name"]
        if a["kind"] == "folder":
            if path.parts[:2] in {("data", "freesurfer_subjects"), ("data", "preproc_eye")}:
                continue
            walk(f["relationships"]["files"]["links"]["related"]["href"], path)
        elif path.parts[:2] == ("data", "parcellated_meg") and path.stem not in MEG:
            continue
        else:
            jobs.append((f["links"]["download"], OUT / path, a["size"]))

walk(ROOT, Path("."))
print(f"{len(jobs)} files, {sum(s for _, _, s in jobs) / 1e9:.2f} GB", flush=True)

def get(job):
    url, dest, size = job
    if dest.exists() and dest.stat().st_size == size:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)

with ThreadPoolExecutor(6) as ex:
    list(ex.map(get, jobs))
print("done", flush=True)

"""Run every test suite; exit non-zero if any fails.

    python tests/run_all.py          # offline suites + the real-model suite (skips if absent)
    python tests/run_all.py --offline
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
SUITES = ["test_offline.py", "test_lexicon_mfa.py", "test_mtrf_export.py", "test_real_gpt2.py"]

SKIP = 77   # a suite's "could not run" exit code

failed, skipped = [], []
for name in SUITES:
    if "--offline" in sys.argv and name == "test_real_gpt2.py":
        continue
    r = subprocess.run([sys.executable, str(HERE / name)], capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
    status = {0: "ok  ", SKIP: "SKIP"}.get(r.returncode, "FAIL")
    print(f"{name:24s} {status}  {last}")
    if r.returncode == SKIP:
        skipped.append(name)
    elif r.returncode != 0:
        failed.append(name)
        print(r.stdout[-2000:], r.stderr[-2000:], sep="\n")
print(f"\n{len(SUITES) - len(failed) - len(skipped)} passed, {len(failed)} failed, "
      f"{len(skipped)} skipped{(': ' + ', '.join(skipped)) if skipped else ''}")
sys.exit(1 if failed else 0)

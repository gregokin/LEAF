"""Resume the approved generic-only no-tools handoff with process receipts."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parent


def main():
    revision = int(sys.argv[1])
    stem = f"{revision:02d}"
    if (TASK / f"launch_{stem}.json").exists():
        raise RuntimeError("revision already exists; preserve evidence")
    previous = json.loads((TASK / "session_config.json").read_text())
    previous["argv"] = ["claude", "--safe-mode", "--print", "--verbose", "--output-format", "stream-json",
                        "--session-id" if revision == 1 else "--resume", previous["session_id"], "--effort", "low",
                        "--tools", "", "--permission-mode", "dontAsk", "--permission-prompts", "none",
                        "--strict-mcp-config", "--system-prompt",
                        "Return only the requested exact edits to the attached original generic numerical benchmark. No tools, local files, private context, external models or execution claims. Keep reasoning brief."]
    prompt_path = TASK / f"claude_prompt_{stem}.txt"
    prompt = prompt_path.read_text()
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    start = time.perf_counter()
    timed_out = False
    with (TASK / f"claude_{stem}.stdout.jsonl").open("w") as out, (TASK / f"claude_{stem}.stderr.log").open("w") as err:
        child = subprocess.Popen(previous["argv"], cwd=previous["cwd"], stdin=subprocess.PIPE,
                                 stdout=out, stderr=err, text=True, start_new_session=True)
        stat = Path(f"/proc/{child.pid}/stat").read_text()
        ticks = stat.rsplit(")", 1)[1].split()[19]
        launch = {"started_at_utc": started, "cwd": previous["cwd"], "argv": previous["argv"],
                  "session_id": previous["session_id"], "pid": child.pid, "proc_stat_at_launch": stat,
                  "prompt": str(prompt_path), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                  "tools": [], "safe_mode": True, "permission_bypass": False, "timeout_seconds": 900,
                  "submitted_content": "Generic-code corrections only; no private project files or outline."}
        (TASK / f"launch_{stem}.json").write_text(json.dumps(launch, indent=2) + "\n")
        try:
            child.communicate(prompt, timeout=900)
        except subprocess.TimeoutExpired:
            timed_out = True
            if Path(f"/proc/{child.pid}/stat").read_text().rsplit(")", 1)[1].split()[19] != ticks:
                raise RuntimeError("process identity changed; refusing signal")
            child.terminate()
            try:
                child.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                if Path(f"/proc/{child.pid}/stat").read_text().rsplit(")", 1)[1].split()[19] != ticks:
                    raise RuntimeError("process identity changed; refusing kill")
                child.kill()
                child.communicate(timeout=10)
    result = {"finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "returncode": child.returncode, "elapsed_seconds": time.perf_counter() - start,
              "timed_out": timed_out, "child_reaped": child.poll() is not None,
              "session_id": previous["session_id"]}
    (TASK / f"exit_{stem}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return child.returncode


if __name__ == "__main__":
    raise SystemExit(main())

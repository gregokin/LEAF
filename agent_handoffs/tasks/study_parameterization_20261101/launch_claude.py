"""Launch the authorized public-study candidate with ordinary bounded file tools."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import time

TASK = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    cfg = json.loads((TASK / "session.json").read_text())
    if (TASK / "launch_01.json").exists():
        raise FileExistsError("Preserve previous launches")
    cwd = Path(cfg["cwd"])
    public = TASK / "public_evidence"
    reads = [public / x for x in ["input_schemas.json", "image_projection.json", "zoom_calibration_data.txt", "peterson_figure_readings.json"]]
    rules = [f"Read(/{cwd}/**)"] + [f"Read(/{p})" for p in reads]
    for name in cfg["write_files"]:
        rules += [f"Write(/{cwd / name})", f"Edit(/{cwd / name})"]
    argv = ["claude", "--safe-mode", "--print", "--verbose", "--output-format", "stream-json",
            "--session-id", cfg["session_id"], "--effort", "high", "--tools", "Read,Edit,Write",
            "--allowedTools", *rules, "--permission-mode", "dontAsk", "--permission-prompts", "none",
            "--strict-mcp-config", "--system-prompt",
            "Implement only the bounded public-study analysis in the supplied prompt. Read only permitted public metadata and candidate files. No shell, outside context, execution claims or invented observations. Follow the embedded Codex-Claude scope and preserve all other files."]
    prompt = (TASK / "prompt_01.md").read_text()
    started = now()
    clock = time.perf_counter()
    timed_out = False
    with (TASK / "claude_01.stdout.jsonl").open("x") as out, (TASK / "claude_01.stderr.log").open("x") as err:
        child = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.PIPE, stdout=out, stderr=err, text=True, start_new_session=True)
        stat = Path(f"/proc/{child.pid}/stat").read_text()
        ticks = stat.rsplit(")", 1)[1].split()[19]
        (TASK / "launch_01.json").write_text(json.dumps({"argv": argv, "cwd": str(cwd), "session_id": cfg["session_id"],
            "pid": child.pid, "start_ticks": ticks, "proc_stat_at_launch": stat, "started_at_utc": started,
            "prompt_file": "prompt_01.md", "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "permission_bypass": False, "allowed_tools": ["Read", "Edit", "Write"], "read_metadata": [str(p) for p in reads],
            "write_files": cfg["write_files"], "submitted_context": "Public study metadata and bounded method/analysis contract only; no project brief, private data or upstream source."}, indent=2) + "\n")
        print("Claude candidate launched; PID", child.pid, "session", cfg["session_id"], flush=True)
        try:
            child.communicate(prompt, timeout=cfg["timeout_seconds"])
        except subprocess.TimeoutExpired:
            timed_out = True
            current = Path(f"/proc/{child.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
            if current != ticks:
                raise RuntimeError("Process start identity changed; refusing signal")
            child.terminate()
            try:
                child.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                current = Path(f"/proc/{child.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
                if current != ticks:
                    raise RuntimeError("Process start identity changed; refusing kill")
                child.kill()
                child.communicate(timeout=10)
    exit_record = {"returncode": child.returncode, "timed_out": timed_out, "child_reaped": child.poll() is not None,
                   "ended_at_utc": now(), "elapsed_seconds": time.perf_counter() - clock, "session_id": cfg["session_id"]}
    (TASK / "exit_01.json").write_text(json.dumps(exit_record, indent=2) + "\n")
    print(json.dumps(exit_record), flush=True)
    raise SystemExit(child.returncode or (1 if timed_out else 0))

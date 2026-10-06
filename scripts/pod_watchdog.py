"""Laptop-side watchdog for a RunPod training run: restarts the pod and resumes after a
crash, or rebuilds on a fresh pod from the latest HF checkpoint if the pod is gone.

    demo\\.venv-win\\Scripts\\python scripts\\pod_watchdog.py --pod <id> --run train_<ts>

Prints one line per event (for a Claude Monitor). Recovery cases:
  1. pod stopped (auto-stop after a crash), disk intact -> `pod start` + RESUME=1 (exact)
  2. pod gone / won't start -> new pod, setup, download newest step_N from HF,
     RESUME_FROM=<dir> (weights + data position; fresh optimizer)
Stops after --max-recoveries, or when the run's final weights are on HF.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEY = str(Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key")
ENV_FILE = Path.home() / ".runpod" / "hf_write.env"
RUNPOD_CFG = Path.home() / ".runpod" / "config.toml"
REPO = "RSD002/hypersteer-gemma2-2b-l20"
CREATE = ["--gpu-id", "NVIDIA A100-SXM4-80GB", "--cloud-type", "SECURE",
          "--image", "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
          "--volume-in-gb", "150", "--container-disk-in-gb", "50", "--ports", "22/tcp"]
W = "/workspace/hypersteer"
LAUNCH_ENV = ""  # e.g. "EPOCHS=3 CKPT_EVERY=9000", set from --launch-env


def say(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def sh(cmd, timeout=120):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def runpodctl(*args, timeout=120):
    code, out = sh(["runpodctl", *args], timeout)
    try:
        return code, json.loads(out[out.index("{"):] if "{" in out else out)
    except (ValueError, json.JSONDecodeError):
        return code, out


class Pod:
    def __init__(self, pod_id):
        self.id, self.ip, self.port = pod_id, None, None

    def info(self):
        code, d = runpodctl("pod", "get", self.id)
        if code or not isinstance(d, dict):
            return None
        ssh = d.get("ssh") or {}
        self.ip, self.port = ssh.get("ip", self.ip), ssh.get("port", self.port)
        return d

    def ssh(self, cmd, timeout=60):
        return sh(["ssh", "-n", "-i", KEY, "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                   "-o", "StrictHostKeyChecking=accept-new", "-p", str(self.port),
                   f"root@{self.ip}", cmd], timeout)

    def scp(self, src, dst, timeout=300):
        return sh(["scp", "-q", "-i", KEY, "-o", "BatchMode=yes", "-o",
                   "StrictHostKeyChecking=accept-new", "-P", str(self.port), str(src),
                   f"root@{self.ip}:{dst}"], timeout)

    def bg(self, cmd):
        """Start a detached command on the pod (doesn't hold the ssh session open)."""
        return self.ssh(f"nohup bash -c {json.dumps(cmd)} >/dev/null 2>&1 < /dev/null &")

    def wait_ssh(self, minutes=10):
        for _ in range(minutes * 4):
            d = self.info()
            if d and d.get("runtimeStatus") == "running" and self.port:
                if self.ssh("true", 20)[0] == 0:
                    return True
            time.sleep(15)
        return False


def hf_state(run):
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(REPO)
    final = f"{run}/train/HyperSteer_weight.safetensors" in files
    steps = sorted(int(f.split("step_")[1].split("/")[0]) for f in files
                   if f.startswith(f"{run}/train/checkpoints/step_") and f.endswith(".safetensors"))
    return final, steps


def training_alive(pod):
    code, out = pod.ssh(# [r]: keep pgrep from matching this ssh command line itself
        "pgrep -f '[r]unpod_train.sh' >/dev/null && echo ALIVE || echo DEAD")
    return None if code else "ALIVE" in out


def pod_listed(pod_id):
    code, out = sh(["runpodctl", "pod", "list", "--all"])
    return None if code else pod_id in out


def finished_but_not_uploaded(pod, run, mode):
    """Training exited 0 but the final HF upload failed (runpod_train.sh then skips the
    auto-stop): copy the weights to the laptop and stop the pod instead of retraining."""
    code, out = pod.ssh(f"grep -l 'Exit status: 0' $(ls -t {W}/logs/{mode}*.out) 2>/dev/null | head -1")
    if code or not out.strip():
        return False
    dst = ROOT / "assets" / "checkpoints" / "hf" / run / "train"
    dst.mkdir(parents=True, exist_ok=True)
    src = f"root@{pod.ip}:{W}/assets/checkpoints/{run}"
    opts = ["-q", "-i", KEY, "-o", "BatchMode=yes", "-P", str(pod.port)]
    ok = (sh(["scp", *opts, f"{src}/train/HyperSteer_weight.safetensors", str(dst)], 1800)[0] == 0
          and sh(["scp", *opts, f"{src}/config.yaml", str(dst.parent)], 120)[0] == 0)
    if ok:
        runpodctl("pod", "stop", pod.id)
        say(f"DONE: training finished but HF upload failed; final weights copied to {dst}; pod stopped")
    else:
        say("DONE?: training finished, HF upload failed, AND copying weights failed - pod left RUNNING")
    return True


def check_nan(pod, mode):
    _, out = pod.ssh(f"tail -n 50 $(ls -t {W}/logs/train_{mode}_*.log | head -1) | grep -ci \"loss_train/main': nan\"")
    return out.strip().isdigit() and int(out.strip()) > 0


def resume_same_pod(pod, mode):
    log = f"logs/{mode}_resume_{time.strftime('%H%M%S')}.out"  # keep the crashed run's log
    pod.bg(f"cd {W} && {LAUNCH_ENV} RESUME=1 bash scripts/runpod_train.sh {mode} > {log} 2>&1")
    time.sleep(90)
    code, out = pod.ssh(f"grep -m1 -E 'Resuming from|RESUME=1 but|Refusing' {W}/{log}")
    return "Resuming from" in out, out.strip()


def rebuild_on_new_pod(run, mode):
    final, steps = hf_state(run)
    if not steps:
        return None, "no checkpoint on HF to resume from"
    step = steps[-1]
    say(f"creating replacement pod (resume from HF step {step})")
    code, d = runpodctl("pod", "create", "--name", f"hypersteer-{mode}-r", *CREATE, "--wait",
                        timeout=900)
    if code or not isinstance(d, dict) or "id" not in d:
        return None, f"pod create failed: {str(d)[:300]}"
    pod = Pod(d["id"])
    say(f"replacement pod {pod.id} up; uploading code")
    if not pod.wait_ssh():
        return pod, "replacement pod ssh never came up"
    zip_path = ROOT / "hypersteer.zip"
    for src, dst in [(zip_path, "/workspace/"), (RUNPOD_CFG, "/root/.runpod/config.toml")]:
        if pod.scp(src, dst)[0]:
            return pod, f"scp {src.name} failed"
    pod.ssh(f"cd /workspace && python -m zipfile -e hypersteer.zip . && mkdir -p {W}/logs")
    if pod.scp(ENV_FILE, f"{W}/.env")[0]:
        return pod, "scp .env failed"
    say("running setup on replacement pod (~15 min)")
    pod.bg(f"cd {W} && bash scripts/runpod_setup.sh > logs/setup.out 2>&1")
    for _ in range(40):
        time.sleep(60)
        _, out = pod.ssh(f"tail -n 2 {W}/logs/setup.out; pgrep -f '[r]unpod_setup.sh' >/dev/null && echo RUNNING")
        if "Setup OK" in out:
            break
        if "RUNNING" not in out:
            return pod, f"setup failed: {out[-300:]}"
    else:
        return pod, "setup timed out"
    ckpt = f"assets/checkpoints/{run}/train/checkpoints/step_{step}"
    dl = (f"cd {W} && mkdir -p {ckpt} && .venv/bin/python -c \"from huggingface_hub import "
          f"hf_hub_download as d; import shutil; shutil.copy(d('{REPO}', "
          f"'{run}/train/checkpoints/step_{step}/HyperSteer_weight.safetensors'), "
          f"'{ckpt}/HyperSteer_weight.safetensors')\"")
    code, out = pod.ssh(f"set -a; . {W}/.env; set +a; {dl}", timeout=1200)
    if code:
        return pod, f"checkpoint download failed: {out[-300:]}"
    pod.bg(f"cd {W} && {LAUNCH_ENV} RESUME_FROM={ckpt} bash scripts/runpod_train.sh {mode} > logs/{mode}.out 2>&1")
    time.sleep(120)
    _, out = pod.ssh(f"grep -m1 -E 'Resuming from|Refusing' {W}/logs/{mode}.out")
    return pod, ("ok" if "Resuming from" in out else f"launch unclear: {out.strip()[:200]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pod", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--mode", default="large")
    p.add_argument("--max-recoveries", type=int, default=2)
    p.add_argument("--interval", type=int, default=300)
    p.add_argument("--launch-env", default="", help='env for relaunches, e.g. "EPOCHS=3"')
    a = p.parse_args()
    global LAUNCH_ENV
    LAUNCH_ENV = a.launch_env

    pod, recoveries, dead_checks, unknown_checks, nan_warned = Pod(a.pod), 0, 0, 0, False
    say(f"watching pod {pod.id} run {a.run} (max {a.max_recoveries} recoveries, env '{LAUNCH_ENV}')")
    while True:
        try:
            d = pod.info()
            if d is None:
                # API/network blips must not look like a vanished pod (-> rebuild)
                unknown_checks += 1
                listed = pod_listed(pod.id)
                if unknown_checks < 3 or listed is not False:
                    say(f"pod status unknown ({unknown_checks}x, listed={listed}); waiting")
                    time.sleep(a.interval)
                    continue
                status = "GONE"
            else:
                unknown_checks = 0
                status = d.get("desiredStatus", "UNKNOWN")
            if status == "RUNNING":
                alive = training_alive(pod)
                if alive and not nan_warned and check_nan(pod, a.mode):
                    say("NAN: loss is NaN in recent steps - training is likely diverging")
                    nan_warned = True
                dead_checks = 0 if alive in (True, None) else dead_checks + 1
                # Not alive while RUNNING: final upload or the 60 s auto-stop countdown;
                # only act if it stays that way for 3 checks
                if dead_checks < 3:
                    time.sleep(a.interval)
                    continue
                final, _ = hf_state(a.run)
                if final:
                    say("DONE: final weights on HF (pod still running - stop it)")
                    return
                if finished_but_not_uploaded(pod, a.run, a.mode):
                    return
                say("STALL: pod running but no training for 3 checks")
            else:
                final, steps = hf_state(a.run)
                if final:
                    say(f"DONE: final weights on HF; pod {status}")
                    return
                say(f"CRASH: pod {status}, newest HF checkpoint step {steps[-1] if steps else None}")

            if recoveries >= a.max_recoveries:
                say(f"GIVING UP: {recoveries} recoveries used; leaving it for the user")
                return
            recoveries += 1
            say(f"RECOVERY {recoveries}/{a.max_recoveries}")
            if status not in ("RUNNING", "GONE"):
                runpodctl("pod", "start", pod.id)
                ok = pod.wait_ssh()
                if ok:
                    resumed, out = resume_same_pod(pod, a.mode)
                    if resumed:
                        say(f"RESUMED on same pod: {out}")
                        dead_checks = 0
                        time.sleep(a.interval)
                        continue
                    say(f"same-pod resume failed: {out[:200]}")
                else:
                    say("pod start failed / ssh unreachable")
            elif status == "RUNNING":
                resumed, out = resume_same_pod(pod, a.mode)
                if resumed:
                    say(f"RESUMED on same pod: {out}")
                    dead_checks = 0
                    time.sleep(a.interval)
                    continue
            new_pod, msg = rebuild_on_new_pod(a.run, a.mode)
            if new_pod and msg == "ok":
                say(f"RESUMED on replacement pod {new_pod.id} (old pod {pod.id} left as-is)")
                pod, dead_checks = new_pod, 0
            else:
                say(f"REBUILD FAILED: {msg}" + (f" (pod {new_pod.id} still exists)" if new_pod else ""))
                return
        except Exception as e:  # keep watching through transient errors
            say(f"watchdog error (continuing): {type(e).__name__}: {e}")
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())

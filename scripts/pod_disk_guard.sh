#!/bin/bash
# Laptop-side hourly disk check for a RunPod pod's /workspace volume. Grows the volume by
# 50 GB if usage passes 80%, or it grew two checks in a row (monotonic, not just a
# checkpoint landing before the trainer prunes the oldest) and that trend projects past
# 95% within END_HOURS.
# A volume resize restarts the pod; pod_watchdog.py then resumes training (RESUME=1).
#   bash scripts/pod_disk_guard.sh <pod-id> [end-hours-from-now] [interval-s]
POD=$1; END_HOURS=${2:-7}; INTERVAL=${3:-3600}
KEY="$HOME/.runpod/ssh/runpodctl-ssh-key"
start=$(date +%s); prev=""; prev_rate=0; resized=0

say() { echo "$(date +%H:%M:%S) $*"; }

while true; do
    info=$(runpodctl pod get "$POD" 2>/dev/null)
    vol=$(echo "$info" | grep -o '"volumeInGb": [0-9]*' | grep -o '[0-9]*$')
    ip=$(echo "$info" | grep -o '"ip": "[0-9.]*"' | head -1 | grep -o '[0-9.]*')
    port=$(echo "$info" | grep -o '"port": [0-9]*' | head -1 | grep -o '[0-9]*$')
    used=$(ssh -n -i "$KEY" -o BatchMode=yes -o ConnectTimeout=15 -p "$port" "root@$ip" \
        'du -sB1G /workspace 2>/dev/null | cut -f1' 2>/dev/null)
    if [ -z "$used" ] || [ -z "$vol" ]; then
        say "DISK check failed (pod down or restarting?)"
    else
        pct=$((100 * used / vol))
        hours_left=$(( END_HOURS - ($(date +%s) - start) / 3600 )); [ $hours_left -lt 1 ] && hours_left=1
        rate=$([ -n "$prev" ] && echo $((used - prev)) || echo 0)  # GB per interval (~hour)
        rising=$([ $rate -gt 0 ] && [ $prev_rate -gt 0 ] && echo 1 || echo 0)
        proj=$((used + (rising ? (rate < prev_rate ? rate : prev_rate) : 0) * hours_left))
        say "DISK ${used}/${vol} GB (${pct}%), +${rate} GB/h, projected ${proj} GB at end"
        if [ $resized -lt 2 ] && { [ $pct -ge 80 ] || [ $((100 * proj / vol)) -ge 95 ]; }; then
            new=$((vol + 50))
            say "DISK RESIZE: ${vol} -> ${new} GB (pod restarts; watchdog resumes training)"
            runpodctl pod update "$POD" --volume-in-gb "$new" >/dev/null 2>&1 \
                && resized=$((resized + 1)) || say "DISK RESIZE FAILED"
        fi
        prev=$used; prev_rate=$rate
    fi
    sleep "$INTERVAL"
done

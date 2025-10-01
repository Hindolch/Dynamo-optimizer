import subprocess
import threading
import time
import os


def _gpu_worker(logfile, interval=5):
    """Background thread: logs nvidia-smi usage every `interval` seconds."""
    with open(logfile, "w") as f:
        f.write("time, gpu_util, mem_used(MB), mem_total(MB)\n")
        while True:
            try:
                result = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"]
                )
                line = result.decode("utf-8").strip()
                gpu_util, mem_used, mem_total = [x.strip() for x in line.split(",")]
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{timestamp}, {gpu_util}, {mem_used}, {mem_total}\n")
                f.flush()
            except Exception as e:
                f.write(f"Error: {e}\n")
                f.flush()
            time.sleep(interval)


def start_gpu_monitor(optimizer_name, logdir="results/gpu_logs", interval=5):
    """
    Launch GPU monitor in background for a specific optimizer.
    Logfile will be named gpu_log_<optimizer_name>.csv
    """
    os.makedirs(logdir, exist_ok=True)
    logfile = os.path.join(logdir, f"gpu_log_{optimizer_name.lower()}.csv")
    t = threading.Thread(target=_gpu_worker, args=(logfile, interval), daemon=True)
    t.start()
    print(f"[GPU Monitor] Logging to {logfile} every {interval}s")
    return t

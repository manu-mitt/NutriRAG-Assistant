import subprocess
import sys
import time

# Target command
cmd = ["npx", "--yes", "cloudflared", "tunnel", "--url", "http://localhost:8000"]

print("Starting Cloudflare Tunnel...", flush=True)

try:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
        shell=True
    )
except Exception as e:
    print(f"Failed to start subprocess: {e}", flush=True)
    sys.exit(1)

with open("tunnel_live.log", "w", encoding="utf-8") as f:
    # Read output line by line as it is generated
    for line in iter(process.stdout.readline, ""):
        f.write(line)
        f.flush()
        sys.stdout.write(line)
        sys.stdout.flush()

process.wait()
print(f"Tunnel process exited with code {process.returncode}", flush=True)

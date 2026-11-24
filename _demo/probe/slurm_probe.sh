#!/bin/bash
#SBATCH --job-name=prl-probe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=prl-probe-%j.log
#SBATCH --error=prl-probe-%j.log
#
# Milestone 0 probe, SLURM side: kernel-mode Tailscale in a batch job, then reach the
# Modal container started by modal_probe.py.
#
#   sbatch _demo/probe/slurm_probe.sh <remote tailnet ip> <http port> <random port> <sink port>
#
# Run from the prime-rl checkout (its .env holds TS_OAUTH_CLIENT_SECRET).
set -uo pipefail

REMOTE_IP=$1
HTTP_PORT=$2
RANDOM_PORT=$3
SINK_PORT=$4
TS_VERSION=1.104.7
TAG=tag:prime-rl

cd "$SLURM_SUBMIT_DIR"
source .env

step() { echo; echo "=== $*"; }

step "node"
hostname
uname -r

step "sudo -n inside a batch job"
sudo -n true && echo "OK" || { echo "FAIL: sudo -n true"; exit 1; }
ls -l /dev/net/tun

step "binaries (use ~/.local/bin if present, else the pinned download)"
if [ ! -x "$HOME/.local/bin/tailscale" ] || [ ! -x "$HOME/.local/bin/tailscaled" ]; then
    DEST="$HOME/.local/tailscale/$TS_VERSION"
    if [ ! -x "$DEST/tailscale" ] || [ ! -x "$DEST/tailscaled" ]; then
        TGZ="tailscale_${TS_VERSION}_amd64.tgz"
        mkdir -p "$DEST"
        curl -fsSL "https://pkgs.tailscale.com/stable/$TGZ" -o "$DEST/$TGZ"
        echo "$(curl -fsSL "https://pkgs.tailscale.com/stable/$TGZ.sha256")  $DEST/$TGZ" | sha256sum -c - || exit 1
        tar xzf "$DEST/$TGZ" -C "$DEST" --strip-components=1
    fi
    mkdir -p "$HOME/.local/bin"
    ln -sfn "$DEST/tailscale" "$HOME/.local/bin/tailscale"
    ln -sfn "$DEST/tailscaled" "$HOME/.local/bin/tailscaled"
fi
TAILSCALE="$HOME/.local/bin/tailscale"
TAILSCALED="$HOME/.local/bin/tailscaled"
"$TAILSCALE" version

step "host network state before"
cp /etc/resolv.conf /tmp/prl-probe-resolv.before
ip -br addr
ip route

STATE=$(mktemp -d /tmp/prl-ts-XXXXXX)
TUN=prlprobe
ts() { "$TAILSCALE" --socket="$STATE/sock" "$@"; }

teardown() {
    step "teardown"
    ts logout || true
    sudo -n pkill -f "tailscaled.*--statedir=$STATE" || true
    sleep 2
    sudo -n "$TAILSCALED" --cleanup --tun="$TUN" --statedir="$STATE" --socket="$STATE/sock" >/dev/null 2>&1 || true
    sudo -n rm -rf "$STATE"
    ip -br addr
    diff /tmp/prl-probe-resolv.before /etc/resolv.conf && echo "resolv.conf unchanged after teardown"
    pgrep -a tailscaled || echo "no tailscaled left"
}
trap teardown EXIT

step "tailscaled (kernel mode)"
sudo -n "$TAILSCALED" --tun="$TUN" --statedir="$STATE" --socket="$STATE/sock" --port=0 > "$STATE/tailscaled.log" 2>&1 &
for _ in $(seq 1 50); do [ -S "$STATE/sock" ] && break; sleep 0.2; done
[ -S "$STATE/sock" ] || { echo "FAIL: tailscaled did not start"; sudo -n cat "$STATE/tailscaled.log"; exit 1; }

step "tailscale up"
KEY_FILE="$STATE/authkey"
( umask 077; printf '%s' "${TS_OAUTH_CLIENT_SECRET}?ephemeral=true&preauthorized=true" > "$KEY_FILE" )
sudo -n "$TAILSCALE" --socket="$STATE/sock" up --auth-key="file:$KEY_FILE" --advertise-tags="$TAG" \
    --hostname=prl-probe-t0 --operator="$USER" --accept-dns=false --netfilter-mode=off
UP_CODE=$?
rm -f "$KEY_FILE"
[ "$UP_CODE" -eq 0 ] || { echo "FAIL: tailscale up"; sudo -n tail -30 "$STATE/tailscaled.log"; exit 1; }
echo "tailnet ip: $(ts ip -4)"
ts status

step "host network state with tailscale up"
ip -br addr
ip rule
ip route show table 52
diff /tmp/prl-probe-resolv.before /etc/resolv.conf && echo "resolv.conf unchanged"

step "tailscale ping (direct vs DERP)"
ts ping -c 5 "$REMOTE_IP"
ts status | grep -E "prl-probe-i0|$REMOTE_IP"

step "inbound TCP to the Modal container by tailnet IP"
curl -sS -m 20 -o /dev/null -w "fixed port $HTTP_PORT: http %{http_code} in %{time_total}s\n" "http://$REMOTE_IP:$HTTP_PORT/"
curl -sS -m 20 -o /dev/null -w "random port $RANDOM_PORT: http %{http_code} in %{time_total}s\n" "http://$REMOTE_IP:$RANDOM_PORT/"

step "throughput home -> remote (python streams; iperf3: $(command -v iperf3 || echo "not installed here"))"
python3 - "$REMOTE_IP" "$SINK_PORT" <<'PY'
import socket, sys, threading, time

host, port = sys.argv[1], int(sys.argv[2])
chunk = bytes(1 << 20)


def run(streams: int, seconds: float = 10.0) -> None:
    sent = [0] * streams
    deadline = time.monotonic() + seconds

    def send(i: int) -> None:
        with socket.create_connection((host, port)) as conn:
            while time.monotonic() < deadline:
                sent[i] += conn.send(chunk)

    start = time.monotonic()
    threads = [threading.Thread(target=send, args=(i,)) for i in range(streams)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.monotonic() - start
    print(f"{streams} stream(s): {sum(sent) * 8 / elapsed / 1e9:.3f} Gbit/s ({sum(sent) / 2**20:.0f} MiB in {elapsed:.1f}s)")


for streams in (1, 8, 32):
    run(streams)
PY

step "tailscale ping after traffic"
ts ping -c 3 "$REMOTE_IP"
ts netcheck

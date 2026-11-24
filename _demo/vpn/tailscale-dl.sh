#!/usr/bin/env bash
set -exo pipefail
TS_VERSION=1.104.7
ARCH=amd64 # check with `uname -m`: x86_64 → amd64, aarch64 → arm64
DEST=$HOME/.local/tailscale/$TS_VERSION
mkdir -p "$DEST" && cd "$DEST"
curl -fsSLO "https://pkgs.tailscale.com/stable/tailscale_${TS_VERSION}_${ARCH}.tgz"
curl -fsSL "https://pkgs.tailscale.com/stable/tailscale_${TS_VERSION}_${ARCH}.tgz.sha256" -o tgz.sha256
echo "$(cat tgz.sha256)  tailscale_${TS_VERSION}_${ARCH}.tgz" | sha256sum -c -
tar xzf "tailscale_${TS_VERSION}_${ARCH}.tgz" --strip-components=1
./tailscale version

# Milestone 0 probe results (2026-11-23)

Setup: one Modal CPU container (`poc-h200-cu129`, `us-central`, landed on GCP) running `modal_probe.py`, and SLURM
batch job 19 on `prime-nebius-puku-h200-gpu-001` running `slurm_probe.sh`. Both joined the tailnet as
`tag:prime-rl` with the OAuth client secret. The Nebius Object Storage part was not run (no credentials yet).

## What worked

| Check | Result |
|---|---|
| `sudo -n true` inside a batch job | OK |
| Static binaries from `~/.local/tailscale/1.104.7`, symlinked into `~/.local/bin` | OK (the symlinks now exist) |
| Kernel-mode `tailscaled --tun=prlprobe --port=0` + `tailscale up --operator=$USER` | OK |
| `/etc/resolv.conf` with `--accept-dns=false` | unchanged while up and after teardown |
| Inbound TCP to the Modal container by tailnet IP, fixed port and OS-assigned port | HTTP 200 on both |
| Teardown (`logout`, kill, `tailscaled --cleanup`) | interface gone, no daemon left |
| Modal userspace `tailscaled`, join, ephemeral node | OK |

## What did not

| Check | Result |
|---|---|
| Direct tailnet path | **never established**; every ping `via DERP(ord)`, 24 ms |
| Throughput home -> remote over the tailnet | **~0.025 Gbit/s** on 1, 8 and 32 streams |

Cause: the SLURM node is a pod behind a symmetric NAT (`netcheck`: `MappingVariesByDestIP: true`, egress
`204.12.188.68`), so Tailscale cannot hole-punch. The Modal side's mapping is endpoint-independent
(`MappingVariesByDestIP: false`), which is not enough by itself.

## Alternative measured: Modal raw-TCP tunnel

`modal.forward(port, unencrypted=True)` on the same container, dialled outbound from the same SLURM node, no
tailnet involved:

| Streams | Gbit/s |
|---|---|
| 1 | 0.585 |
| 8 | 2.857 |
| 32 | 9.516 |

Rates are measured at the sender over 10 s (`streams.py`), so they include whatever the tunnel proxy buffers.

## Other observations

- The SLURM nodes have no `iperf3`; the probe used Python streams. The Modal image has it.
- `tailscale up` needs `--accept-dns=false` to leave the resolver alone (the default is to accept DNS).
- `--netfilter-mode=off` was used on SLURM; outbound connections from the node work without Tailscale's iptables
  chains.
- `tailscale status` on the SLURM node listed every device of the tailnet, personal ones included, and installed a
  route for each. The policy lets `tag:prime-rl` see them.

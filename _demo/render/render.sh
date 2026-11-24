#!/bin/bash
# Render rl.sbatch for existing multi-node examples into $1 (for byte-identical diffs).
# Usage: _demo/render/render.sh <out_dir>
set -u
OUT=$1; mkdir -p "$OUT"
WORK=/tmp/prl-render
for toml in $(grep -ln 'type = "multi_node"' -r examples configs | sort); do
    name=$(echo "$toml" | tr '/' '_' | sed 's/\.toml$//')
    rm -rf "$WORK"
    if uv run rl @ "$toml" --dry-run --output-dir "$WORK" --run.name render > "$OUT/$name.log" 2>&1; then
        sb=$(find "$WORK" -name 'rl.sbatch' | head -1)
        if [ -n "$sb" ]; then cp "$sb" "$OUT/$name.sbatch"; echo "ok   $toml"; else echo "nosb $toml"; fi
    else
        echo "FAIL $toml"
    fi
done

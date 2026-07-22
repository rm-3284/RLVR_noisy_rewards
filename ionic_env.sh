# Sourceable environment for RLVR_noisy_rewards on the CS ionic cluster (user: aw2418, account: allcs).
# Mirror of della_env.sh but with ionic paths: home is only 5GB, so everything lives on /n/fs/scratch.
# Usage: source ionic_env.sh
export SCR=/n/fs/scratch/aw2418
export ROOT="$SCR/RLVR_noisy_rewards"

# uv binary in home, all caches on the 5.8TB scratch (home is only 5GB)
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$SCR/uv_cache"

# HF + torch + xdg caches on scratch. Compute nodes HAVE internet, so datasets download at runtime.
export HF_HOME="$SCR/hf_cache"
export TORCH_HOME="$SCR/torch"
export XDG_CACHE_HOME="$SCR/xdg_cache"
export TMPDIR="$SCR/tmp"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMPDIR" 2>/dev/null || true

# Don't let user site-packages shadow the venv (same fix as Della)
export PYTHONPATH="$ROOT"
export PYTHONNOUSERSITE=1

cd "$ROOT" 2>/dev/null || true

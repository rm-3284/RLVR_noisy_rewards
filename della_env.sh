# Sourceable environment for RLVR_noisy_rewards on Della (user: aw2418, account: griffith).
# Ported from the collaborator's Neuronic setup (/n/fs/vision-mix/rm4411) to GRIFFITHS scratch.
# Usage: source della_env.sh   (used by both interactive prep on login/vis nodes and the SLURM jobscript)

export SCR=/scratch/gpfs/GRIFFITHS/aw2418
export ROOT="$SCR/RLVR_noisy_rewards"

# uv: binary in home, cache on scratch (home is only 50GB and .venv caches are large)
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$SCR/uv-cache"

# Hugging Face: cache models + datasets on scratch so air-gapped compute nodes load them offline
export HF_HOME="$SCR/huggingface"
[ -f "$HOME/.cache/huggingface/token" ] && export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
export TORCH_HOME="$SCR/torch"
export XDG_CACHE_HOME="$SCR/cache"

# Do NOT prepend $ROOT:$PYTHONPATH blindly; user site-packages can shadow the venv. Mirror collaborator's fix.
export PYTHONPATH="$ROOT"
export PYTHONNOUSERSITE=1

cd "$ROOT" 2>/dev/null || true

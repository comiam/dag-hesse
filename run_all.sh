#!/usr/bin/env bash
set -euo pipefail

SESSION="dag_hesse"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="source ${SCRIPT_DIR}/.venv/bin/activate"
CD="cd ${SCRIPT_DIR}"

# ======================================================================== #
# Experiment definitions (name -> python main.py arguments, without --gpu) #
# ======================================================================== #

ALL_EXPS=(
	exp1
	exp1b
	exp2
	exp3
	exp3_ln
	exp1_appendix
	exp2_unified
	exp3_appendix
	exp4
	exp5
	exp6
	exp7
)

declare -A EXP_CMD
declare -A EXP_DESC

# --- Main figures (exact) ---

EXP_CMD[exp1]="python main.py exp1 \
    --depths 8 10 12 \
    --width 64 \
    --epochs 50 \
    --seeds 42 43 44 45 46 \
    --hessian-mode exact \
    --hessian-batch-size 32 \
    --output-dir results/exp1"
EXP_DESC[exp1]="Plain vs ResNet, exact (depths 8 10 12, width 64)"

EXP_CMD[exp1b]="python main.py exp1b \
    --depths 8 12 16 \
    --width 64 \
    --epochs 50 \
    --seeds 42 43 44 45 46 \
    --hessian-mode exact \
    --hessian-batch-size 32 \
    --output-dir results/exp1b"
EXP_DESC[exp1b]="Spectral normalization verification (rho <= 1)"

EXP_CMD[exp2]="python main.py exp2 \
    --depths 6 8 \
    --dataset cifar100 \
    --base-width 256 \
    --bottleneck-widths 4 8 16 32 64 128 256 \
    --epochs 50 \
    --seeds 42 43 44 45 46 \
    --hessian-mode exact \
    --hessian-batch-size 32 \
    --output-dir results/exp2"
EXP_DESC[exp2]="Bottleneck ablation, exact (CIFAR-100, base-width 256)"

EXP_CMD[exp3]="python main.py exp3 \
    --depth 6 \
    --width 64 \
    --epochs 20 \
    --seeds 42 43 44 45 46 \
    --hessian-mode exact \
    --hessian-batch-size 16 \
    --output-dir results/exp3"
EXP_DESC[exp3]="Activation GN-Gap, exact (depth 6, width 64)"

EXP_CMD[exp3_ln]="python main.py exp3 \
    --use-layernorm \
    --depth 6 \
    --width 64 \
    --epochs 20 \
    --seeds 42 43 44 45 46 \
    --hessian-mode exact \
    --hessian-batch-size 16 \
    --output-dir results/exp3_ln"
EXP_DESC[exp3_ln]="Exp3 control run with LayerNorm=True"

# --- Appendix (stochastic) ---

EXP_CMD[exp1_appendix]="python main.py exp1 \
    --depths 16 32 \
    --width 128 \
    --epochs 50 \
    --seeds 42 43 44 45 46 \
    --hessian-mode stochastic \
    --n-probes 100 \
    --hessian-batch-size 64 \
    --output-dir results/exp1_appendix"
EXP_DESC[exp1_appendix]="Exp1 appendix: stochastic (depths 16 32, width 128)"

EXP_CMD[exp2_unified]="python main.py exp2 \
    --depths 6 8 \
    --dataset cifar100 \
    --base-width 512 \
    --bottleneck-widths 4 8 16 32 64 128 256 512 \
    --epochs 50 \
    --seeds 42 43 44 45 46 \
    --hessian-mode stochastic \
    --n-probes 100 \
    --hessian-batch-size 64 \
    --output-dir results/exp2_unified"
EXP_DESC[exp2_unified]="Exp2 unified sweep d_u=4..512 (base-width 512, stochastic)"

EXP_CMD[exp3_appendix]="python main.py exp3 \
    --depth 6 \
    --width 64 \
    --epochs 20 \
    --seeds 42 43 44 45 46 \
    --hessian-mode stochastic \
    --n-probes 100 \
    --hessian-batch-size 16 \
    --output-dir results/exp3_appendix"
EXP_DESC[exp3_appendix]="Exp3 appendix: stochastic (depth 6, width 64)"

EXP_CMD[exp4]="python main.py exp4 \
    --branch-depths 1 2 3 \
    --width 32 \
    --merge-types sum cat \
    --activations relu silu \
    --epochs 50 \
    --seeds 42 43 44 45 46 \
    --hessian-mode exact \
    --hessian-batch-size 32 \
    --output-dir results/exp4"
EXP_DESC[exp4]="Diamond MLP, exact (branch depths 1 2 3, width 32)"

EXP_CMD[exp5]="python main.py exp5 \
    --d-model 16 \
    --seq-len 8 \
    --n-train 2048 \
    --n-val 512 \
    --noise-std 0.1 \
    --epochs 30 \
    --seeds 42 43 44 45 46 \
    --hessian-mode exact \
    --hessian-batch-size 64 \
    --output-dir results/exp5"
EXP_DESC[exp5]="Toy-Attention vs ReLU-MLP (H^T_{Q,K} != 0 verification)"

EXP_CMD[exp6]="python main.py exp6 \
    --activations relu silu \
    --lr 0.1 \
    --batch-size 128 \
    --epochs 100 \
    --seeds 42 43 44 45 46 \
    --hessian-mode stochastic \
    --n-probes 30 \
    --hessian-batch-size 32 \
    --output-dir results/exp6"
EXP_DESC[exp6]="ResNet-18 conv: GN-Gap + R/C/D decay (CIFAR-10, stochastic)"

# --- Exp7: COUPLE-FAC overlay repair (downloads Stanford Cars, full ResNet-50 finetune) ---

EXP_CMD[exp7]="python main.py exp7 \
    --profile b1 \
    --output-dir results/exp7"
EXP_DESC[exp7]="COUPLE-FAC overlay finetune on Stanford Cars (repair, B1 headline)"

# ======================================================================
# GPU detection
# ======================================================================

detect_gpus() {
	# Returns available GPU indices, one per line.
	# Priority: $CUDA_VISIBLE_DEVICES > nvidia-smi > empty (CPU fallback).
	if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
		tr ',' '\n' <<<"$CUDA_VISIBLE_DEVICES"
	elif command -v nvidia-smi &>/dev/null; then
		nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null || true
	fi
}

# ====================================================================== #
# Usage / help                                                           #
# ====================================================================== #

usage() {
	cat <<'EOF'
Usage: run_all.sh [OPTIONS] [EXPERIMENT [--gpu N]]

Modes:
  (no arguments)            Run ALL experiments in a tmux session.
                            GPUs are assigned round-robin from detected/specified set.
  EXPERIMENT                Run a single experiment in the current terminal.
                            Uses first detected GPU (or CPU if none).
  EXPERIMENT --gpu N        Run a single experiment on GPU N.
  --all                     Run all experiments in tmux (same as no arguments).

Options:
  --gpus IDS    Comma-separated GPU indices for round-robin (e.g. --gpus 0,2).
                Overrides auto-detection. Applies to --all / default mode.
  --list        List available experiments and exit.
  --help, -h    Show this help and exit.

Examples:
  ./run_all.sh                          # all experiments, auto-detect GPUs
  ./run_all.sh --gpus 0,1               # all experiments on GPUs 0 and 1
  ./run_all.sh exp1                     # single experiment, first GPU
  ./run_all.sh exp1 --gpu 2             # single experiment on GPU 2
  make exp1 GPU=2                       # same via Makefile

Available experiments:
EOF
	for name in "${ALL_EXPS[@]}"; do
		printf "  %-18s %s\n" "$name" "${EXP_DESC[$name]}"
	done
}

list_experiments() {
	for name in "${ALL_EXPS[@]}"; do
		printf "%-18s %s\n" "$name" "${EXP_DESC[$name]}"
	done
}

# ====================================================================== #
# Run helpers                                                            #
# ====================================================================== #

run_single() {
	local name="$1"
	local gpu_arg="$2" # "" or "--gpu N"

	if [[ -z "${EXP_CMD[$name]+x}" ]]; then
		echo "Error: unknown experiment '$name'" >&2
		echo "Run '$0 --list' to see available experiments." >&2
		exit 1
	fi

	echo "=== Running $name ${gpu_arg:+(GPU: ${gpu_arg#--gpu })} ==="
	cd "$SCRIPT_DIR"
	# shellcheck disable=SC1090
	source "${SCRIPT_DIR}/.venv/bin/activate"
	# shellcheck disable=SC2086
	exec ${EXP_CMD[$name]} $gpu_arg
}

run_all_tmux() {
	local -a gpu_ids=("$@")
	local n_gpus=${#gpu_ids[@]}

	tmux kill-session -t "$SESSION" 2>/dev/null || true

	local first=1
	local i=0
	for name in "${ALL_EXPS[@]}"; do
		local gpu_idx=""
		if ((n_gpus > 0)); then
			gpu_idx="--gpu ${gpu_ids[$((i % n_gpus))]}"
		fi

		if ((first)); then
			tmux new-session -d -s "$SESSION" -n "$name"
			first=0
		else
			tmux new-window -t "$SESSION" -n "$name"
		fi
		tmux send-keys -t "$SESSION:$name" \
			"$VENV && $CD && ${EXP_CMD[$name]} $gpu_idx" Enter

		((i++)) || true
	done

	echo "Started tmux session '$SESSION' with ${#ALL_EXPS[@]} windows."
	if ((n_gpus > 0)); then
		echo "GPUs: ${gpu_ids[*]} (round-robin)"
	else
		echo "GPUs: none detected (CPU mode)"
	fi
	echo "Attach: tmux attach -t $SESSION"
}

# ====================================================================== #
# CLI parsing                                                            #
# ====================================================================== #

MODE="" # "single", "all", "list", "help"
EXPERIMENT=""
GPU_ARG=""       # "--gpu N" for single mode
GPUS_OVERRIDE="" # comma-separated, for --all mode

while (($#)); do
	case "$1" in
	--help | -h)
		usage
		exit 0
		;;
	--list)
		list_experiments
		exit 0
		;;
	--all)
		MODE="all"
		shift
		;;
	--gpus)
		GPUS_OVERRIDE="$2"
		shift 2
		;;
	--gpu)
		GPU_ARG="--gpu $2"
		shift 2
		;;
	-*)
		echo "Unknown option: $1" >&2
		usage >&2
		exit 1
		;;
	*)
		if [[ -z "$EXPERIMENT" ]]; then
			EXPERIMENT="$1"
			MODE="single"
		else
			echo "Unexpected argument: $1" >&2
			usage >&2
			exit 1
		fi
		shift
		;;
	esac
done

# Default mode: run all in tmux
if [[ -z "$MODE" ]]; then
	MODE="all"
fi

case "$MODE" in
single)
	if [[ -z "$GPU_ARG" ]]; then
		# Use first detected GPU
		first_gpu=$(detect_gpus | head -n1)
		if [[ -n "$first_gpu" ]]; then
			GPU_ARG="--gpu $first_gpu"
		fi
	fi
	run_single "$EXPERIMENT" "$GPU_ARG"
	;;
all)
	# Build GPU list
	if [[ -n "$GPUS_OVERRIDE" ]]; then
		IFS=',' read -ra GPU_IDS <<<"$GPUS_OVERRIDE"
	else
		mapfile -t GPU_IDS < <(detect_gpus)
	fi
	run_all_tmux "${GPU_IDS[@]}"
	;;
esac

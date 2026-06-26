.PHONY: help install lint format typecheck test check clean \
       experiments exp1 exp1b exp2 exp3 exp3_ln \
       exp1_appendix exp2_unified exp3_appendix exp4 exp5 exp6 exp7

GPU ?=

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Dev
# ---------------------------------------------------------------------------

install: ## Install project + dev deps via uv
	uv sync

lint: ## Run ruff + flake8
	uv run ruff check .
	uv run flake8 . --exclude=.venv

format: ## Format with black + ruff --fix
	uv run black .
	uv run ruff check --fix .

typecheck: ## Run mypy
	uv run mypy . --exclude '.venv'

test: ## Run pytest
	uv run pytest hessian/tests/ -v

check: lint typecheck test ## Run lint + typecheck + test

# ---------------------------------------------------------------------------
# Experiments  (GPU=N to select GPU, e.g. make exp1 GPU=2)
# ---------------------------------------------------------------------------

experiments: ## Run ALL experiments via tmux (round-robin GPUs)
	bash run_all.sh --all

exp1: ## Plain vs ResNet decay (exact)
	bash run_all.sh exp1 $(if $(GPU),--gpu $(GPU))

exp1b: ## Spectral normalization verification (exact)
	bash run_all.sh exp1b $(if $(GPU),--gpu $(GPU))

exp2: ## Bottleneck ablation (exact, CIFAR-100)
	bash run_all.sh exp2 $(if $(GPU),--gpu $(GPU))

exp3: ## Activation GN-Gap (exact)
	bash run_all.sh exp3 $(if $(GPU),--gpu $(GPU))

exp3_ln: ## Exp3 + LayerNorm control
	bash run_all.sh exp3_ln $(if $(GPU),--gpu $(GPU))

exp1_appendix: ## Exp1 appendix (stochastic, deep)
	bash run_all.sh exp1_appendix $(if $(GPU),--gpu $(GPU))

exp2_unified: ## Exp2 unified sweep (stochastic)
	bash run_all.sh exp2_unified $(if $(GPU),--gpu $(GPU))

exp3_appendix: ## Exp3 appendix (stochastic)
	bash run_all.sh exp3_appendix $(if $(GPU),--gpu $(GPU))

exp4: ## Diamond MLP tensor term (exact)
	bash run_all.sh exp4 $(if $(GPU),--gpu $(GPU))

exp5: ## Toy-Attention vs ReLU-MLP (exact)
	bash run_all.sh exp5 $(if $(GPU),--gpu $(GPU))

exp6: ## ResNet-18 conv GN-Gap + decay (stochastic)
	bash run_all.sh exp6 $(if $(GPU),--gpu $(GPU))

exp7: ## COUPLE-FAC overlay finetune on Stanford Cars (repair)
	bash run_all.sh exp7 $(if $(GPU),--gpu $(GPU))

# ---------------------------------------------------------------------------

clean: ## Remove caches
	find . -type d -name __pycache__   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache   -exec rm -rf {} + 2>/dev/null || true

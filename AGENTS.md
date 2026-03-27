# AGENTS.md

This is the short operational guide for coding agents working in this repository. Prefer the current filesystem and code over stale planning text.

## Project snapshot

This repo currently implements a calo-only Panda-style self-distillation pipeline on ColliderML Release 1.

- runtime path: `calo_hits` only
- shared config: `config/default.yaml`
- current training defaults: batch size 8, BF16 mixed precision, flash attention disabled by default
- optional flash backend selection exists when flash is enabled: `torch` or `flash_attn`

## Source-of-truth docs

- `README.md`: project overview and common workflows
- `markdown/WORKFLOWS.md`: runtime details and user-facing local commands
- `markdown/PLAN.md`: current roadmap
- `markdown/HPC.md`: cluster-specific notes and SLURM usage

## Operating rules

- Read relevant files before editing.
- Prefer current repo state over aspirational or stale docs.
- Keep the current runtime path calo-only unless the user explicitly asks for a different scope.
- Use `Panda_repo/` as a reference implementation, not as the main runtime path.
- Do not run long downloads or long GPU jobs interactively.
- Prefer the checked-in SLURM jobs for heavyweight work.
- Leave generated outputs in `runs/` and `diagnostics/` uncommitted unless the user explicitly asks otherwise.
- Do not touch unrelated submodule changes in `Panda_repo/`.

## Cluster notes

- Load `uv` with `module load uv` before using `uv` in a session.
- You only need to run `module load uv` once at the beginning of the session.
- The documented HF cache is `/mnt/ceph/users/ewulff/data/hf`.
- Runtime SLURM jobs source `slurm/load_env.sh`.
- The checked-in short and medium training jobs target single-GPU `a100-40gb` nodes.
- Setup, smoke-test, and long-train jobs currently target `h100` nodes.

## Documentation maintenance

- Keep `AGENTS.md` short and operational.
- Keep all project markdown files except `README.md` and `AGENTS.md` under `markdown/`.
- Put the user-facing overview in `README.md`.
- Put runtime details and local command examples in `markdown/WORKFLOWS.md`.
- Put cluster specifics in `markdown/HPC.md`.
- Put roadmap items in `markdown/PLAN.md`.

---
name: colab-operator
description: Operate Google Colab runtimes through the installed colab CLI for WallZero training campaigns. Use when authenticating a Colab account, running the scripts/colab_*.py pipeline on Colab GPUs, managing Colab sessions and files, installing remote packages, retrieving training output bundles, or inspecting Colab execution logs. Do not use for ordinary local Python execution or browser-only notebook editing.
---

# Google Colab CLI Operator (WallZero)

Use the local Colab CLI to run explicitly requested workloads on the user's Google Colab account. Treat runtime allocation as metered external state and session shutdown as part of task completion.

## Local installation

- Prefer `colab` from `PATH`; this Mac currently resolves it to `/Users/isaaczhang/.local/bin/colab`.
- Run `colab version` when behavior may be version-dependent. Version 0.6.0 was installed when this skill was created.
- The CLI is the integration surface. It does not expose an MCP server subcommand.
- If a permission prompt blocks `~/.config/colab-cli` writes or Google network access, retry only the exact Colab command with scoped approval. Prefer a reusable approval for `/Users/isaaczhang/.local/bin/colab`; do not weaken general permissions.

## Authenticate the correct account

Use explicit OAuth2 commands in this environment:

```bash
colab --auth=oauth2 whoami
colab --auth=oauth2 sessions
```

The installed 0.6.0 executable defaults to OAuth2 even though older bundled README and skill text may say ADC. OAuth2 uses a bundled public client when `~/.colab-cli-oauth-config.json` is absent.

On first use, the CLI prints an authorization URL and waits for a code:

1. Keep the command's PTY open.
2. Ask the user to open the printed URL, select the Google account that owns the intended Colab Pro subscription, approve access, and return the displayed authorization code.
3. Send only that code to the waiting process.
4. Re-run `colab --auth=oauth2 whoami` and verify the email is the intended identity.
5. Run `colab --auth=oauth2 sessions` as the read-only API check.

The refresh token is cached at `~/.config/colab-cli/token.json`. Never read, print, copy, or commit this file. Colab Pro entitlement follows the authenticated Google identity; there is no separate Pro key to configure.

Do not use `colab auth` to fix CLI login. That command injects separate GCP credentials into an already running VM. Do not switch to ADC unless the user requests it and `gcloud` is installed with the required Colab scopes.

## WallZero training pipeline

This project trains a 9x9 Quoridor AlphaZero engine on Colab. The `scripts/` directory holds the remote-side pipeline; the campaign scripts assume an A100 (`device_name="cuda"`) and the `colab` preset, and write durable state to `/content/wallzero-output`:

| Script | Purpose |
| --- | --- |
| `colab_bootstrap.py` | Install an uploaded source archive (`/content/wallzero-source.tar.gz`) into the runtime |
| `build_colab_payload.py` | Local fallback: build a self-extracting payload script when file upload is blocked |
| `colab_validate.py` | A100 correctness and throughput probe — run before any metered training |
| `colab_chunk.py` | Generate one durable self-play shard |
| `colab_round.py` | Train and arena-gate one candidate from durable replay shards |
| `colab_iteration.py` | Run exactly one additional resumable training iteration |
| `colab_train.py` | General non-interactive entry point (`--preset`, `--iterations`, `--output`) |
| `colab_bundle.py` | Bundle `/content/wallzero-output` into `/content/wallzero-output.tar.gz` for download |
| `colab_restore.py` | Restore a previously downloaded bundle uploaded back to `/content` |

Standard campaign cycle on a persistent session:

1. Tar the local source, upload it to `/content/wallzero-source.tar.gz`, and `exec` `colab_bootstrap.py`. If upload fails, run `build_colab_payload.py` locally and `exec` the generated `/tmp/wallzero-colab-chunk.py` instead.
2. Resuming a campaign: upload the prior `wallzero-output.tar.gz` and `exec` `colab_restore.py`.
3. `exec` `colab_validate.py` and confirm the reported GPU, CUDA, and bf16 status before spending compute on training.
4. Run the requested workload (`colab_chunk.py`, `colab_round.py`, `colab_iteration.py`, or `colab_train.py`). Kernel state persists between `exec` calls, but durable progress lives only in `/content/wallzero-output` (`run-state.json` tracks completed iterations).
5. Always `exec` `colab_bundle.py` and download `/content/wallzero-output.tar.gz` locally (keep bundles under `artifacts/`) before stopping the session — the VM's filesystem is lost on stop.

## Choose a session workflow

Prefer one-shot execution when possible:

```bash
colab --auth=oauth2 run script.py
colab --auth=oauth2 run --gpu T4 script.py
```

`colab run` allocates a fresh VM, executes the script, and stops the VM unless `--keep` is supplied. Avoid `--keep` unless continued inspection is explicitly needed.

Use a persistent named session for iterative work — the normal case for WallZero campaigns, since bootstrap, validation, training, and bundling are separate `exec` calls against one VM:

```bash
colab --auth=oauth2 new -s wallzero
colab --auth=oauth2 exec -s wallzero -f scripts/colab_validate.py
colab --auth=oauth2 status -s wallzero
colab --auth=oauth2 stop -s wallzero
```

Before allocating:

- Run `colab --auth=oauth2 sessions` and avoid name collisions.
- State the requested hardware and session name. If the user did not explicitly request allocation, obtain confirmation because it consumes compute units.
- Do not assume accelerator entitlement or availability. Campaign scripts expect an A100, but confirm with the user before requesting one; for non-training tasks prefer CPU.
- Always pass `-s <name>` for persistent sessions.

After allocation:

- Remember that kernel state persists between `exec` calls.
- Use absolute remote paths under `/content`.
- Stop the named session as soon as the task is complete, including after failures. If a command is interrupted, inspect `sessions` and release any orphaned assignment.

## Execute and move files

Prefer file-based, non-interactive execution:

```bash
colab --auth=oauth2 exec -s wallzero -f scripts/colab_chunk.py
colab --auth=oauth2 install -s wallzero -r requirements.txt
colab --auth=oauth2 upload -s wallzero /tmp/wallzero-source.tar.gz /content/wallzero-source.tar.gz
colab --auth=oauth2 download -s wallzero /content/wallzero-output.tar.gz ./artifacts/wallzero-output.tar.gz
```

Notebook execution writes `<basename>_output.ipynb` beside the local input. Use `--output-image <path>` when plots must be captured at a predictable path.

Avoid unpiped `repl`, `console`, `auth`, and `drivemount` in autonomous runs because they require live terminal or browser interaction. For batch Python use `exec`; for a necessary shell command, pipe input into `console` and expect terminal-control bytes.

## Diagnose and recover

- `colab --auth=oauth2 status -s <name>`: inspect hardware and busy state.
- `colab --auth=oauth2 log -s <name> -n 20`: inspect recent structured events.
- `colab --auth=oauth2 log -s <name> -o summary.ipynb`: preserve a notebook record.
- `colab --auth=oauth2 restart-kernel -s <name>`: reset a wedged kernel without releasing the VM.
- On a stale or missing session, run `sessions`, then create a new named session if needed. Durable campaign state survives only as a downloaded bundle — restore it with `colab_restore.py` on the new VM.
- On an identity or scope error, run `whoami`. Do not remove or replace cached credentials without explicit user approval.

## Completion checklist

1. Confirm the output bundle was downloaded locally (or expected outputs otherwise retrieved) — training progress on the VM is lost at stop.
2. Stop every session allocated for the task unless the user explicitly requested it remain active.
3. Run `sessions` to verify no accidental runtime remains.
4. Report hardware used, output locations, and whether any explicitly retained session is still consuming compute units.

# isaac-pc GTX 1080 wedge incident log

Signature of a wedge (all incidents so far): `nvidia-smi` reports 100% GPU
utilization but power draw sits at ~60-63W (healthy self-play or training on
this card draws 140-198W). The evaluation server process spins at ~100% CPU
inside a CUDA synchronization call; all self-play workers block in
`unix_stream_data_wait` pipe reads; no Xid appears in the kernel log.
Unrecoverable from Python — only process death clears it.

Environment: GTX 1080 (Pascal, sm_61), driver 580.173.02 (final branch with
Pascal support), kernel 7.0.0-28-generic, torch 2.6.0+cu124. Community
reports implicate exactly this kernel-7 + driver-580 + Pascal combination in
no-Xid hangs.

## Incident 1 — 2026-07-25, chunk 82 (192 games)

Wedged partway through a multi-hour chunk; discovered after ~4h of fake
"100% utilization". Killed the process group manually. Response: rebuilt
`gen_loop.sh` with a 3h `timeout --kill-after` cap plus a power watchdog
(<80W for 15 consecutive minutes → kill process group, sweep `spawn_main`
orphans, restart), cut chunks to 96 games, enabled persistence mode and GPU
runtime PM `on`, staged `pcie_aspm=off pcie_port_pm=off` in GRUB (inactive
until reboot).

## Incident 2 — 2026-07-25 23:09, chunk 82 retry (96 games)

Wedged within ~1 minute of chunk start, immediately after the r32 training
round (56 min at 167-198W) finished on the same GPU. Verified at 23:13-23:21:
server pid at 99.9% CPU, all 12 workers in `unix_stream_data_wait`, flat
61-63W for 12+ minutes, no Xid. New facts vs incident 1: a wedge can strike
at chunk start (not only after hours), and both wedges followed sustained
heavy GPU use, consistent with driver-state corruption rather than a
workload bug. r32 artifacts were pulled to the Mac before any recovery
action.

## Incidents 3-4 and root cause — 2026-07-25 23:24-23:47

The watchdog fired correctly at 23:24 (first live test) but the restarted
chunk wedged again immediately, and so did the next attempt after a full
reboot with `pcie_aspm=off` active — eliminating accumulated driver state as
the cause. Incident 4 also wedged at 88W, *above* the original 80W watchdog
threshold (post-boot persistence mode raises idle draw), so the threshold is
now 110W.

`py-spy dump --native` on the live wedge showed the evaluation-server thread
spinning in `sched_yield` inside `libcuda.so.580.173.02`, launching
`cublasLtTSTMatmul` — a **bfloat16** cuBLASLt matmul (`T` = bf16 in cuBLAS
naming) whose `cuLaunchKernel` never returns.

Root cause: `TorchEvaluator` hardcoded `torch.autocast(dtype=torch.bfloat16)`
for all CUDA inference. Pascal has no native bf16; torch emulates it (the
same trap caught in training on 2026-07-25, 9.6x slower), and driver 580's
emulated-bf16 cublasLt kernels can hang at launch on sm_61. This is why the
A100 (native bf16) and the Mac (MPS, fp32) never wedged.

Fix: `network.py` now gates inference autocast on compute capability >= 8,
mirroring the training-side gate. Self-play on Pascal runs fp32. Verified
23:48-23:55: sustained 176-205W at 99% utilization, workers actively
consuming CPU. Side effect: the 1080's prior ~1.5-2 pos/s self-play
throughput was measured under bf16 emulation and is expected to improve
substantially.

## Escalation ladder

1. Power watchdog kill + restart (automatic, ≤15 min lost per wedge).
2. Reboot the node — clears wedged driver state and activates the staged
   `pcie_aspm=off pcie_port_pm=off` kernel parameters.
3. If wedges recur with ASPM off: boot the older installed kernel
   6.14.0-37-generic (GRUB default switch) — the strongest community-backed
   mitigation for this hang class.
4. If still recurring: pin an older 5xx-series driver branch.

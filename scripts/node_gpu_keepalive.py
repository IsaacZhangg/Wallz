#!/usr/bin/env python3
"""Keep the GPU in the P2 compute state at all times on the generation node.

Latch prevention (2026-07-27): the driver-580 P2 clock latch (core offset
silently ignored, pinned at stock 1670 MHz) has struck at 5 of 5
train<->generate transitions -- exactly when the card briefly drops to an
idle P-state between jobs. Precedent: the SETI@home community's keepP2
utility, built because Pascal cards misbehave at compute-load boundaries.
A trivial kernel 10x/second keeps a compute context active so the card
never leaves P2 and the latch's trigger condition never occurs. GPU cost
is microseconds per tick; VRAM cost is the CUDA context (~400 MiB, vs
~4 GiB free at training's peak). Runs supervised as
wallzero-keepalive.service; the gen-loop watchdogs ignore it (their load
threshold is 110 W, far above what this draws on its own).
"""

import time

import torch

torch.cuda.init()
a = torch.ones((64, 64), device="cuda")
print("keepalive: CUDA context up, ticking at 10 Hz", flush=True)
while True:
    (a @ a).sum().item()  # tiny kernel; .item() syncs so work really runs
    time.sleep(0.1)

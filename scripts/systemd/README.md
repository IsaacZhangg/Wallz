# Training-node service templates

The `.service.example` files describe the recorded Linux deployment. Copy each needed
template to a local `.service` file and replace every `@NODE_USER@` and `@NODE_HOME@`
placeholder with the service account's username and absolute home directory. These
local `.service` copies are ignored by Git.

For example, copy `wallzero-gen.service.example` to `wallzero-gen.service`, then edit
the copy. Use the same account and home directory in all units. The templates expect
the checkout and Python environment under `@NODE_HOME@/wallzero/` and the following
deployment copies in that directory:

| Repository script               | Deployment filename |
| ------------------------------- | ------------------- |
| `scripts/node_gen_loop.sh`      | `gen_loop.sh`       |
| `scripts/node_flywheel.sh`      | `flywheel.sh`       |
| `scripts/node_boot_prep.sh`     | `boot_prep.sh`      |
| `scripts/node_gpu_canary.py`    | `gpu_canary.py`     |
| `scripts/node_gpu_oc.sh`        | `gpu-oc.sh`         |
| `scripts/node_gpu_keepalive.py` | `gpu_keepalive.py`  |

Review the scripts' GPU settings, watchdogs, reboot behavior, and `sudo` requirements
before installing any unit. The prep service applies GPU clock offsets, and the
keepalive service remains a documented failed experiment. The templates do not enable
services or configure access to the machine.

After configuring a copy, validate it with `systemd-analyze verify` on the intended
Linux host before installing it under `/etc/systemd/system/`. Preserve the unit names
so the generation, training, and keepalive dependencies on `wallzero-prep.service`
continue to resolve.

# Configuration

There is one model configuration: **moe.yaml**.
There is one training configuration: **training.yaml** (currently 20,000 updates).

Edit these files directly. Train with `python scripts/train.py --restart`, then
serve with `python scripts/serve.py`. Both workflows use `checkpoints/moe`.
See the root README for setup, data, resume, and demo commands.

The checkpoint stores the resolved configuration used to train it. Editing YAML
does not change an already-loaded model. Existing reports and backups preserve
historical settings; they are not additional active presets.

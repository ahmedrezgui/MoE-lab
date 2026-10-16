"""Launch the Gradio story playground on localhost."""

import argparse

import torch

from moelab.serving.demo import DEFAULT_CHECKPOINT, DemoModel
from moelab.serving.ui import build_ui

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args()
    if a.threads < 1 or not 1 <= a.port <= 65535:
        p.error("Use positive threads and a valid port.")
    torch.set_num_threads(a.threads)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    try:
        engine = DemoModel(a.checkpoint, device)
    except (ValueError, FileNotFoundError) as error:
        p.error(str(error))
    build_ui(engine).launch(
        server_name="127.0.0.1", server_port=a.port, share=False, inbrowser=False, show_error=False
    )

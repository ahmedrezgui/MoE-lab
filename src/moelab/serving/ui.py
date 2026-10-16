"""Local Gradio playground for a small from-scratch language model."""

from moelab.serving.demo import EXAMPLE_PROMPTS


def build_ui(engine):
    import gradio as gr

    cfg = engine.model.cfg
    with gr.Blocks(title="MoE Lab | Story playground", analytics_enabled=False) as app:
        gr.Markdown(
            "# MoE Lab\n### A tiny model. A window into how it works.\n"
            "Continue a short story with a language model trained from scratch. "
            "**This is an architecture experiment, not a chat assistant.**"
        )
        gr.Markdown(
            f"**{engine.info['total_parameters']:,} parameters** · "
            f"**{cfg.moe.n_experts if cfg.moe else 1} experts** · "
            f"**{cfg.max_seq_len}-token context** · **{engine.device}**"
        )
        with gr.Tab("Playground"):
            with gr.Row():
                with gr.Column(scale=1):
                    prompt = gr.Textbox(
                        label="Start a story",
                        lines=5,
                        value=EXAMPLE_PROMPTS[0],
                        placeholder="Once upon a time...",
                    )
                    gr.Examples([[p] for p in EXAMPLE_PROMPTS], inputs=[prompt])
                    count = gr.Slider(1, 256, value=64, step=1, label="Maximum new tokens")
                    temperature = gr.Slider(
                        0,
                        2,
                        value=0.8,
                        step=0.05,
                        label="Temperature",
                        info="0 = greedy. Higher values increase randomness.",
                    )
                    with gr.Accordion("Sampling settings", open=False):
                        top_k = gr.Slider(
                            0,
                            engine.tokenizer.vocab_size,
                            value=min(40, engine.tokenizer.vocab_size),
                            step=1,
                            label="Top-k",
                            info="Keep this many candidates; 0 disables filtering.",
                        )
                        top_p = gr.Slider(
                            0.05,
                            1,
                            value=0.95,
                            step=0.05,
                            label="Top-p",
                            info="Keep the smallest candidate set reaching this probability mass.",
                        )
                        seed = gr.Number(
                            value=42,
                            precision=0,
                            label="Sampling seed",
                            minimum=0,
                            maximum=2**32 - 1,
                        )
                    go = gr.Button("Continue story", variant="primary")
                with gr.Column(scale=1):
                    output = gr.Textbox(label="Prompt + continuation", lines=14, interactive=False)
                    status = gr.Markdown(
                        "Ready. Generate a continuation to see token counts and stopping reason."
                    )
                    with gr.Accordion("Generation details", open=False):
                        details = gr.JSON(label="Measured generation")
            gr.Markdown(
                "Outputs may repeat, drift or contain malformed text. "
                "A token can be a byte or a learned byte sequence. "
                "Generation stops on EOS or the token limit; longer outputs use a sliding context."
            )
        with gr.Tab("About this model"):
            gr.Markdown(
                "### Small by design\n"
                "The default checkpoint uses 4 experts with top-2 routing, trained for "
                "200 updates on 1,000 TinyStories documents. It is not instruction-tuned.\n\n"
                "Use **MODEL_CARD.md** in the repository for the training protocol, "
                "measured comparisons, fixed-prompt samples and limitations. "
                "The checkpoint is selected at launch with `--checkpoint`."
            )
            gr.JSON(value=engine.info, label="Loaded checkpoint")

        def respond(prompt, count, temperature, top_k, top_p, seed):
            try:
                result = engine.complete(
                    prompt, int(count), temperature, int(top_k), top_p, int(seed)
                )
            except (TypeError, OverflowError) as error:
                raise gr.Error(
                    "Enter a valid integer seed and numeric generation settings."
                ) from error
            except ValueError as error:
                raise gr.Error(str(error)) from error
            except RuntimeError as error:
                raise gr.Error(
                    "Generation failed. Check the terminal; try fewer tokens or launch on CPU."
                ) from error
            summary = (
                f"**{result['generated_tokens']} new tokens** · "
                f"{result['generation_seconds']:.2f} s | "
                f"stopped on **{result['finish_reason']}** | "
                f"{result['prompt_tokens']} prompt tokens"
            )
            return result["text"], summary, result

        go.click(
            respond,
            [prompt, count, temperature, top_k, top_p, seed],
            [output, status, details],
            api_name="generate",
            concurrency_limit=1,
        )
    return app.queue(max_size=8, default_concurrency_limit=1)

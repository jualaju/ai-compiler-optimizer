"""Interactive AI Compiler Optimization demo."""

import argparse
import ast
from datetime import datetime

from demo_ui import DemoUI


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="open raw compiler details",
    )
    args = parser.parse_args()

    ui = DemoUI()
    ui.intro()

    # Choose what actually enters the compiler.
    ui.console.print()
    ui.console.print("[bold cyan]MODEL INPUT[/]")
    ui.console.print("[1] Manual Tensor")
    ui.console.print("[2] Trained PyTorch MLP")
    ui.console.print()

    choice = ui.console.input(
        "[cyan]Select model[/] [dim](2 = Trained MLP)[/]: "
    ).strip() or "2"

    if choice not in {"1", "2"}:
        choice = "2"

    # Manual mode gets the original interactive inputs.
    if choice == "1":
        x = ui.numbers("Tensor", "2 -1 3 4")
        scale = ui.numbers("Scale", "1.5", scalar=True)
        bias = ui.numbers("Bias", "0.5", scalar=True)
        model_mode = "manual"
        model_name = "MANUAL AFFINE + RELU"
    else:
        x = None
        scale = None
        bias = None
        model_mode = "trained_mlp"
        model_name = "TRAINED PYTORCH MLP"

    ui.console.clear()
    trace = []

    def record(tag, message):
        trace.append(
            (
                datetime.now().strftime("%H:%M:%S"),
                tag,
                message,
            )
        )

    with ui.console.status(
        "[cyan]Compiling model and validating CPU semantics…[/]",
        spinner="dots",
    ):
        import torch

        from compiler import (
            AffineReLU,
            capture,
            fuse,
            generate_triton,
            interpret,
        )
        from runtime import gpu_unavailable_reason

        if model_mode == "manual":
            # ---------------------------------------------------------
            # EXISTING MANUAL DEMO
            # ---------------------------------------------------------
            inputs = tuple(
                torch.tensor(v, dtype=torch.float32)
                for v in (x, scale, bias)
            )

            model = AffineReLU()

            record("MODEL", "Manual Affine + ReLU model selected")
            before = capture(model, inputs)

            record("GRAPH", "FX graph captured")
            record("IR", "Lowered to custom IR")

            after = fuse(before)

            fused = any(
                op.kind == "fused_affine_relu"
                for op in after.ops
            )

            record(
                "FUSE",
                "MUL → ADD → ReLU pattern detected"
                if fused
                else "No fusion pattern found",
            )

            count = lambda graph: sum(
                op.kind != "input" for op in graph.ops
            )

            record(
                "FUSE",
                f"{count(before)} ops → {count(after)} fused op"
                if fused
                else "Graph preserved",
            )

            path = generate_triton(after)
            source = path.read_text()
            ast.parse(source)

            record(
                "CODE",
                "Triton source generated; syntax verified",
            )

            expected = model(*inputs)
            original = interpret(before, *inputs)[0]
            optimized = interpret(after, *inputs)[0]

            torch.testing.assert_close(original, expected)
            torch.testing.assert_close(optimized, expected)
            torch.testing.assert_close(original, optimized)

            record(
                "CHECK",
                "CPU numerical equivalence verified ✓",
            )

            display_inputs = inputs
            model_summary = "User-provided tensor → Affine + ReLU"

            debug = (
                f"FX GRAPH\n{torch.fx.symbolic_trace(model).graph}\n\n"
                f"CUSTOM IR / BEFORE\n{before}\n\n"
                f"CUSTOM IR / AFTER\n{after}\n\n"
                f"GENERATED SOURCE\n{path}\n{source}"
            ) if args.debug else None

        else:
            # ---------------------------------------------------------
            # TRAINED PYTORCH MLP DEMO
            # ---------------------------------------------------------
            from examples.trained_mlp import TinyMLP

            torch.manual_seed(7)
            torch.set_num_threads(1)

            # Tiny synthetic training dataset.
            train_x = torch.linspace(
                -2.0, 2.0, 128
            ).reshape(-1, 1)

            target = (
                0.6 * train_x
                + 0.8 * torch.relu(train_x - 0.25)
                - 0.2
            )

            model = TinyMLP()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=0.03,
            )

            initial_loss = torch.nn.functional.mse_loss(
                model(train_x),
                target,
            ).item()

            model.train()

            for _ in range(250):
                optimizer.zero_grad()

                loss = torch.nn.functional.mse_loss(
                    model(train_x),
                    target,
                )

                loss.backward()
                optimizer.step()

            model.eval()

            with torch.no_grad():
                final_loss = torch.nn.functional.mse_loss(
                    model(train_x),
                    target,
                ).item()

            record(
                "MODEL",
                "Trained PyTorch MLP selected",
            )

            record(
                "TRAIN",
                f"250 steps • MSE {initial_loss:.5f} → {final_loss:.5f}",
            )

            # Validation samples.
            validation_x = torch.linspace(
                -2.25,
                2.25,
                32,
            ).reshape(-1, 1)

            # The compiler currently understands elementwise
            # multiply → add → ReLU, so compile the trained
            # hidden layer of the MLP.
            expanded_x = validation_x.expand(
                -1,
                8,
            ).contiguous()

            learned_scale = (
                model.hidden_weight.detach()
                .expand_as(expanded_x)
                .contiguous()
            )

            learned_bias = (
                model.hidden_bias.detach()
                .expand_as(expanded_x)
                .contiguous()
            )

            compile_inputs = (
                expanded_x,
                learned_scale,
                learned_bias,
            )

            display_inputs = (
                validation_x,
                model.hidden_weight.detach(),
                model.hidden_bias.detach(),
            )

            before = capture(
                model.hidden,
                compile_inputs,
            )

            record(
                "GRAPH",
                "Trained hidden layer captured with FX",
            )

            record(
                "IR",
                "Lowered trained operations to custom IR",
            )

            after = fuse(before)

            fused = any(
                op.kind == "fused_affine_relu"
                for op in after.ops
            )

            record(
                "FUSE",
                "MUL → ADD → ReLU pattern detected"
                if fused
                else "No fusion pattern found",
            )

            count = lambda graph: sum(
                op.kind != "input" for op in graph.ops
            )

            record(
                "FUSE",
                f"{count(before)} ops → {count(after)} fused op"
                if fused
                else "Graph preserved",
            )

            path = generate_triton(
                after,
                "generated/trained_mlp_kernel.py",
            )

            source = path.read_text()
            ast.parse(source)

            record(
                "CODE",
                "Triton kernel generated; syntax verified",
            )

            # CPU semantic validation.
            with torch.no_grad():
                expected_hidden = model.hidden(
                    validation_x,
                    model.hidden_weight,
                    model.hidden_bias,
                )

                original = interpret(
                    before,
                    *compile_inputs,
                )[0]

                optimized = interpret(
                    after,
                    *compile_inputs,
                )[0]

                torch.testing.assert_close(
                    original,
                    expected_hidden,
                )

                torch.testing.assert_close(
                    optimized,
                    expected_hidden,
                )

                torch.testing.assert_close(
                    original,
                    optimized,
                )

                # Verify that the compiled hidden representation
                # still produces the same full MLP prediction.
                expected_output = model(validation_x)
                optimized_output = model.readout(optimized)

                torch.testing.assert_close(
                    optimized_output,
                    expected_output,
                )

            record(
                "CHECK",
                "Trained MLP numerical equivalence verified ✓",
            )

            model_summary = (
                "Trained MLP: 1 → 8 ReLU → 1"
            )

            debug = (
                f"MODEL\n{model_summary}\n"
                f"Training MSE: {initial_loss:.6f} → "
                f"{final_loss:.6f}\n\n"
                f"FX GRAPH\n"
                f"{torch.fx.symbolic_trace(model.hidden).graph}\n\n"
                f"CUSTOM IR / BEFORE\n{before}\n\n"
                f"CUSTOM IR / AFTER\n{after}\n\n"
                f"GENERATED SOURCE\n{path}\n{source}"
            ) if args.debug else None

        reason = gpu_unavailable_reason()

        record(
            "GPU",
            "Execution unavailable"
            if reason
            else "Execution not requested; CPU demo",
        )

    ui.show(
        dict(
            before=before,
            after=after,
            inputs=display_inputs,
            original=original,
            optimized=optimized,
            source=source,
            path=path,
            trace=trace,
            reason=reason,
            fused=fused,
            debug=debug,
            model_name=model_name,
            model_mode=model_mode,
            model_summary=model_summary,
        ),
        debug=args.debug,
    )


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        pass
    except Exception as exc:
        from rich.console import Console

        Console(stderr=True).print(
            f"Compiler demo failed: {exc}",
            style="bold red",
            markup=False,
        )

        raise SystemExit(1)

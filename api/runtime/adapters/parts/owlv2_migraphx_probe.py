"""Synthetic only: check whether the pinned OWLv2 forward lowers to MIGraphX."""
import json
import time
from pathlib import Path

from PIL import Image, ImageDraw
import torch

from . import owlv2_pinned as owl


def main() -> int:
    model, processor, device, torch_module, versions = owl.load_model()
    image = Image.new("RGB", (768, 768), (96, 105, 112))
    draw = ImageDraw.Draw(image)
    draw.rectangle((190, 170, 580, 610), fill=(155, 65, 35))
    draw.rectangle((90, 300, 190, 350), fill=(25, 25, 25))
    queries = [f"a photo of the {row['id']}: {row['definition']}" for row in owl.frozen_query_labels()]
    inputs = processor(text=queries, images=image, return_tensors="pt", padding=True).to(device)
    report = {"versions": versions, "device": str(device), "migraphx_attempted": False}
    report["encoded_query_token_lengths"] = [int(x) for x in inputs["attention_mask"].sum(dim=1).tolist()]
    report["text_max_position_embeddings"] = int(model.config.text_config.max_position_embeddings)
    if not versions.get("torch_migraphx"):
        report["migraphx_status"] = "unavailable-in-pinned-image"
    else:
        report["migraphx_attempted"] = True
        started = time.monotonic()
        try:
            import torch_migraphx  # noqa: F401 - registers the backend with torch.compile
            compiled = torch.compile(model, backend="migraphx")
            with torch.inference_mode():
                eager_output = model(**inputs)
                compiled_output = compiled(**inputs)
            report["migraphx_compile_and_run_seconds"] = time.monotonic() - started
            report["migraphx_max_abs_logits_delta"] = float(
                (eager_output.logits - compiled_output.logits).abs().max().item())
            report["migraphx_status"] = "compiled-and-compared"
        except Exception as exc:
            report["migraphx_status"] = "failed"
            report["migraphx_error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
    destination = Path("/results/owlv2-migraphx-synthetic-probe.json")
    destination.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

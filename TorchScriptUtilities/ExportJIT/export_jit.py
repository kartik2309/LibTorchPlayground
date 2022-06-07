import logging
import torch


@torch.no_grad()
def export_model_to_jit(model, path: str) -> None:
    model.eval()
    model.cpu()
    traced_model = torch.jit.script(model)
    logging.info(traced_model.graph)
    torch.jit.save(traced_model, path)

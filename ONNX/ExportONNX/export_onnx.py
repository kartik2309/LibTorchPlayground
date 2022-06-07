import logging
from typing import Union, Tuple, List, Dict
import torch


@torch.no_grad()
def export_model_to_onnx(
        model,
        path: str,
        sample_inputs: Tuple[torch.Tensor],
        input_names: List[str],
        output_names: List[str],
        dynamic_axes: Union[Dict[str, Dict[int, str]], Dict[str, List[int]]],
        opset_version=12,
        sample_outputs: Union[Tuple[torch.Tensor], None] = None
) -> None:
    model.eval()
    logging.info("Exporting model to ONNX")
    try:
        torch.onnx.export(
            model,
            sample_inputs,
            path,
            input_names=input_names,
            example_outputs=sample_outputs,
            opset_version=opset_version,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
    except (Exception, ) as e:
        logging.error("Error Occurred in ONNX Export! See the error log below")
        logging.exception(e)

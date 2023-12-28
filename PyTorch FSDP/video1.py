import torch
print(torch.__version__)

from torch.distributed.fsdp.wrap import(
    transformer_auto_wrap_policy,
)
import functools
from deep_vit import Transformer

transformer_auto_wrapper_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls = {
        Transformer,
    }
)
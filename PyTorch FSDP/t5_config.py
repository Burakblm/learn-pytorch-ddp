import time
import tqdm
import torch
from dataclasses import dataclass

from torch.distributed.fsdp import StateDictType
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.models.t5.modeling_t5 import T5Block
from base_config import base_config, fsdp_checkpointing_base, get_policy_base

@dataclass
class train_config(base_config):

    # model
    model = "google/t5-v1_1-small"
    # available models
    # t5-base
    # google/t5-v1_1-small
    # google/t5-v1_1-large
    # google/t5-v1_1-xl     #3b 
    # google/t5-v1_1-xxl    #11b

    # tokenizer
    tokenizer = "t5-small"
    


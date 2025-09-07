import json
import os
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers

from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_pert_model,
    prepare_model_for_kbit_training
)
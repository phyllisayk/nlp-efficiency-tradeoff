# Based on the Seq2Seq Trainer from Huggingface library 
# The original code has an error in calculating # samples/sec if early stopping was involved
# This code differes from the original by taking into account the number of steps that were completed when
#   calculating the # samples/sec

from typing import Any, Dict, List, Optional, Tuple, Union

import logging
import collections
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, Dataset
import time

from transformers import (
    Seq2SeqTrainer
)

from transformers.trainer_utils import PredictionOutput, EvalLoopOutput, EvalPrediction, denumpify_detensorize, speed_metrics
from transformers.file_utils import is_torch_tpu_available
from transformers.trainer_pt_utils import find_batch_size, nested_concat, nested_numpify, nested_truncate

from torch.profiler import profile, record_function, ProfilerActivity
import time
import math

logger = logging.getLogger(__name__)


class CustomTrainer(Seq2SeqTrainer):
    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        # call the original trainer and obtain its results
        orig_results = super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)

        # recalculate the train speed
        runtime = orig_results.metrics["train_runtime"]
        num_steps_completed = orig_results.global_step
        epochs_completed = orig_results.metrics["epoch"]

        args = self.args

        # check whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        # calculate using epochs completed and length of dataset
        if train_dataset_is_sized:
            num_train_samples = len(self.train_dataset) * epochs_completed
        else:
        # calculate number of steps and batch size if len() of dataset cannot be determined
            total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size
            num_train_samples = num_steps_completed * total_train_batch_size

        samples_per_second = num_train_samples / runtime
        orig_results.metrics["train_samples_per_second"] = round(samples_per_second, 3)

        return orig_results

        
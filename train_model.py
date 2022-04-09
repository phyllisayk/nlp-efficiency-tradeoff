"""
Fine-tuning the Huggingface library's seq2seq models for using the Seq2SeqTrainer.
"""

import logging
import os
import sys
import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import datasets
import nltk
import numpy as np
from datasets import load_dataset, load_metric

import wandb

import transformers
from transformers import (
    EncoderDecoderConfig, 
    EncoderDecoderModel,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed,
    LEDConfig,
    EarlyStoppingCallback
)
from transformers.trainer_utils import EvalPrediction, get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from metrics.f1 import compute_f1
from metrics.exact_match import compute_exact_match
from metrics.rouge import compute_rouge
from custom_trainer import CustomTrainer

from utils.load_configs import get_config_dict

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_is_qa: bool = field(
        default=False, metadata={"help": "Whether the dataset is a question-answering dataset - needed for LED."}
    )
    context_field: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the contexts (for question answering)."},
    )
    input_field: Optional[str] = field(
        default="input",
        metadata={"help": "The name of the column in the datasets containing the questions (for question answering)."},
    )
    output_field: Optional[str] = field(
        default="output",
        metadata={"help": "The name of the column in the datasets containing the answers (for question answering)."},
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_output_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    val_max_output_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_output_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    results_dir: str = field(
        default="./results",
        metadata={"help": "Directory path in which to store results"},
    )

    num_early_stopping: int = field(
        default=3,
        metadata={
            "help": "Stop training if metric worsens for this number of evaluation calls ."
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training/validation file/test_file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."
        if self.val_max_output_length is None:
            self.val_max_output_length = self.max_output_length

def main(model_name_or_path: str, dataset_name: str, seq_len: int, bsz: int, trial_num: int, wandb_proj: str, wandb_ent: str):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    config_dict = get_config_dict(model_name_or_path, dataset_name, seq_len, bsz, trial_num)
    model_args, data_args, training_args = parser.parse_dict(config_dict)

    if config_dict["n_gpu"] == 1:
        training_args.local_rank = -1

    # setup wandb
    wandb.init(project=wandb_proj, entity=wandb_ent, name=training_args.output_dir)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]

        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset('json', data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # check if config is a type of seq2seq model, set encoder-decoder config if not
    is_seq2seq_config = type(config) in AutoModelForSeq2SeqLM._model_mapping.keys()
    if is_seq2seq_config:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        config_encoder = config
        config_decoder = config
        config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

        # set decoder config to causal lm
        config.decoder.is_decoder = True
        config.decoder.add_cross_attention = True

        model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_args.model_name_or_path, model_args.model_name_or_path)
        
        # set some parameters for training
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.eos_token_id = tokenizer.sep_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.vocab_size = model.config.encoder.vocab_size


    model.config.max_length = data_args.max_seq_length
    model.config.min_length = 3
    model.config.no_repeat_ngram_size = 2
    model.config.early_stopping = True
    model.config.num_beams = data_args.num_beams if data_args.num_beams is not None else 1

    if is_seq2seq_config:
        model.resize_token_embeddings(len(tokenizer))
    else:
        model.encoder.resize_token_embeddings(len(tokenizer))
        model.decoder.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # Preprocessing the datasets.
    # We need to generate and tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    if data_args.input_field is None:
        input_field = column_names[0]
    else:
        input_field = data_args.input_field
        if input_field not in column_names:
            raise ValueError(
                f"--input_field' value '{data_args.input_field}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.context_field is not None:
        context_field = column_names[1]
    else:
        # Context field is none for non-QA task 
        context_field = None
    # else:
    #     context_field = data_args.context_field
    #     if context_field not in column_names:
    #         raise ValueError(
    #             f"--context_field' value '{data_args.context_field}' needs to be one of: {', '.join(column_names)}"
    #         )
    if data_args.output_field is None:
        output_field = column_names[2]
    else:
        output_field = data_args.output_field
        if output_field not in column_names:
            raise ValueError(
                f"--output_field' value '{data_args.output_field}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_output_length for training.
    max_output_length = data_args.max_output_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    logger.info(f"Max sequence length = {max_seq_length}")

    def preprocess_batch(
        examples,
        input_field: str,
        context_field: Optional[str],
        output_field: str,
    ) -> Tuple[List[str], List[str]]:
        """
        Preprocess the batch such that it is in the input/output format
        Params
            examples: a set of examples in dictionary format
            input field: string associated with the input in the dataset
            context field: string associated with the context in the dataset if context was not part of the input
            output field: string associated with the expected output in the dataset
        """
        
        # extract the raw words from the examples
        questions = examples[input_field]
        if context_field is not None:
            contexts = []
            for context in examples['ctxs']:
                context_str_list = [ctx['text'] for ctx in context]
                contexts.append(' '.join(context_str_list))
        answers = examples[output_field]

        # only include context if there is a context field
        if context_field is not None:
            inputs = [(question, context) for question, context in zip(questions, contexts)]
        else:
            inputs = questions

        # select the first answer if multiple answers available
        if isinstance(answers[0], list):
            targets = [answer[0] if len(answer) > 0 else "" for answer in answers]
        else:
            targets = answers

        return inputs, targets
    
    def preprocess_function(examples):
        """
        Preprocess the examples through tokenization and padding
        Params
            examples: a set of examples in dictionary format
            input field: string associated with the input in the dataset
            context field: string associated with the context in the dataset if context was not part of the input
            output field: string associated with the expected output in the dataset
        """

        # cast into input/output format for seq2seq tasks
        inputs, targets = preprocess_batch(examples, input_field, context_field, output_field)

        # Tokenize the examples
        model_inputs = tokenizer(inputs, max_length=max_seq_length, padding=padding, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_output_length, padding=padding, truncation=True)

        # Set mask if model was not intriniscally a encoder-decoder model
        if not is_seq2seq_config:
            model_inputs["decoder_input_ids"] = labels["input_ids"]
            model_inputs["decoder_attention_mask"] = labels["attention_mask"]

        # set global attention if LED 
        if isinstance(config, LEDConfig):
            # summarization: global attention on the first token only
            # QA: global attention on all the question tokens

            #####################QA######################################
            if data_args.dataset_is_qa:
                sep_indices = [indiv_input.index(tokenizer.sep_token_id) for indiv_input in model_inputs['input_ids']]

                model_inputs["global_attention_mask"] = np.zeros((len(model_inputs["input_ids"]), len(model_inputs["input_ids"][0])), dtype=int)

                for (idx, sep_idx) in enumerate(sep_indices):
                    model_inputs["global_attention_mask"][idx, 0:sep_idx] = 1
            else:
            ##################summarization########################################################
                # create 0 global_attention_mask lists
                model_inputs["global_attention_mask"] = [
                    [1] + [0] * (len(attn_mask) - 1) for attn_mask in model_inputs["attention_mask"]
                ]

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"].copy()
        return model_inputs

    # Set up the datasets
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            # We will select sample from whole data if agument is specified
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        # Create train feature from dataset
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        if data_args.max_train_samples is not None:
            # Number of samples might increase during Feature Creation, We select only specified max samples
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_examples = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            # We will select sample from whole data
            eval_examples = eval_examples.select(range(data_args.max_eval_samples))
        # Validation Feature Creation
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_examples.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        if data_args.max_eval_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_examples = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            # We will select sample from whole data
            predict_examples = predict_examples.select(range(data_args.max_predict_samples))
        # Predict Feature Creation
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_examples.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        if data_args.max_predict_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    def compute_metrics(eval_preds: EvalPrediction):
        """
        Compute the F1, exact match, and rouge score of the model predictions
        Param
            EvalPrediction: output from the model along with the global label
        """
        preds, labels = eval_preds

        # apply argmax to get the correct predictions
        preds = np.argmax(preds, axis=-1)

        # decode the model output
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        #compute exact match
        avg_exact_match = compute_exact_match(decoded_preds, decoded_labels)

        #compute f1
        avg_f1 = compute_f1(decoded_preds, decoded_labels)

        #compute rouge
        scores = compute_rouge(decoded_preds, decoded_labels)

        scores['exact_match'] = avg_exact_match
        scores['f1'] = avg_f1
        scores["rouge_mean"] = (scores["rouge1"] * scores["rouge2"] * scores["rougeL"]) ** (1.0 / 3.0)
        return scores

    # Initialize our Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=data_args.num_early_stopping)]
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint, ignore_keys_for_eval=["past_key_values", "encoder_last_hidden_state"])
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_output_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams 
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval", ignore_keys=["past_key_values", "encoder_last_hidden_state"])

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        results = trainer.predict(predict_dataset, max_length=max_length, num_beams=num_beams, metric_key_prefix="eval", ignore_keys=["past_key_values", "encoder_last_hidden_state"])
        metrics = results.metrics

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name_or_path", type=str,
        help='Name of model from HuggingFace or the directory path to the checkpoint of a trained model')
    parser.add_argument("dataset", type=str,
        help='Name of dataset from HuggingFace or the local directory path to the datasets')
    parser.add_argument("seq_len", type=int,
        help='Input sequence lengths')
    parser.add_argument("bsz", type=int,
        help='Batch sizes')
    parser.add_argument("--trial_num", type=int, default=1,
        help='Optional argument for the trial number to be included in the run name')
    parser.add_argument("--wandb_proj", type=str, default='1A6000_GPU_exp',
        help='Optional argument for name of the project on wandb')
    parser.add_argument("--wandb_ent", type=str, default='phyllisang',
        help='Optional argument for username or entity on wandb')
    args = parser.parse_args()
    
    main(args.model_name_or_path, args.dataset, args.seq_len, args.bsz, args.trial_num, args.wandb_proj, args.wandb_ent)
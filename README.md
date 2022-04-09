# nlp-bmk

 This code base utilizes the [HuggingFace 🤗 library](https://huggingface.co/) and many of the code is heavily inspired by the examples in the [HuggingFace github](https://github.com/huggingface/transformers/tree/master/examples). 

***

## Quick Links
1. [Repository Structure](#repository-structure)
2. [Setup](#setup)
3. [Configuration Files](#configuration-files)
4. [Datasets](#other-datasets)
5. [Fine-tuning](#Fine-tuning)
6. [Inference](#Inference)
7. [Wandb](#wandb)
8. [Reproducing Paper](#reproducing-paper)

*** 

## Repository Structure
    nlp-bmk 
      |
      |--> requirements.txt (contains all the required packages needed)
      |--> setup.sh (script that sets up a conda environment and install all the required packages)
      |--> train_model.py (script to fine-tune HuggingFace models)
      |--> run_training.sh (Bash script to show an example of how to run the code to fine-tune model)
      |--> run_inference.sh (Bash script to show an example of how to run the code to fine-tune model)
      |--> metrics (contains the files needed to calculate the accuracy scores)
              |
              |--> exact_match.py
              |--> f1.py
              |--> rouge.py 
              |--> metrics_utils.py (code to normalize strings)
      |--> utils 
              |
              |--> load_configs.py (set up and format the parameters)
      |--> wandb_scripts (contains the scripts used to extract metrics for wandb) 
              |
              |--> extract_inference.py (extract the inference metrics from wandb)
              |--> extract_training.py (extract the training metrics from wandb)
      |--> configs (contains the configuration files that contains the arguments for different datasets)
              |
              |--> base.json (configuration common to all datasets)
              |--> contract_nli.json (configuration for ContractNLI dataset)
              |--> gov_report.json (configuration for GovReport dataset)
              |--> narrative_qa.json (configuration for NarrativeQA dataset)
              |--> nq.json (configuration for Natural Questions dataset)
              |--> qasper.json (configuration for Qasper dataset)
              |--> qmsum.json (configuration for QMSum dataset)
              |--> quality.json (configuration for QuALITY dataset)
              |--> summ_screen_fd.json (configuration for SummScreenFD dataset)
      |--> metrics (contains the files needed to calculate the accuracy scores)
              |
              |--> exact_match.py
              |--> f1.py
              |--> rouge.py 
              |--> metrics_utils.py (code to normalize strings)
      |        

*** 

## Setup

We have provided a setup script to install all the necessary packages. This script utilizes the [Miniconda environment](https://docs.conda.io/en/latest/miniconda.html). Please ensure that Miniconda is downloaded in your system before running the setup script. If you prefer to use other python environment manager, please comment out lines 4-5 in the setup script before running it. To execute the setup script, run
```
cd nlp-bmk
source setup.sh
```

*** 

## Fine-tuning and Inference
The following sections details how to train pretrained models with different datasets.
#### Configuration Files
When training a specific model, the arguments in the configuration files customizes specific factors of training. Each dataset has its own configuration file under the name of `<dataset_name>.json` under the configs folder. For arguments common to all dataset, there is a `base.config` file that contains all the arguments.

Many of the arguments are part of the HuggingFace Training Arguments and are detailed [here](https://huggingface.co/docs/transformers/main_classes/trainer#trainingarguments). However, we have outlined some of the more important arguments below. 

| Argument              | Description                                                                           |
| ----------------------| --------------------------------------------------------------------------------------|
| n_gpu                 | Number of GPU used                                                                    |
| model_name_or_path    | The name of the pretrained model from HuggingFace                                     |
| max_train_samples     | Number of training samples to use. Remove argument if whole training set is used      |
| max_eval_samples      | Number of validation samples to use. Remove argument if whole validation set is used  |
| output_dir            | An output directory to store intermediate models and checkpoints                      |

#### Other Datasets
The structure in this codebase is optimized for performing fine-tuning HuggingFace Model on the datasets from the SCROLLS benchmark. To train models with new datasets, check that the arguments in `base.config` apply and create a new json file in the configs folder with the name `<dataset_name>.config`. In this json file specific to the dataset, please make sure to include the dataset name if directly taken from HuggingFace or the path to the dataset if not taken from HuggingFace. 

For datasets specific to HuggingFace, `dataset_name` is the dataset name from HuggingFace that can be found [here](https://huggingface.co/datasets). If the dataset is a benchmark suite that contains multiple datasets in it, you can specify a specific dataset using `dataset_config_name`.  Please refer to `gov_report.json` as an example. 
```
"dataset_name": "tau/scrolls",
"dataset_config_name": "gov_report",
```

For datasets outside of the HuggingFace datasets, you will need to provide path to the training, validation (evaluation), and testing sets.  Please refer to `nq.json` as an example. 
```
"train_file": "./datasets/nq-train.json",
"validation_file": "./datasets/nq-dev.json",
"test_file": "./datasets/nq-test.json",
```

### Fine-tuning
To train any model on the HuggingFace interface, make sure the "do_train" argument in the `base.config` file is set to true before running the following command: 

```
conda activate nlp-bmk
torchrun --nnodes=<num_gpu_nodes> --nproc_per_node=<num_gpu_per_node> train_model.py <huggingface_model_name> <dataset_name_or_path> <input_sequence_length> <batch_size_per_gpu>
```

Valid HuggingFace models can be found [here](https://huggingface.co/models). The dataset that can be used include any valid [HuggingFace dataset](https://huggingface.co/datasets). 

An example of running fine-tuning for LED-base model on the GovReport dataset with an input sequence length of 1024 and batch size of 24 can be found in run_training.sh. 

### Inference
To perform valiation on Longformer, make sure the "do_eval" argument in the config file is set to true before running the following command: 

```
conda activate nlp-bmk
torchrun --nnodes=<num_gpu_nodes> --nproc_per_node=<num_gpu_per_node> train_model.py <trained_model_directory> <dataset_name_or_path> <input_sequence_length> <batch_size_per_gpu>
```

Valid HuggingFace models can be found [here](https://huggingface.co/models). The dataset that can be used include any valid [HuggingFace dataset](https://huggingface.co/datasets). 

If the default output directory was used, the `trained_model_directory` will be `./results/<run_name_used_to_train_model>`

An example of running inference for LED-base model on the GovReport dataset with an input sequence length of 1024 and batch size of 24 can be found in run_inference.sh.

*** 

## Wandb
We utilize the [Weights and Baises (wandb)](https://wandb.ai/site) tool to help collect metrics during fine-tuning and inference. 

There are two ways to set your wandb username/entity (wandb_ent) and project (wandb_proj):

1. Change the default value of the parser arguments in train_model.py (line 719 or 721)
2. Pass in your username and project string as a command line argument.
    ```
    conda activate nlp-bmk
    torchrun --nnodes=<num_gpu_nodes> --nproc_per_node=<num_gpu_per_node> train_model.py <huggingface_model_name> <dataset_name_or_path> <input_sequence_length> <batch_size_per_gpu> --wandb_proj <wandb_project_name> --wandb_ent <wandb_entity_or_username>
    ```

### Wandb Scripts
Once you have finished running the experiments, you can use the scripts under wandb_scripts to extract the metrics. Make sure that you change the wandb username/entity and project name in line 8 of the files before running the following commands. 

For fine-tuning: 
```
python3 extract_training.py
```

For inference: 
```
python3 extract_inference.py
```

*** 

## Reproducing Paper
The code here has been used to generate the results found in our paper: [Characterizing the Efficiency vs. Accuracy Trade-off for Long-Context NLP Models](). 

The different `dataset_name_or_path` used in the paper were:
```
["gov_report", "summ_screen_fd", "qmsum", "qasper"]
```

The different `huggingface_model_name` used in the paper were:
```
["allenai/led-base-16384", "allenai/led-large-16384", "google/bigbird-pegasus-large-pubmed"]
```

To run fine-tuning on the model, refer to the command in the [fine-tuning section](#Fine-tuning). 

To run inference on the model, refer to the command in the [inference section](#inference). 
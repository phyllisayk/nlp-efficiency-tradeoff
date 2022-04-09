import wandb
import csv
from tqdm import tqdm

# set up wandb 
api = wandb.Api()

entity, project = 'longrange-nlp-bmk', '1A6000_GPU_exp'
csv_file = 'training_metrics.csv'

print('Connecting...')
runs = api.runs(entity + "/" + project) 

# set up the headers for the final CSV file
csv_header = ['Run Name', 'Model Name', "Dataset Name", "Sequence Length", "Train batch_size", 'Train Speed', '#Epoch', '#Iterations', 'Train Time', 'GPU Memory (GB)', 'GPU Utilization (%)', 'Mean GPU Power(W)', 'Maximum GPU power(W)', 'Total GPU Energy(kW)', "Train Flos", "exact_match", "f1", "rouge1", "rouge2", "rougeL", "rouge_mean"]

# normalize all the dataset name
DATASET_NAME = {
    "qmsum": "qmsum", 
    "govreport": "gov_report",
    "gov": "gov_report",
    "summ": "summ_screen_fd",
    "sumscreen": "summ_screen_fd",
    "qasper": "qasper",
    "qaspar": "qasper", 
    "quality": "quality",
    "contractnli": "contract_nli", 
    "contract": "contract_nli", 
    "narrative": "narrative_qa"
}

# name of the metrics
METRICS = ["exact_match", "f1", "rouge1", "rouge2", "rougeL", "rouge_mean"]

#open csv file and write metrics to it
with open(csv_file, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = csv_header)
    writer.writeheader()
    for i, run in enumerate(tqdm(runs)):
        # get GPU hardware metrics (power, memory)
        gpu_utilization = run.history(stream="events")['system.gpu.0.gpu']
        mean_gpu_utilization = gpu_utilization.mean(axis=0)
        gpu_memory = run.history(stream="events")['system.gpu.0.memoryAllocated']
        mean_gpu_mem = (gpu_memory.mean(axis=0)/100) * 48 # get percent and multiply vy 48GB
        power_metrics = run.history(stream="events")['system.gpu.0.powerWatts']
        mean_power = power_metrics.mean(axis=0)
        max_power = power_metrics.max(axis=0)
        total_power = (power_metrics.sum(axis=0))/1000

        metrics_dict = {'GPU Memory (GB)': mean_gpu_mem, 'GPU Utilization (%)': mean_gpu_utilization,'Mean GPU Power(W)': mean_power, 'Maximum GPU power(W)':max_power, 'Total GPU Energy(kW)':total_power}

        # get dataset name
        name_split = run.name.split('/')
        name = name_split[-1]
        if name == '': 
            name = name_split[-2]
        metrics_dict['Run Name'] = name
        extracted_dataset = name.split('_')[1]
        dataset = DATASET_NAME[extracted_dataset]
        metrics_dict['Dataset Name'] = dataset

        # get model name, batch size, and sequence length
        config = run.config
        if config:
            model = (config['_name_or_path']).split('/')[-1]
            metrics_dict['Model Name'] = model

            batch_size = config['train_batch_size']
            metrics_dict['Train batch_size'] = batch_size

            seq_len = config['max_length']
            metrics_dict['Sequence Length'] = seq_len

        # get training metrics
        summary = run.summary
        train_speed = summary.get('train/train_samples_per_second', 0)
        train_time = summary.get('train/train_runtime', 0)
        train_flos = summary.get('train/total_flos', 0)/1000000
        train_epochs = summary.get('train/epoch', 0)
        train_iter = summary.get('train/global_step', 0)
        metrics_dict['Train Speed'] = train_speed
        metrics_dict['Train Time'] = train_time
        metrics_dict['Train Flos'] = train_flos
        metrics_dict['#Epoch'] = train_epochs
        metrics_dict['#Iterations'] = train_iter

        # add all the accuracy metrics
        for met_str in METRICS:
            key_str = "eval/"+met_str
            met = summary[key_str]
            metrics_dict[met_str] = met

        writer.writerows([metrics_dict])
import wandb
import csv
from tqdm import tqdm

# set up wandb 
api = wandb.Api()

entity, project = 'longrange-nlp-bmk', 'inference_1A6000'
csv_file = 'inference_hardware.csv'

print('Connecting...')
runs = api.runs(entity + "/" + project) 

# set up the headers for the final CSV file
csv_header = ['Model Name', "Dataset Name", "Sequence Length", "Batch_size", "Trial", "Inference Speed", 'Inference Time', 'GPU Memory (GB)', 'GPU Utilization (%)', 'Mean GPU Power(W)', 'Maximum GPU power(W)', 'Total GPU Energy(kW)', "exact_match", "f1", "rouge1", "rouge2", "rougeL", "rouge_mean"]

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

# list of model names used and normalizing model name
ACCEPTED_MODELS = ['led-base-16384', 'led-large-16384', 'bigbird-pegasus-large-pubmed']
MODEL_NAME = {
    "led_large": 'led-large-16384',
    "bigbird_large": 'bigbird-pegasus-large-pubmed'
}

# name of the metrics
METRICS = ["exact_match", "f1", "rouge1", "rouge2", "rougeL", "rouge_mean"]

#open csv file and write metrics to it
with open(csv_file, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = csv_header)
    writer.writeheader()
    for i, run in enumerate(tqdm(runs)):
        # get dataset name and trial number
        name_split = run.name.split('/')
        name = name_split[-1]
        if name == '': 
            name = name_split[-2]
        name_components = name.split('_')
        dataset_idx = 4
        extracted_dataset = name_components[dataset_idx]
        # find the actual dataset name
        while extracted_dataset not in DATASET_NAME:
            dataset_idx = dataset_idx + 1
            extracted_dataset = name_components[dataset_idx]

        dataset = DATASET_NAME[extracted_dataset]
        trial_num = name_components[-1]

        # get GPU hardware metrics (power, memory)
        gpu_utilization = run.history(stream="events")['system.gpu.0.gpu']
        mean_gpu_utilization = gpu_utilization.mean(axis=0)
        gpu_memory = run.history(stream="events")['system.gpu.0.memoryAllocated']
        mean_gpu_mem = (gpu_memory.mean(axis=0)/100) * 48 # get percent and multiply vy 48GB
        power_metrics = run.history(stream="events")['system.gpu.0.powerWatts']
        mean_power = power_metrics.mean(axis=0)
        max_power = power_metrics.max(axis=0)
        total_power = (power_metrics.sum(axis=0))/1000

        # get model name, batch size, and sequence length
        config = run.config
        trained_path_split = (config['_name_or_path']).split('/')
        trained_path = trained_path_split[-1]
        if trained_path == '': 
            trained_path = trained_path_split[-2]
        path_components = trained_path.split('_')
        model = path_components[0]

        #normalize model name
        if model not in ACCEPTED_MODELS:
            orig_model = path_components[0] + "_" + path_components[1]
            model = MODEL_NAME[orig_model]
        batch_size = config['eval_batch_size']
        seq_len = config['max_length']

        metrics_dict = {'Model Name': model, "Dataset Name": dataset, "Sequence Length": seq_len, "Batch_size": batch_size, "Trial": trial_num, 'GPU Memory (GB)': mean_gpu_mem, 'GPU Utilization (%)': mean_gpu_utilization,'Mean GPU Power(W)': mean_power, 'Maximum GPU power(W)':max_power, 'Total GPU Energy(kW)':total_power}
        
        # get inference metrics
        summary = run.summary
        eval_speed = summary.get('eval/samples_per_second', 0)
        eval_time = summary.get('eval/runtime', 0)
        metrics_dict['Inference Speed'] = eval_speed
        metrics_dict['Inference Time'] = eval_time


        # print(summary)
        # add all the metrics
        for met_str in METRICS:
            key_str = "eval/"+met_str
            met = summary[key_str]
            metrics_dict[met_str] = met

        writer.writerows([metrics_dict])
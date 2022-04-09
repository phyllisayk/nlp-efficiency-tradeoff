import json
from typing import List, Dict, Any

CONFIG_DIR="./configs/"
BASE_CONFIG="base.json"

def get_config_dict(model_name_or_path:str, dataset_name: str, seq_len: int, bsz: int, trial_num: int) -> Dict[str, Any]:
    '''
    Combine the base configs and the configs associated into a dictionary
    '''
    # Open and load base config
    base_file = open(CONFIG_DIR+BASE_CONFIG)
    base_dict = json.load(base_file)
    base_file.close()

    # Open and load config associated with dataset
    dataset_file = open(CONFIG_DIR+dataset_name+'.json')
    dataset_config = json.load(dataset_file)
    dataset_file.close()

    # combine dictionary
    base_dict.update(dataset_config)

    # Update dictionary based on other parameters
    base_dict["model_name_or_path"] = model_name_or_path
    base_dict["max_seq_length"] = seq_len
    base_dict["per_device_train_batch_size"] = bsz
    base_dict["per_device_eval_batch_size"] = bsz

    assert (base_dict["do_train"] or base_dict["do_eval"] or base_dict["do_predict"]), (
        "Nothing to do. Need to set one of the following to true: do_train, do_eval, do_predict")

    # Create a unique name for this run - <model_name>_<dataset>_<seq_len>_bsz<batch_size>_<num_gpu>gpus_<num_epochs>epochs
    model_name = model_name_or_path.split('/')[-1]
    if model_name == '': 
        model_name =model_name_or_path.split('/')[-2]

    # if training, should obtain all the necessary field
    if base_dict["do_train"]:
        seq_len = str(base_dict['max_seq_length'])
        batch_size = "bsz"+str(base_dict['per_device_train_batch_size'] * base_dict['n_gpu'])
        gpus = str(base_dict['n_gpu'])+"gpus"
        epochs = str(base_dict['num_train_epochs'])+"epochs"
        name_list = [model_name, dataset_name, seq_len, batch_size, gpus, epochs]
    # if not training, fields should already be populated in model path - add inference batch size and gpu number
    else:
        # run_string = "inference_"+model_name
        batch_size = "bsz"+str(base_dict['per_device_train_batch_size'] * base_dict['n_gpu'])
        gpus = str(base_dict['n_gpu'])+"gpus"
        trial = str(trial_num)
        name_list = ["inference", batch_size, gpus, model_name, trial]


    run_string = '_'.join(name_list)
    base_dict["output_dir"] = base_dict["results_dir"] + "/" + run_string + "/"
    print(f"Results saved in: {base_dict['output_dir']}")

    return base_dict
     

if __name__ == "__main__":
    main()
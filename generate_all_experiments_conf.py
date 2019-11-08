import os

def generate_experiments_configurations():
    seed_lst = [12345, 23456, 34567, 45678, 56789]

    factors_dict = {
        "Imbalance": [200, 100, 50, 20, 10, 1],
        "Uniform noise": [0, 0.4, 0.6],
        "Flip noise": [0, 0.2, 0.4],
    }

    cifar_type_dict = {
        10: None,
        100: None
    }

    experiment_type_dict = {
        "Imbalance": None,
        "Uniform noise": None,
        "Flip noise": None
    }

    model_type_dict = {
        "MWN": None,
        "Baseline": None,
        "FineTune": None
    }

    cmd = ""
    for cifar_type in cifar_type_dict:
        for model_type in model_type_dict:
            for experiment_type in experiment_type_dict:
                for factor in factors_dict[experiment_type]:
                    for seed in seed_lst:
                        path_log_file = os.path.join("Logs",
                                                     f"log_file_{cifar_type}_{model_type}_{experiment_type.replace(' ', '_')}_{factor}_{seed}.txt")
                        cmd += f"nohup python adl_project_script.py --cifar_type {cifar_type} --model_type {model_type} --experiment_type '{experiment_type}' --factor {factor} --seed {seed} > {path_log_file} &\n"

    with open("all_experiments.sh", "w") as fl:
        fl.write(cmd)
    # os.chmod("all_experiments.sh", 777)

generate_experiments_configurations()
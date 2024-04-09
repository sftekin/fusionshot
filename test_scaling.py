import numpy as np
from ens_pruning_src.run_ga import run as run_ga
from fusionshot_src.train_ensemble import create_data_loaders, train_ensemble
from torch.utils.data import DataLoader
import time

all_model_names = ["DeepEMD", "easy_ResNet18", "matchingnet_Conv4", "matchingnet_Conv6", "matchingnet_ResNet10",
                   "matchingnet_ResNet18",
                   "matchingnet_ResNet34",
                   "protonet_Conv4", "protonet_Conv6", "protonet_ResNet10", "protonet_ResNet18", "protonet_ResNet34",
                   "relationnet_Conv4", "relationnet_Conv6", "relationnet_ResNet10", "relationnet_ResNet18",
                   "relationnet_ResNet34",
                   "maml_approx_Conv4", "maml_approx_Conv6", "maml_approx_ResNet10", "maml_approx_ResNet18",
                   "maml_approx_ResNet34",
                   "simpleshot_Conv4", "simpleshot_Conv6", "simpleshot_ResNet10", "simpleshot_ResNet18",
                   "simpleshot_ResNet34", "simpleshot_DenseNet121", "simpleshot_WideRes"]


def test_scale(model_names):
    if len(model_names) > 2:
        selected_models = run_ga(model_names, n_query, n_shot, n_way, dataset,
                                 weights=[0.6, 0.4], size_penalty=True)
    else:
        selected_models = model_names

    settings = [n_way, n_shot, n_query, 600]
    all_data = create_data_loaders(selected_models, dataset_name='miniImagenet',
                                   class_types=["base", "val", "novel"], settings=settings, shuffle=False)

    split = [2000, 500, 500]
    data_split = np.split(all_data, 3)
    loaders = [DataLoader(data_split[i][:split[i]], batch_size=64, shuffle=True) for i in range(3)]
    exp_result = train_ensemble(selected_models, loaders[0], loaders[1],
                                loaders[2], n_epochs=150, save_dir="test_scale/",
                                verbose=True)

    return exp_result


if __name__ == '__main__':
    exp_num = 5
    n_query = 15
    n_shot = 1
    n_way = 5
    dataset = "miniImagenet"

    for exp_k in range(exp_num):
        time_arr = []
        acc_arr = []
        conf_arr = []
        pool = [all_model_names[0]]
        for idx in range(1, len(all_model_names)):
            pool.append(all_model_names[idx])
            try:
                start_time = time.time()
                res = test_scale(model_names=pool)
                time_arr.append(time.time() - start_time)
                acc_arr.append(res["test_acc"])
                conf_arr.append(res["test_conf"])
            except Exception as e:
                print(e)

        scale_stats = np.stack([np.array(acc_arr), np.array(conf_arr), np.array(time_arr)])
        np.save(f"scale_stats_{exp_k}", scale_stats)

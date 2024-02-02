import numpy as np
from train_ensemble import create_data_loaders, train_ensemble
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def roll_data(input_data, window_size=1000, stride=100):
    for i in range(0, len(input_data) - window_size, stride):
        yield input_data[i:i + window_size]


def get_base_pred(in_data, model_names, n_way):
    y = in_data[:, -1]
    base_acc = []
    for i in range(0, len(model_names) * n_way, n_way):
        base_acc.append(np.mean(in_data[:, i:i + n_way].argmax(axis=1) == y) * 100)
    return base_acc


def run():
    save_dir = "/home/selim/Documents/PycharmProjects/fusionshot/fusionshot_src/ens_checkpoints/rolling_exp"
    n_way, n_shot, n_query, ep_num = (5, 5, 15, 300)
    model_names = ["DeepEMD_ResNet18", "simpleshot_ResNet18", "protonet_ResNet18"]
    settings = [n_way, n_shot, n_query, ep_num]
    mini_data = create_data_loaders(model_names, dataset_name='miniImagenet',
                                    class_types=["val", "novel"], settings=settings, shuffle=False)

    cub_data_1 = create_data_loaders(model_names[1:], dataset_name='CUB',
                                     class_types=["val", "novel"], settings=settings, shuffle=False)
    cub_data_0 = create_data_loaders([model_names[0]], dataset_name="cross", class_types=["val", "novel"],
                                     settings=settings, shuffle=False)
    cub_data = np.concatenate([cub_data_0[:, :-1], cub_data_1], axis=1)

    all_data = np.concatenate([mini_data, cub_data])

    window_len = 2000
    stride = 500
    val_size = 500
    novel_size = 500
    train_size = window_len - (2 * val_size)
    rolling_loader = iter(roll_data(input_data=all_data, stride=stride, window_size=window_len))
    acc_all, conf_all = [], []
    base_all = []
    for i, in_data in enumerate(rolling_loader):
        train_data = in_data[:train_size]
        val_data = in_data[train_size:train_size + val_size]
        novel_data = in_data[train_size + val_size:]

        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=64, shuffle=True)
        novel_loader = DataLoader(novel_data, batch_size=64, shuffle=True)

        save_dir = f"{save_dir}/{i}"
        exp_result = train_ensemble(model_names, train_loader, val_loader,
                                    novel_loader, n_epochs=150, save_dir=save_dir,
                                    verbose=False)
        acc_all.append(exp_result["test_acc"])
        conf_all.append(exp_result["test_conf"])
        base_all.append(get_base_pred(novel_data, model_names, n_way))

    acc_all, conf_all, base_all = np.array(acc_all), np.array(conf_all), np.array(base_all)
    np.save("acc_all_rolling.npy", acc_all)
    np.save("conf_all_rolling.npy", conf_all)

    plt.figure()
    plt.plot(acc_all, label="FusionShot")
    for i, mn in enumerate([model_names[0]]):
        plt.plot(base_all[:, i], label=mn)
    plt.ylim(0, 100)
    plt.legend()
    plt.show()

    print()

    # cub_data = create_data_loaders(model_names, dataset_name='CUB',
    #                                class_types=["val", "novel"], settings=settings)

    print()


if __name__ == '__main__':
    run()

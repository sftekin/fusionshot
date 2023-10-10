import os
import glob
import itertools
import pickle as pkl
import numpy as np
import scipy
import torch.nn
import torch.nn as nn
from torch.utils.data import DataLoader


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        self.net = nn.Sequential(
            # nn.BatchNorm1d(num_features=input_dim),
            nn.Linear(input_dim, hidden_dim[0]),
            nn.Sigmoid(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.Sigmoid(),
            nn.Linear(hidden_dim[1], output_dim),
            nn.Sigmoid()
        )
        self.net.apply(self.init_weights)

    def forward(self, x):
        out = self.net(x)
        out = torch.softmax(out, dim=-1)
        return out

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)


def min_max(in_arr):
    min_ = in_arr.min()
    max_ = in_arr.max()
    return (in_arr - min_) / (max_ - min_), (min_, max_)


def standardize(in_arr):
    return (in_arr - in_arr.mean()) / in_arr.std()


def load_logits(model_names, dataset, class_type, nway, nshot, perform_norm=True, norm_params=None, ep_count=600):
    save_dir = "model_outs"
    logits = []
    for model_n in model_names:
        method_str = model_n.split("_")
        if len(method_str) > 1:
            extension_name = method_str[1]
            method_name = f"{method_str[0]}_{extension_name}" if extension_name in ["approx", "softmax"] else method_str[0]
        else:
            method_name = method_str[0]

        # with open(f"{save_dir}/{dataset}/{method_name}/{model_n}_{class_type}_{nway}way_{nshot}shot.pkl", "rb") as f:
        #     results = pkl.load(f)
        # logit = results["logits"]
        logit = np.load(f"{save_dir}/{dataset}/{method_name}/{model_n}_{class_type}_{nway}way_{nshot}shot_logits.npy")
        if method_name == "DeepEMD":
            logit = np.transpose(logit.reshape(ep_count, 15, 5, 5), axes=[0, 2, 1, 3])
            logit = logit.reshape(ep_count, 75, 5)
        elif method_name == "simpleshot":
            logit *= -1
        logits.append(logit.reshape(-1, 5))

    if perform_norm:
        logits_t = []
        stats = []
        for l in range(len(model_names)):
            # if norm_params is not None:
            #     logit_ = (logits[l] - norm_params[0]) / (norm_params[1] - norm_params[0])
            # else:
            #     logit_, stat_ = min_max(logits[l])
            #     stats.append(stat_)
            logits_t.append(scipy.special.softmax(logits[l], axis=1))
        logits = logits_t
    return logits


def create_data(logits, n_way, n_query, ep_count=600, shuffle=True):
    x = np.concatenate(logits, axis=1)
    y = np.tile(np.repeat(range(n_way), n_query), ep_count).flatten()
    data = np.concatenate([x, y[:, None]], axis=1)
    if shuffle:
        random_idx = np.random.permutation(range(len(data)))
        data = data[random_idx]

    return data


def test_loop(model, data_loader, ret_logit=False, device="cuda"):
    assert device in ["cuda", "cpu"]
    acc_all = []
    logits = []
    labels = []
    for i, batch_data in enumerate(data_loader):
        in_x = batch_data[:, :-1].to(device).float()
        scores = model(in_x)
        label = batch_data[:, -1].numpy()

        scores = scores.detach().cpu().numpy()
        in_x = in_x.detach().cpu().numpy()
        pred = np.argmax(scores, axis=1)
        corrects = np.sum(pred == label)
        acc_all.append(corrects / len(label) * 100)
        if ret_logit:
            logits.append(np.concatenate([in_x, scores], axis=1))
            labels.append(label)

    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)

    if ret_logit:
        logits = np.concatenate(logits)
        labels = np.concatenate(labels)
        return acc_mean, acc_std, logits, labels
    else:
        return acc_mean, acc_std


def run(model_names, dataset, normalize_flag, save_dir, nway, nshot,):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_logits = load_logits(model_names, dataset=dataset, nway=n_way, nshot=n_shot,
                               class_type="base", perform_norm=normalize_flag)
    train_data = create_data(train_logits, n_way, n_query)

    val_logits = load_logits(model_names, dataset=dataset, nway=n_way, nshot=n_shot,
                             class_type="val", perform_norm=normalize_flag)
    val_data = create_data(val_logits, n_way, n_query)

    novel_logits = load_logits(model_names, dataset=dataset, nway=n_way, nshot=n_shot,
                               class_type="novel", perform_norm=normalize_flag)
    novel_data = create_data(novel_logits, n_way, n_query)

    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=True)
    novel_loader = DataLoader(novel_data, batch_size=64, shuffle=True)
    model = MLP(len(model_names) * 5, [100, 100], 5)

    model = model.to("cuda")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_acc = 0
    for epoch in range(n_epochs):
        avg_loss = []
        for i, batch_data in enumerate(train_dataloader):
            in_x = batch_data[:, :-1].to("cuda").float()
            label = batch_data[:, -1].type(torch.long).to("cuda")

            optimizer.zero_grad()
            out = model(in_x)
            loss = loss_fn(out, label)

            # if lambda_1 > 0:
            #     # get L1 over weights
            #     l1_reg = torch.tensor(0., requires_grad=True).float().to("cuda")
            #     for name, param in model.named_parameters():
            #         if "weight" in name:
            #             l1_reg = l1_reg + torch.norm(param, p=1)
            #     # return regularized loss (L2 is applied with optimizer)
            #     loss = loss + lambda_1 * l1_reg

            loss.backward()
            optimizer.step()
            avg_loss.append(loss.item())

        if epoch % 10 == 0:
            run_loss = np.mean(avg_loss)
            print(f'Epoch {epoch} | Loss {run_loss:.4f}')

        acc_mean, acc_std = test_loop(model, val_loader)

        if acc_mean > best_val_acc:
            conf = 1.96 * acc_std / np.sqrt(len(val_loader))
            print(f'best model Val Acc = {acc_mean:.4f} +- {conf:.2f}')

            outfile = os.path.join(save_dir, f'best_model.tar')
            torch.save({'epoch': epoch,
                        'state': model.state_dict(),
                        "accuracy": acc_mean,
                        "confidence": conf}, outfile)
            best_val_acc = acc_mean

    best_dict = torch.load(f"{save_dir}/best_model.tar")
    model.load_state_dict(best_dict["state"])
    model.eval()

    acc_mean, acc_std = test_loop(model, novel_loader)
    conf = 1.96 * acc_std / np.sqrt(len(novel_loader))
    print(f'Novel Acc = {acc_mean:.4f} +- {conf:.2f}')
    exp_result = dict(val_acc=best_dict["accuracy"],
                      val_conf=best_dict["confidence"],
                      test_acc=acc_mean,
                      test_conf=conf,
                      state=model.state_dict(),
                      model_names=model_names)
    torch.save(exp_result, f"{save_dir}/results.tar")
    print(f"{model_names} finished with Acc = {acc_mean:.4f} +- {conf:.2f}...")
    return acc_mean, conf


def modify_save_path(save_path):
        save_path = save_path.replace("matchingnet", "mn")
        save_path = save_path.replace("protonet", "pn")
        save_path = save_path.replace("maml_approx", "maml")
        save_path = save_path.replace("relationnet_softmax", "rn")
        save_path = save_path.replace("ResNet", "RN")
        save_path = save_path.replace("Conv", "C")
        save_path = save_path.replace("simpleshot", "ss")
        return save_path


if __name__ == '__main__':
    n_query = 15
    n_way = 5
    n_shot = 5
    n_epochs = 300
    dataset_name = "cross"
    # lambda_1 = 0.01
    # temperatures = [1, 1, 1]
    # temp_flag = True
    normalize = True

    # # *** Simple Shot Exps ***
    # all_names = ["simpleshot_Conv4", "simpleshot_Conv6", "simpleshot_ResNet10",
    #              "simpleshot_ResNet18", "simpleshot_ResNet34",
    #              "simpleshot_WideRes", "simpleshot_DenseNet121"]
    # ens_sizes = np.arange(2, len(all_names) + 1)
    # for ens_size in ens_sizes:
    #     combinations = itertools.combinations(all_names, ens_size)
    #     for comb in combinations:
    #         sv_path = f"ens_checkpoints/{'-'.join(comb)}"
    #         sv_path = modify_save_path(sv_path)
    #         run(model_names=comb, normalize_flag=normalize, save_dir=sv_path)
    #
    # # *** Exps Method Wise ***
    # model_back_bones = ["Conv4", "Conv6", "ResNet10", "ResNet18", "ResNet34"]
    # method_names = ["matchingnet", "protonet", "maml_approx", "relationnet", "simpleshot"]
    # for bb in model_back_bones:
    #     model_names = [f"{m}_{bb}" for m in method_names]
    #     ens_sizes = np.arange(2, len(model_names) + 1)
    #     best_acc, best_conf, best_comb = 0, 0, None
    #     for ens_size in ens_sizes:
    #         combinations = itertools.combinations(model_names, ens_size)
    #         for comb in combinations:
    #             sv_path = f"ens_checkpoints/{'-'.join(comb)}"
    #             sv_path = modify_save_path(sv_path)
    #             print(f"Training {sv_path}")
    #             exp_acc, exp_conf = run(model_names=comb, normalize_flag=normalize, save_dir=sv_path)
    #             if exp_acc > best_acc:
    #                 best_acc = exp_acc
    #                 best_conf = exp_conf
    #                 best_comb = comb
    #     print(f"Experiment finished Best Comb: {best_comb}, with {best_acc:.2f} +- {best_conf}")

    # # *** Exps Backbone Wise ***
    # model_back_bones = ["Conv4", "Conv6", "ResNet10", "ResNet18", "ResNet34"]
    # method_names = ["matchingnet", "protonet", "maml_approx", "relationnet", "simpleshot"]
    # for m in method_names:
    #     model_names = [f"{m}_{bb}" for bb in model_back_bones]
    #     ens_sizes = np.arange(2, len(model_names) + 1)
    #     best_acc, best_conf, best_comb = 0, 0, None
    #     for ens_size in ens_sizes:
    #         combinations = itertools.combinations(model_names, ens_size)
    #         for comb in combinations:
    #             sv_path = f"ens_checkpoints/{'-'.join(comb)}"
    #             sv_path = modify_save_path(sv_path)
    #             print(f"Training {sv_path}")
    #             exp_acc, exp_conf = run(model_names=comb, normalize_flag=normalize, save_dir=sv_path)
    #             if exp_acc > best_acc:
    #                 best_acc = exp_acc
    #                 best_conf = exp_conf
    #                 best_comb = comb
    #     print(f"Experiment finished Best Comb: {best_comb}, with {best_acc:.2f} +- {best_conf}")

    # # *** Baseline Comparison model Exps ***
    # all_names = ["matchingnet_ResNet18", "protonet_ResNet18", "maml_approx_ResNet18",
    #              "relationnet_softmax_ResNet18", "simpleshot_ResNet18", "DeepEMD"]
    # ens_sizes = np.arange(2, len(all_names) + 1)
    # for ens_size in ens_sizes:
    #     combinations = itertools.combinations(all_names, ens_size)
    #     for comb in combinations:
    #         sv_path = f"ens_checkpoints/{'-'.join(comb)}"
    #         sv_path = modify_save_path(sv_path)
    #         sv_path += f"_{n_way}way_{n_shot}shot"
    #         if os.path.exists(sv_path):
    #             print(f"Passed {comb}")
    #             continue
    #         else:
    #             if os.path.exists(sv_path):
    #                 print(f"Passed {comb}")
    #                 continue
    #         print(f"Training {sv_path}")
    #         run(model_names=comb, normalize_flag=normalize, save_dir=sv_path,  nway=n_way, nshot=n_shot, dataset=dataset_name)

    # *** Best model Exps ***
    # all_names = ["maml_approx_ResNet18", "protonet_ResNet18",  "matchingnet_ResNet18", "relationnet_softmax_ResNet18"]
    # all_names = ["simpleshot_Conv4", "simpleshot_ResNet18",  "simpleshot_ResNet34",
    #              "simpleshot_ResNet18", "simpleshot_WideRes", "simpleshot_DenseNet121"]
    # all_names = ["protonet_ResNet18", "relationnet_softmax_ResNet18", "matchingnet_ResNet18", "simpleshot_ResNet18",  "maml_approx_ResNet18", "DeepEMD"]

    # # ** GA decided **
    # all_names = ['matchingnet_Conv4', 'matchingnet_ResNet10', 'matchingnet_ResNet18',
    #          'matchingnet_ResNet34', 'protonet_Conv4', 'protonet_ResNet10',
    #          'protonet_ResNet18', 'protonet_ResNet34', 'maml_approx_Conv4',
    #          'maml_approx_ResNet10', 'maml_approx_ResNet18', 'maml_approx_ResNet34',
    #           'relationnet_Conv4', 'relationnet_ResNet10', 'relationnet_ResNet18',
    #           'relationnet_ResNet34', 'simpleshot_ResNet34', 'simpleshot_DenseNet121',
    #           'simpleshot_WideRes', 'DeepEMD']
    # all_names = ['maml_approx_ResNet18', 'maml_approx_ResNet34', 'simpleshot_WideRes', 'DeepEMD']

    # # *** Cross Domain ***
    # n_query = 15
    # n_way = 5
    # n_shot = 5
    # n_epochs = 300
    # dataset_name = "cross"
    # all_names = ["maml_approx_ResNet18", "matchingnet_ResNet18",
    #              "protonet_ResNet18", "relationnet_softmax_ResNet18",
    #              "simpleshot_ResNet18"]
    # ens_sizes = np.arange(2, len(all_names) + 1)
    # for ens_size in ens_sizes:
    #     combinations = itertools.combinations(all_names, ens_size)
    #     for comb in combinations:
    #         sv_path = f"ens_checkpoints/{dataset_name}/{'-'.join(comb)}"
    #         sv_path = modify_save_path(sv_path)
    #         run(model_names=all_names, dataset=dataset_name, normalize_flag=normalize, save_dir=sv_path,
    #             nway=n_way, nshot=n_shot)

    all_names = ["matchingnet_ResNet18",
                 "protonet_ResNet18",
                 "simpleshot_ResNet18"]

    sv_path = f"ens_checkpoints/{dataset_name}/{'-'.join(all_names)}"
    sv_path = modify_save_path(sv_path)
    sv_path += f"_{n_way}way_{n_shot}shot"
    run(model_names=all_names, dataset=dataset_name, normalize_flag=normalize, save_dir=sv_path,
        nway=n_way, nshot=n_shot)

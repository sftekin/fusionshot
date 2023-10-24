import os
import argparse
import pickle as pkl
import numpy as np
import scipy
import torch.nn
import torch.nn as nn
from torch.utils.data import DataLoader

MODEL_OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "base_model_src", "inference_out")


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


def load_logits(model_names, dataset, class_type, nway, nshot, ep_count=600):
    logits = []
    for model_n in model_names:
        method_str = model_n.split("_")
        if len(method_str) > 1:
            extension_name = method_str[1]
            method_name = f"{method_str[0]}_{extension_name}" if extension_name in ["approx", "softmax"] else \
                method_str[0]
        else:
            method_name = method_str[0]

        with open(
                f"{MODEL_OUT_DIR}/{dataset}/model_outs/{method_name}/{model_n}_{class_type}_{nway}way_{nshot}shot.pkl",
                "rb") as f:
            results = pkl.load(f)
        logit = results["logits"]
        # logit = np.load(f"{inderence_out_dir}/{dataset}/{method_name}/{model_n}_{class_type}_{nway}way_{nshot}shot_logits.npy")
        if method_name == "DeepEMD":
            logit = np.transpose(logit.reshape(ep_count, 15, 5, 5), axes=[0, 2, 1, 3])
            logit = logit.reshape(ep_count, 75, 5)
        elif method_name == "simpleshot":
            logit *= -1
        logits.append(logit.reshape(-1, 5))

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


def run(model_names, dataset, save_dir, n_way, n_shot, n_query, n_epochs):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_logits = load_logits(model_names, dataset=dataset, nway=n_way, nshot=n_shot,
                               class_type="base")
    train_data = create_data(train_logits, n_way, n_query)

    val_logits = load_logits(model_names, dataset=dataset, nway=n_way, nshot=n_shot,
                             class_type="val")
    val_data = create_data(val_logits, n_way, n_query)

    novel_logits = load_logits(model_names, dataset=dataset, nway=n_way, nshot=n_shot,
                               class_type="novel")
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
    parser = argparse.ArgumentParser(description='focal diversity pruning')
    parser.add_argument('--dataset_name', default="miniImagenet", choices=["CUB", "miniImagenet"])
    parser.add_argument("--n_query", default=15, type=int)
    parser.add_argument("--n_way", default=5, type=int)
    parser.add_argument("--n_shot", default=1, type=int)
    parser.add_argument("--n_epochs", default=300, type=int)
    parser.add_argument('--model_names', nargs='+',
                        help='Model name and backbone e.g. protonet_ResNet18', required=True)
    args = parser.parse_args()

    sv_path = f"ens_checkpoints/{args.dataset_name}/{'-'.join(args.model_names)}"
    sv_path = modify_save_path(sv_path)
    sv_path += f"_{args.n_way}way_{args.n_shot}shot"
    run(model_names=args.model_names, dataset=args.dataset_name, save_dir=sv_path,
        n_way=args.n_way, n_shot=args.n_shot, n_query=args.n_query, n_epochs=args.n_epochs)

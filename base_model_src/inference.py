import os
import time
import torch
import numpy as np
import random
import pickle as pkl
import argparse
from methods.emd_utils import deep_emd_episode
from methods.protonet import euclidean_dist
from methods.simpleshot_utils import ss_step, ss_episode
from data_manager.episode_loader import get_episode_loader
from data_manager.simple_loader import get_simple_loader
from utils import load_model, get_image_size
import torch.multiprocessing
from data_manager.simple_loader import get_cpu_loader

torch.multiprocessing.set_sharing_strategy('file_system')
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATHS = {
    "CUB": f'{CUR_PATH}/filelists/CUB/',
    "miniImagenet": f"{CUR_PATH}/filelists/miniImagenet/"
}


def infer(model, loader, mode, method, model_name, n_query, n_way, n_shot, **kwargs):
    print(f"obtaining {mode} outputs")
    acc_all = []
    logits = np.zeros((len(loader), n_query * n_way, n_way))
    predicts = np.zeros((len(loader), n_query * n_way))
    negatives = np.zeros((len(loader), n_query * n_way))
    labels = np.zeros((len(loader), n_query * n_way))
    one_forward_pass = 0
    train_features = kwargs.get("train_features", None)
    if train_features is not None:
        mu = train_features.reshape(-1, train_features.shape[2]).mean(dim=0)
        # train_features = train_features - mu
        # train_features = train_features / torch.norm(train_features, p=2, dim=1, keepdim=True)
    if method == "easy":
        for m in model:
            m.eval()

    for i, (x, y) in enumerate(loader):
        if method == "DeepEMD":
            with torch.no_grad():
                start_time = time.time()
                model.eval()
                scores = deep_emd_episode(model, x, y, n_way=n_way, n_support=n_shot, n_query=n_query)
                one_forward_pass += time.time() - start_time
                y_query = np.tile(range(n_way), n_query)
                pred = scores.argmax(dim=1).detach().cpu().numpy()
                logits[i, :] = scores.detach().cpu().numpy()
        elif method == "easy":
            x = x.reshape(n_way * (n_shot + n_query), *x.size()[2:]).cuda()
            with torch.no_grad():
                features = torch.cat([m(x)[1] for m in model], dim=1)
                # _, features = model(x)
                features = features - mu.unsqueeze(0)
                features = features / torch.norm(features, p=2, dim=1, keepdim=True)
                features = features.reshape(n_way, n_shot + n_query, -1)
                z_support = features[:, :n_shot]
                z_query = features[:, n_shot:]
                means = z_support.mean(dim=1)
                dist = euclidean_dist(z_query.reshape(n_way * n_query, -1), means)
                # means = torch.mean(features[:, :n_shot], dim=1)
                # distances = torch.norm(
                #     features[:, n_shot:].reshape(1, n_way, 1, -1, 1920) - means.reshape(
                #         1, 1, n_way, 1, 1920), dim=4, p=2)
                scores = -dist
            pred = scores.argmax(dim=1).detach().cpu().numpy()
            logits[i, :] = scores.detach().cpu().numpy()
            y_query = np.repeat(range(n_way), n_query)
        elif "simpleshot" in method:
            with torch.no_grad():
                start_time = time.time()
                pred, distance = ss_episode(model, x, n_way, n_shot, n_query, out_mean=kwargs["base_mean"])
                one_forward_pass += time.time() - start_time
                logits[i, :] = distance.T
                y_query = np.repeat(range(n_way), n_query)
                pred = pred.squeeze()
        else:
            model.n_query = x.size(1) - n_shot
            start_time = time.time()
            scores = model.set_forward(x)
            one_forward_pass += time.time() - start_time
            y_query = np.repeat(range(n_way), model.n_query)
            pred = scores.data.cpu().numpy().argmax(axis=1)
            logits[i, :] = scores.detach().cpu().numpy()

        predicts[i, :] = pred
        labels[i, :] = y[:, n_shot:].flatten()
        negatives[i, pred != y_query] = 1
        corrects = np.sum(pred == y_query)
        acc = corrects / len(y_query) * 100

        print(f"\rEpisode {i + 1} / {len(loader)}: {acc:.2f}", end="", flush=True)
        acc_all.append(acc)

    print(f"\nApprox, one forward pass takes {one_forward_pass / len(loader):.3f} seconds")

    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print(f'{mode}->{len(loader)} Acc = {acc_mean:.2f} +- {1.96 * acc_std / np.sqrt(len(loader)):.2f}')

    model_outs_dir = os.path.join(CUR_PATH, "inference_out", kwargs["dataset_name"], "model_outs", method)
    if not os.path.exists(model_outs_dir):
        os.makedirs(model_outs_dir)

    save_str = f"{method}_{model_name}_{mode}" if method != "DeepEMD" else f"{method}_{mode}"
    save_str += f"_{n_way}way_{n_shot}shot"
    print(f"saving to {model_outs_dir}/{save_str}")
    results = {
        "logits": logits,
        "predicts": predicts,
        "negatives": negatives,
        "labels": labels
    }
    save_path = os.path.join(model_outs_dir, f"{save_str}.pkl")
    with open(save_path, "wb") as f:
        pkl.dump(results, f)


def calc_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('model size: {:.3f}MB'.format(size_all_mb))


def get_features(model, imagesize, dataset_name, class_name, num_classes, n_shot, n_way):
    for m in model:
        m.eval()
    save_path = f"{class_name}_features.tar"
    if os.path.exists(save_path):
        features_total = torch.load(save_path)
    else:
        # simple_loader = get_simple_loader(meta_file_path=DATA_PATHS[dataset_name] + f"{class_name}.json",
        #                                   image_size=imagesize, batch_size=64,
        #                                   num_workers=8, augmentation=False)
        simple_loader = get_cpu_loader(imagesize, class_name)
        all_features = []
        for batch_idx, (data, target) in enumerate(simple_loader):
            print(f"\r{class_name} Episode {batch_idx} / {len(simple_loader)}", end="", flush=True)
            with torch.no_grad():
                data, target = data.cuda(), target.cuda()
                features = torch.cat([m(data)[1] for m in model], dim=1)
                # _, features = model(data)
                all_features.append(features)
        features_total = torch.cat(all_features, dim=0).reshape(num_classes, -1, all_features[0].shape[1])
        torch.save(features_total, f"{class_name}_features.tar")

    return features_total


def run(method, dataset_name, class_type, ep_num, model_name,
        n_query, n_way, n_shot, aug_used=False, cross=False):
    base_file = DATA_PATHS[dataset_name] + f'{class_type}.json'

    image_size = get_image_size(method, model_name)

    loader = get_episode_loader(meta_file_path=base_file, image_size=image_size, n_episodes=ep_num,
                                augmentation=False, n_way=n_way, n_shot=n_shot, n_query=n_query,
                                num_workers=8, load_sampler_indexes=True, dataset_name=dataset_name)

    model = load_model(method=method, model_name=model_name, n_way=n_way, n_shot=n_shot, n_query=n_query,
                       dataset_name=dataset_name, aug_used=aug_used, cross=cross, args=args)

    # prep infer args
    infer_args = {
        "model": model,
        "loader": loader,
        "method": method,
        "model_name": model_name,
        "mode": class_type,
        "n_query": n_query,
        "n_way": n_way,
        "n_shot": n_shot,
        "save_features": False,
        "dataset_name": "cross" if cross else dataset_name
    }

    # calc_model_size(model)

    if "easy" in method:
        train_features = get_features(model, image_size, dataset_name, "base", 64, n_way, n_shot)
        # val_features = get_features(model, image_size, dataset_name, "val", 16)
        # novel_features = get_features(model, image_size, dataset_name, "novel", 20)
        infer_args["train_features"] = train_features
        # infer_args["val_features"] = val_features,
        # infer_args["novel_features"] = novel_features

    if "simpleshot" in method:
        save_path = f"simple_shot_base_features_{method}_{model_name}.tar"
        if os.path.exists(save_path):
            base_mean = torch.load(save_path)
        else:
            print("Simple shot requires the mean of the features extracted from base dataset")
            base_ds_path = DATA_PATHS[dataset_name] + f'base.json'
            base_loader = get_episode_loader(meta_file_path=base_ds_path, image_size=image_size, n_episodes=ep_num,
                                             augmentation=False, n_way=n_way, n_shot=n_shot, n_query=n_query,
                                             num_workers=8, load_sampler_indexes=True, dataset_name=dataset_name)
            base_mean = []
            with torch.no_grad():
                for i, (x, y) in enumerate(base_loader):
                    print(f"\rBase Episode {i} / {len(base_loader)}", end="", flush=True)
                    output, fc_output = ss_step(model, x, n_way, n_shot, n_query)
                    base_mean.append(output.detach().cpu().data.numpy())
            base_mean = np.concatenate(base_mean, axis=0).mean(0)
            torch.save(base_mean, save_path)
        infer_args["base_mean"] = base_mean

    # train
    infer(**infer_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='few-shot inference')
    parser.add_argument('--seed', default=50, type=int)
    parser.add_argument('--dataset_name', default="miniImagenet", choices=["CUB", "miniImagenet"])
    parser.add_argument('--method', default='simpleshot',
                        choices=["maml_approx", "matchingnet", "protonet", "relationnet",
                                 "relationnet_softmax", "DeepEMD",
                                 "simpleshot", "easy"])
    parser.add_argument('--model_name', default="Conv6", choices=['Conv4', 'Conv6', 'ResNet10', 'ResNet18',
                                                                     'ResNet34', "WideRes", "DenseNet121"])
    parser.add_argument('--class_type', default="novel", choices=["base", "val", "novel"])
    parser.add_argument("--n_query", default=15, type=int)
    parser.add_argument("--n_way", default=5, type=int)
    parser.add_argument("--n_shot", default=1, type=int)
    parser.add_argument('--ep_num', default=600, type=int)
    parser.add_argument('--aug_used', action='store_true', help='performed train augmentation')
    parser.add_argument('--cross', action='store_true')

    # Additional Deep Emd Arguments
    parser.add_argument('-temperature', type=float, default=12.5)
    parser.add_argument('-metric', type=str, default='cosine', choices=['cosine'])
    parser.add_argument('-norm', type=str, default='center', choices=['center'])
    parser.add_argument('-deepemd', type=str, default='fcn', choices=['fcn', 'grid', 'sampling'])
    parser.add_argument('-feature_pyramid', type=str, default=None)
    parser.add_argument('-num_patch', type=int, default=9)
    parser.add_argument('-patch_list', type=str, default='2,3')
    parser.add_argument('-patch_ratio', type=float, default=2)
    parser.add_argument('-solver', type=str, default='opencv', choices=['opencv'])
    parser.add_argument('-sfc_lr', type=float, default=100)
    parser.add_argument('-sfc_wd', type=float, default=0, help='weight decay for SFC weight')
    parser.add_argument('-sfc_update_step', type=float, default=100)
    parser.add_argument('-sfc_bs', type=int, default=4)

    args = parser.parse_args()
    print(vars(args))

    # set the seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    run(method=args.method,
        dataset_name=args.dataset_name,
        class_type=args.class_type,
        ep_num=args.ep_num,
        model_name=args.model_name,
        aug_used=args.aug_used,
        cross=args.cross,
        n_query=args.n_query,
        n_shot=args.n_shot,
        n_way=args.n_way)

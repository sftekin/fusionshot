import os
import torch
import glob
import backbone
import numpy as np
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from methods.deep_emd import DeepEMD
import methods.ss_backbones as ss_backbones
from methods.emd_utils import emd_load_model

model_dict = dict(
    Conv4=backbone.Conv4,
    Conv4S=backbone.Conv4S,
    Conv6=backbone.Conv6,
    ResNet10=backbone.ResNet10,
    ResNet18=backbone.ResNet18,
    ResNet34=backbone.ResNet34,
    ResNet50=backbone.ResNet50,
    ResNet101=backbone.ResNet101)

CUR_PATH = os.path.dirname(os.path.abspath(__file__))


def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)


def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist = [x for x in filelist if os.path.basename(x) != 'best_model.tar']
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file


def one_hot(y, num_class):
    return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1), 1)


def get_image_size(method, model_name):
    if method == "DeepEMD":
        image_size = 84
    elif "simpleshot" in method:
        if model_name.lower() in ["conv4", "conv6"]:
            image_size = 84
        else:
            image_size = 96
    else:
        if "Conv" in model_name:
            image_size = 84
        else:
            image_size = 224
    return image_size


def load_model(method, model_name, n_way, n_shot, n_query, dataset_name, args, aug_used=False, cross=False):
    if cross:
        trained_dataset = "miniImagenet"
    else:
        trained_dataset = dataset_name

    if method in ['relationnet', 'relationnet_softmax']:
        if 'Conv4' in model_name:
            feature_model = backbone.Conv4NP
        elif 'Conv6' in model_name:
            feature_model = backbone.Conv6NP
        else:
            feature_model = lambda: model_dict[model_name](flatten=False)
        loss_type = 'mse' if method == 'relationnet' else 'softmax'

        model = RelationNet(feature_model, loss_type=loss_type, n_way=n_way, n_support=n_shot)
    elif method in ['maml', 'maml_approx']:
        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.BottleneckBlock.maml = True
        backbone.ResNet.maml = True
        model = MAML(model_dict[model_name], approx=(method == 'maml_approx'), n_way=n_way, n_support=n_shot)
    elif method == "protonet":
        model = ProtoNet(model_dict[model_name], n_way=n_way, n_support=n_shot)
    elif method == "matchingnet":
        model = MatchingNet(model_dict[model_name], n_way=n_way, n_support=n_shot)
    elif method == "DeepEMD":
        model = DeepEMD(args=args)
        model = model.cuda()
    elif "simpleshot" in method:
        bb_model = model_name.lower()
        bb_mapper = {
            "conv4": ss_backbones.conv4,
            "conv6": ss_backbones.conv6,
            "resnet10": ss_backbones.resnet10,
            "resnet18": ss_backbones.resnet18,
            "resnet34": ss_backbones.resnet34,
            "resnet50": ss_backbones.resnet50,
            "wideres": ss_backbones.wideres,
            "densenet121": ss_backbones.densenet121
        }
        if cross:
            trained_dataset = "miniImagenet"
        else:
            trained_dataset = dataset_name
        num_classes = 100 if trained_dataset == "CUB" else 64
        model = bb_mapper[bb_model](num_classes=num_classes, remove_linear=False)
        model = torch.nn.DataParallel(model).cuda()
    else:
        raise ValueError

    if method == "DeepEMD":
        model = emd_load_model(model, dir=f"{CUR_PATH}/checkpoints"
                                          f"/{trained_dataset}/DeepEMD/{n_shot}shot-{n_way}way/max_acc.pth",
                               mode="cuda")
    elif "simpleshot" in method:
        save_path = f"{CUR_PATH}/checkpoints/{trained_dataset}/SimpleShot/{bb_model}/checkpoint.pth.tar"
        tmp = torch.load(save_path)
        model.load_state_dict(tmp["state_dict"])
    else:
        checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (CUR_PATH, trained_dataset, model_name, method)
        if aug_used:
            checkpoint_dir += "_aug"
        checkpoint_dir += '_%dway_%dshot' % (n_way, n_shot)

        modelfile = get_best_file(checkpoint_dir)

        model = model.cuda()
        tmp = torch.load(modelfile)
        model.load_state_dict(tmp['state'])

    return model

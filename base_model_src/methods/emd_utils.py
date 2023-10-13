# Adapted from https://github.com/icoz69/DeepEMD

import argparse

import cv2
import torch
import torch.nn.functional as F
from qpth.qp import QPFunction
from torch.autograd import Variable


def emd_inference_qpth(distance_matrix, weight1, weight2, form='QP', l2_strength=0.0001):
    """
    to use the QP solver QPTH to derive EMD (LP problem),
    one can transform the LP problem to QP,
    or omit the QP term by multiplying it with a small value,i.e. l2_strngth.
    :param distance_matrix: nbatch * element_number * element_number
    :param weight1: nbatch  * weight_number
    :param weight2: nbatch  * weight_number
    :return:
    emd distance: nbatch*1
    flow : nbatch * weight_number *weight_number

    """

    weight1 = (weight1 * weight1.shape[-1]) / weight1.sum(1).unsqueeze(1)
    weight2 = (weight2 * weight2.shape[-1]) / weight2.sum(1).unsqueeze(1)

    nbatch = distance_matrix.shape[0]
    nelement_distmatrix = distance_matrix.shape[1] * distance_matrix.shape[2]
    nelement_weight1 = weight1.shape[1]
    nelement_weight2 = weight2.shape[1]

    Q_1 = distance_matrix.view(-1, 1, nelement_distmatrix).double()

    if form == 'QP':
        # version: QTQ
        Q = torch.bmm(Q_1.transpose(2, 1), Q_1).double().cuda() + 1e-4 * torch.eye(
            nelement_distmatrix).double().cuda().unsqueeze(0).repeat(nbatch, 1, 1)  # 0.00001 *
        p = torch.zeros(nbatch, nelement_distmatrix).double().cuda()
    elif form == 'L2':
        # version: regularizer
        Q = (l2_strength * torch.eye(nelement_distmatrix).double()).cuda().unsqueeze(0).repeat(nbatch, 1, 1)
        p = distance_matrix.view(nbatch, nelement_distmatrix).double()
    else:
        raise ValueError('Unkown form')

    h_1 = torch.zeros(nbatch, nelement_distmatrix).double().cuda()
    h_2 = torch.cat([weight1, weight2], 1).double()
    h = torch.cat((h_1, h_2), 1)

    G_1 = -torch.eye(nelement_distmatrix).double().cuda().unsqueeze(0).repeat(nbatch, 1, 1)
    G_2 = torch.zeros([nbatch, nelement_weight1 + nelement_weight2, nelement_distmatrix]).double().cuda()
    # sum_j(xij) = si
    for i in range(nelement_weight1):
        G_2[:, i, nelement_weight2 * i:nelement_weight2 * (i + 1)] = 1
    # sum_i(xij) = dj
    for j in range(nelement_weight2):
        G_2[:, nelement_weight1 + j, j::nelement_weight2] = 1
    # xij>=0, sum_j(xij) <= si,sum_i(xij) <= dj, sum_ij(x_ij) = min(sum(si), sum(dj))
    G = torch.cat((G_1, G_2), 1)
    A = torch.ones(nbatch, 1, nelement_distmatrix).double().cuda()
    b = torch.min(torch.sum(weight1, 1), torch.sum(weight2, 1)).unsqueeze(1).double()
    flow = QPFunction(verbose=-1)(Q, p, G, h, A, b)

    emd_score = torch.sum((1 - Q_1).squeeze() * flow, 1)
    return emd_score, flow.view(-1, nelement_weight1, nelement_weight2)


def emd_inference_opencv(cost_matrix, weight1, weight2):
    # cost matrix is a tensor of shape [N,N]
    cost_matrix = cost_matrix.detach().cpu().numpy()

    weight1 = F.relu(weight1) + 1e-5
    weight2 = F.relu(weight2) + 1e-5

    weight1 = (weight1 * (weight1.shape[0] / weight1.sum().item())).view(-1, 1).detach().cpu().numpy()
    weight2 = (weight2 * (weight2.shape[0] / weight2.sum().item())).view(-1, 1).detach().cpu().numpy()

    cost, _, flow = cv2.EMD(weight1, weight2, cv2.DIST_USER, cost_matrix)
    return cost, flow


def emd_inference_opencv_test(distance_matrix, weight1, weight2):
    distance_list = []
    flow_list = []

    for i in range(distance_matrix.shape[0]):
        cost, flow = emd_inference_opencv(distance_matrix[i], weight1[i], weight2[i])
        distance_list.append(cost)
        flow_list.append(torch.from_numpy(flow))

    emd_distance = torch.Tensor(distance_list).cuda().double()
    flow = torch.stack(flow_list, dim=0).cuda().double()

    return emd_distance, flow


def deep_emd_episode(model, x, y, n_way, n_support, n_query):
    x = x.transpose(0, 1)
    x = Variable(x.cuda())
    x = x.contiguous().view(n_way * (n_support + n_query), * x.size()[2:])

    model.mode = 'encoder'
    data = model(x)
    k = n_way * n_support
    data_shot, data_query = data[:k], data[k:]
    if n_support > 1:
        data_shot = model.get_sfc(data_shot)
    model.mode = 'meta'
    logits = model((data_shot, data_query))

    return logits


def emd_load_model(model, dir, mode="cuda"):
    model_dict = model.state_dict()
    print('loading model from :', dir)
    if mode == "cuda":
        pretrained_dict = torch.load(dir)['params']
    else:
        pretrained_dict = torch.load(dir, map_location=torch.device('cpu'))['params']

    if 'encoder' in list(pretrained_dict.keys())[0]:  # load from a parallel meta-trained model
        if 'module' in list(pretrained_dict.keys())[0]:
            pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
        else:
            pretrained_dict = {k: v for k, v in pretrained_dict.items()}
    else:
        pretrained_dict = {'encoder.' + k: v for k, v in pretrained_dict.items()}  # load from a pretrained model
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)  # update the param in encoder, remain others still
    model.load_state_dict(model_dict)

    return model


def get_deep_emd_args(way, shot, query):
    parser = argparse.ArgumentParser()
    parser.add_argument('-way', type=int, default=way)
    parser.add_argument('-shot', type=int, default=shot)
    parser.add_argument('-query', type=int, default=query, help='number of query image per class')
    parser.add_argument('--cross', action='store_true')

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

    parser.add_argument('--method', default='DeepEMD')
    parser.add_argument('--data_set', default="novel", choices=["base", "val", "novel"])
    parser.add_argument('--ep_num', default=1000, type=int)
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    random_seed = True
    if random_seed:
        pass
    else:

        seed = 1
        import random
        import numpy as np

        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    batch_size = 50
    num_node = 25
    form = 'L2'  # in [ 'L2', 'QP' ]

    cosine_distance_matrix = torch.rand(batch_size, num_node, num_node).cuda()

    weight1 = torch.rand(batch_size, num_node).cuda()
    weight2 = torch.rand(batch_size, num_node).cuda()

    emd_distance_cv, cv_flow = emd_inference_opencv_test(cosine_distance_matrix, weight1, weight2)
    emd_distance_qpth, qpth_flow = emd_inference_qpth(cosine_distance_matrix, weight1, weight2, form=form)

    emd_score_cv = ((1 - cosine_distance_matrix) * cv_flow).sum(-1).sum(-1)
    emd_score_qpth = ((1 - cosine_distance_matrix) * qpth_flow).sum(-1).sum(-1)
    print('emd difference:', (emd_score_cv - emd_score_qpth).abs().max())
    pass

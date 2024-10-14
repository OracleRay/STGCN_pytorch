import logging
import os
import gc
import argparse
import math
import random
import warnings
import tqdm
import numpy as np
import pandas as pd
from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

from script import dataloader, utility, earlystopping, opt
from model import models


# import nni

# 设置和配置一些环境变量，以确保实验的稳定性和可重复性
def set_env(seed):
    # Set available CUDA devices
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)  # 指定GPU设备

    # 设置伪随机数生成器的种子，确保在不同运行之间获得相同的随机数
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False  # 确保运行时不会对卷积算法进行优化，提高稳定性
    torch.backends.cudnn.deterministic = True  # 启用确定性的CUDA操作，以确保相同输入下的结果相同
    # torch.use_deterministic_algorithms(True)


# 解析命令行参数，配置模型的各种超参数，以及确定是否使用CUDA
def get_parameters():
    parser = argparse.ArgumentParser(description='STGCN')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--dataset', type=str, default='metr-la', choices=['metr-la', 'pems-bay', 'pemsd7-m'])
    parser.add_argument('--n_his', type=int, default=12)  # 历史时间步数
    parser.add_argument('--n_pred', type=int, default=3,
                        help='the number of time interval for predcition, default as 3')  # 预测时间步数
    parser.add_argument('--time_intvl', type=int, default=5)
    parser.add_argument('--Kt', type=int, default=3)  # 时间卷积核大小
    parser.add_argument('--stblock_num', type=int, default=2)  # 时空卷积块的数量
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'])  # 激活函数类型
    parser.add_argument('--Ks', type=int, default=3, choices=[3, 2])  # 空间卷积核大小
    parser.add_argument('--graph_conv_type', type=str, default='cheb_graph_conv',
                        choices=['cheb_graph_conv', 'graph_conv'])
    parser.add_argument('--gso_type', type=str, default='sym_norm_lap',
                        choices=['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj'])
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    parser.add_argument('--droprate', type=float, default=0.5)  # dropout率
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay_rate', type=float, default=0.001, help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1000, help='epochs, default as 1000')
    parser.add_argument('--opt', type=str, default='nadamw', choices=['adamw', 'nadamw', 'lion'],
                        help='optimizer, default as nadamw')
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    args = parser.parse_args()
    print('Training configs: {}'.format(args))

    # For stable experiment results
    set_env(args.seed)

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device('cuda')
        torch.cuda.empty_cache()  # Clean cache
    else:
        device = torch.device('cpu')
        gc.collect()  # Clean cache

    # Ko: 模型将使用多少有效的时间步进行预测
    # 每经过1个时空卷积块，时间步减少(args.Kt - 1) * 2
    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num

    # blocks: 定义模型中时空卷积块的结构和输出层的通道数
    # using the bottleneck design in st_conv_blocks
    blocks = []
    blocks.append([1])  # 输入层
    for l in range(args.stblock_num):
        blocks.append([64, 16, 64])  # 每个时空卷积块，有64个输入通道、16个瓶颈通道和64个输出通道
    if Ko == 0:
        blocks.append([128])  # Ko=0的输出层，模型的最终输出是一个单一的预测值，因为模型不能使用足够的历史时间步进行预测
    elif Ko > 0:
        blocks.append([128, 128])  # Ko>0的输出层，要充分利用保留下来的历史信息并允许更复杂的输出
    blocks.append([1])  # 最终输出层

    return args, device, blocks


# 数据准备
def data_preparate(args, device):
    adj, n_vertex = dataloader.load_adj(args.dataset)  # 加载数据集的邻接矩阵(adj)和节点数量(n_vertex)
    gso = utility.calc_gso(adj, args.gso_type)  # 计算图拉普拉斯
    if args.graph_conv_type == 'cheb_graph_conv':
        gso = utility.calc_chebynet_gso(gso)  # 计算切比雪夫的拉普拉斯矩阵
    gso = gso.toarray()  # 将 Laplacian 矩阵从稀疏矩阵格式转换为 NumPy 数组
    gso = gso.astype(dtype=np.float32)
    args.gso = torch.from_numpy(gso).to(device)  # 将拉普拉斯矩阵转换为张量格式，并将其发送到指定的设备(CPU/GPU)

    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, args.dataset)
    data_col = pd.read_csv(os.path.join(dataset_path, 'vel.csv')).shape[0]
    # recommended dataset split rate as train: val: test = 60: 20: 20, 70: 15: 15 or 80: 10: 10
    # using dataset split rate as train: val: test = 70: 15: 15
    val_and_test_rate = 0.15  # 数据集中将有 15% 的数据分别分配给验证集和测试集

    len_val = int(math.floor(data_col * val_and_test_rate))
    len_test = int(math.floor(data_col * val_and_test_rate))
    len_train = int(data_col - len_val - len_test)

    # 根据指定的划分比例，将数据集分为训练、验证和测试集
    train, val, test = dataloader.load_data(args.dataset, len_train, len_val)
    zscore = preprocessing.StandardScaler()
    train = zscore.fit_transform(train)
    val = zscore.transform(val)
    test = zscore.transform(test)

    # x_data: 输入数据，y_data: 输出数据
    x_train, y_train = dataloader.data_transform(train, args.n_his, args.n_pred, device)
    x_val, y_val = dataloader.data_transform(val, args.n_his, args.n_pred, device)
    x_test, y_test = dataloader.data_transform(test, args.n_his, args.n_pred, device)

    # 将 x_train（输入） 和 y_train（目标输出） 绑定在一起，形成一个包含所有训练数据的 train_data，方便后续按批次取出数据
    train_data = utils.data.TensorDataset(x_train, y_train)
    train_iter = utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False)  # 生成可迭代对象
    val_data = utils.data.TensorDataset(x_val, y_val)
    val_iter = utils.data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
    test_data = utils.data.TensorDataset(x_test, y_test)
    test_iter = utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    return n_vertex, zscore, train_iter, val_iter, test_iter


# 准备神经网络模型
def prepare_model(args, blocks, n_vertex):
    loss = nn.MSELoss()
    es = earlystopping.EarlyStopping(delta=0.0,
                                     patience=args.patience,
                                     verbose=True,
                                     path="STCGN_" + args.dataset + ".pt")

    # 选择使用的模型
    if args.graph_conv_type == 'cheb_graph_conv':
        model = models.STGCNChebGraphConv(args, blocks, n_vertex).to(device)
    else:
        model = models.STGCNGraphConv(args, blocks, n_vertex).to(device)

    # 选择优化器
    if args.opt == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    elif args.opt == "nadamw":
        optimizer = optim.NAdam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate,
                                decoupled_weight_decay=True)
    elif args.opt == "lion":
        optimizer = opt.Lion(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    else:
        raise ValueError(f'ERROR: The {args.opt} optimizer is undefined.')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    return loss, es, model, optimizer, scheduler


def train(args, model, loss, optimizer, scheduler, es, train_iter, val_iter):
    for epoch in range(args.epochs):
        l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        model.train()  # 启用 batch normalization 和 dropout
        for x, y in tqdm.tqdm(train_iter):
            optimizer.zero_grad()
            y_pred = model(x).view(len(x), -1)  # [batch_size, num_nodes]
            l = loss(y_pred, y)
            l.backward()
            optimizer.step()  # 执行一次优化步骤
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        scheduler.step()
        val_loss = val(model, val_iter)  # 验证模型表现
        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB'. \
              format(epoch + 1, optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc))

        es(val_loss, model)
        if es.early_stop:
            print("Early stopping")
            break


# 在不更新模型参数的情况下查看模型在验证数据上的表现
@torch.no_grad()
def val(model, val_iter):
    model.eval()

    l_sum, n = 0.0, 0
    for x, y in val_iter:
        y_pred = model(x).view(len(x), -1)
        l = loss(y_pred, y)
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    return torch.tensor(l_sum / n)


@torch.no_grad()
def _test(zscore, loss, model, test_iter, args):
    model.load_state_dict(torch.load("STGCN_" + args.dataset + ".pt"))
    model.eval()

    test_MSE = utility.evaluate_model(model, loss, test_iter)
    test_MAE, test_RMSE, test_WMAPE = utility.evaluate_metric(model, test_iter, zscore)
    print(
        f'Dataset {args.dataset:s} | Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f}')


if __name__ == "__main__":
    # Logging
    # logger = logging.getLogger('stgcn')
    # logging.basicConfig(filename='stgcn.log', level=logging.INFO)
    logging.basicConfig(level=logging.INFO)

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    args, device, blocks = get_parameters()
    n_vertex, zscore, train_iter, val_iter, test_iter = data_preparate(args, device)
    loss, es, model, optimizer, scheduler = prepare_model(args, blocks, n_vertex)
    train(args, model, loss, optimizer, scheduler, es, train_iter, val_iter)
    _test(zscore, loss, model, test_iter, args)

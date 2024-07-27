import numpy as np
import torch
import torch.nn as nn
from model import FNN
from util import *
from train import *
from torch.autograd import Variable,grad
import time
import pyvista as pv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24
torch.manual_seed(0)
def output_transform(X):
    X = T_range * nn.Softplus()(X) + T_ref
    return X

def input_transform(X):
    X = 2. * (X - X_min) / (X_max - X_min) - 1.
    return X
def PDE(x, y, z, t, net):
    X = torch.cat([x, y, z, t], dim=-1)
    T = net(X)
    T_t = grad(T, t, create_graph=True, grad_outputs=torch.ones_like(T))[0]
    T_x = grad(T, x, create_graph=True, grad_outputs=torch.ones_like(T))[0]
    T_xx = grad(T_x, x, create_graph=True, grad_outputs=torch.ones_like(T_x))[0]
    T_y = grad(T, y, create_graph=True, grad_outputs=torch.ones_like(T))[0]
    T_yy = grad(T_y, y, create_graph=True, grad_outputs=torch.ones_like(T_y))[0]
    T_z = grad(T, z, create_graph=True, grad_outputs=torch.ones_like(T))[0]
    T_zz = grad(T_z, z, create_graph=True, grad_outputs=torch.ones_like(T_z))[0]
    conduction = k * (T_xx + T_yy + T_zz)
    convection = h * (T - T_inf) + U * T_x
    f = rho * Cp * T_t - conduction - convection
    return f
import numpy as np

def sampling_uniform(density, x_range, y_range, z_range=None, t_range=None, sample_type='domain', t=None):
    """
    Generates uniform sampling points in the specified range.
    """
    if z_range is None and t_range is None:
        # 2D sampling
        x = np.random.uniform(x_range[0], x_range[1], int(density))
        y = np.random.uniform(y_range[0], y_range[1], int(density))
        return np.stack([x, y], axis=-1), None
    elif t_range is None:
        # 3D sampling
        x = np.random.uniform(x_range[0], x_range[1], int(density))
        y = np.random.uniform(y_range[0], y_range[1], int(density))
        z = np.random.uniform(z_range[0], z_range[1], int(density))
        return np.stack([x, y, z], axis=-1), None
    else:
        # 4D sampling
        x = np.random.uniform(x_range[0], x_range[1], int(density))
        y = np.random.uniform(y_range[0], y_range[1], int(density))
        z = np.random.uniform(z_range[0], z_range[1], int(density))
        t = np.random.uniform(t_range[0], t_range[1], int(density))
        return np.stack([x, y, z, t], axis=-1), None
def generate_points(p=[], f=[]):
    t = np.linspace(x_min[3] + 0.01, x_max[3], 121)
    
    bound_x_neg, _ = sampling_uniform(2., [x_min[0], x_max[0]], [x_min[1], x_max[1]], [x_min[2], x_max[2]], [t.min(), t.max()], sample_type='-x')
    bound_x_pos, _ = sampling_uniform(2., [x_min[0], x_max[0]], [x_min[1], x_max[1]], [x_min[2], x_max[2]], [t.min(), t.max()], sample_type='+x')
    bound_y_neg, _ = sampling_uniform(2., [x_min[0], x_max[0]], [x_min[1], x_max[1]], [x_min[2], x_max[2]], [t.min(), t.max()], sample_type='-y')
    bound_y_pos, _ = sampling_uniform(2., [x_min[0], x_max[0]], [x_min[1], x_max[1]], [x_min[2], x_max[2]], [t.min(), t.max()], sample_type='+y')
    bound_z_neg, _ = sampling_uniform(2., [x_min[0], x_max[0]], [x_min[1], x_max[1]], [x_min[2], x_max[2]], [t.min(), t.max()], sample_type='-z')
    bound_z_pos, _ = sampling_uniform(2., [x_min[0], x_max[0]], [x_min[1], x_max[1]], [x_min[2], x_max[2]], [t.min(), t.max()], sample_type='+z')

    melt_pool_x_range = [x_min[0], x_max[0]]
    melt_pool_y_range = [x_min[1], x_max[1]]
    melt_pool_z_range = [x_min[2], x_max[2]]
    melt_pool_t_range = [x_min[3], x_max[3]]

    melt_pool_pts, _ = sampling_uniform(8., melt_pool_x_range, melt_pool_y_range, melt_pool_z_range, melt_pool_t_range, 'melt_pool')

    domain_pts, _ = sampling_uniform(4., [x_min[0], x_max[0]], [x_min[1], x_max[1]], [x_min[2], x_max[2]], [x_min[3], x_max[3]], 'domain')
    init_pts, _ = sampling_uniform(4., [x_min[0], x_max[0]], [x_min[1], x_max[1]], [x_min[2], x_max[2]], [0, 0], 'domain')

    p.extend([
        torch.tensor(bound_x_neg, requires_grad=True, dtype=torch.float).to(device),
        torch.tensor(bound_x_pos, requires_grad=True, dtype=torch.float).to(device),
        torch.tensor(bound_y_neg, requires_grad=True, dtype=torch.float).to(device),
        torch.tensor(bound_y_pos, requires_grad=True, dtype=torch.float).to(device),
        torch.tensor(bound_z_neg, requires_grad=True, dtype=torch.float).to(device),
        torch.tensor(bound_z_pos, requires_grad=True, dtype=torch.float).to(device),
        torch.tensor(init_pts, requires_grad=True, dtype=torch.float).to(device),
        torch.tensor(domain_pts, requires_grad=True, dtype=torch.float).to(device),
        torch.tensor(melt_pool_pts, requires_grad=True, dtype=torch.float).to(device)
    ])
    
    f.extend([
        ['BC', '-x'], ['BC', '+x'], ['BC', '-y'], ['BC', '+y'], 
        ['BC', '-z'], ['BC', '+z'], ['IC', T_ref], ['domain'], ['melt_pool']
    ])
    
    return p, f
 
def load_data(p, f, filename, num):
    data = np.load(filename)
    if num != 0:
        np.random.shuffle(data)
        data = data[0:num, :]
    p.extend([torch.tensor(data[:, 0:4], requires_grad=True, dtype=torch.float).to(device)])
    f.extend([['data', torch.tensor(data[:, 4:5], requires_grad=True, dtype=torch.float).to(device)]])
    return p, f

def BC(x, y, z, t, net, loc):
    X = torch.concat([x, y, z, t], axis=-1)
    T = net(X)
    if loc == '-x':
        T_x = grad(T, x, create_graph=True, grad_outputs=torch.ones_like(T))[0]
        return k * T_x - h * (T - T_ref) - Rboltz * emiss * (T**4 - T_ref**4)
    if loc == '+x':
        T_x = grad(T, x, create_graph=True, grad_outputs=torch.ones_like(T))[0]
        return -k * T_x - h * (T - T_ref) - Rboltz * emiss * (T**4 - T_ref**4)
    if loc == '-y':
        T_y = grad(T, y, create_graph=True, grad_outputs=torch.ones_like(T))[0]
        return k * T_y - h * (T - T_ref) - Rboltz * emiss * (T**4 - T_ref**4)
    if loc == '+y':
        T_y = grad(T, y, create_graph=True, grad_outputs=torch.ones_like(T))[0]
        return -k * T_y - h * (T - T_ref) - Rboltz * emiss * (T**4 - T_ref**4)
    if loc == '-z':
        T_t = grad(T, t, create_graph=True, grad_outputs=torch.ones_like(T))[0]
        return T_t
    if loc == '+z':
        T_z = grad(T, z, create_graph=True, grad_outputs=torch.ones_like(T))[0]
        q = 2 * P * eta / torch.pi / r**2 * torch.exp(-2 * (torch.square(x - 12 - v * t) + torch.square(y - 2.5)) / r**2) * (t <= t_end) * (t > 0)
        return -k * T_z - h * (T - T_ref) - Rboltz * emiss * (T**4 - T_ref**4) + q
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='0', help='GPU name')
    parser.add_argument('--output', type=str, default='bareplate', help='output filename')
    parser.add_argument('--T_ref', type=float, default=293.15, help='ambient temperature')
    parser.add_argument('--T_range', type=float, default=3000., help='temperature range')
    parser.add_argument('--xmax', type=float, default=63.9, help='max x')
    parser.add_argument('--xmin', type=float, default=0., help='min x')
    parser.add_argument('--ymax', type=float, default=11.9, help='max y')
    parser.add_argument('--ymin', type=float, default=0., help='min y')
    parser.add_argument('--zmax', type=float, default=6., help='max z')
    parser.add_argument('--zmin', type=float, default=0., help='min z')
    parser.add_argument('--tmax', type=float, default=4, help='max t')
    parser.add_argument('--tmin', type=float, default=0., help='min t')
    parser.add_argument('--Cp', type=float, default=.9, help='specific heat')
    parser.add_argument('--k', type=float, default=.15, help='heat conductivity')
    parser.add_argument('--x0', type=float, default=12., help='toolpath origin x')
    parser.add_argument('--y0', type=float, default=2.5, help='toolpath origin y')
    parser.add_argument('--r', type=float, default=1, help='beam radius')
    parser.add_argument('--v', type=float, default=11., help='scan speed')
    parser.add_argument('--t_end', type=float, default=5, help='laser stop time')
    parser.add_argument('--h', type=float, default=2e-5, help='convection coefficient')
    parser.add_argument('--eta', type=float, default=.5, help='absorptivity')
    parser.add_argument('--P', type=float, default=1200., help='laser power')
    parser.add_argument('--emiss', type=float, default=.3, help='emissivity')
    parser.add_argument('--rho', type=float, default=2.7e-3, help='rho')
    parser.add_argument('--iters', type=int, default=50000, help='number of iters')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--data', type=str, default='../Temputerdata/combinedtrain.npy', help='filename, default:None')
    parser.add_argument('--data_num', type=int, default=..., help='number of training data used, 0 for all data')
    parser.add_argument('--calib_eta', type=bool, default=False, help='calibrate eta')
    parser.add_argument('--calib_material', type=bool, default=False, help='calibrate cp and k')
    parser.add_argument('--valid', type=str, default='../Temputerdata/combinedval.npy', help='validation data file')
    parser.add_argument('--pretrain', type=str, default='../{}.pt', help='pretrained model file')#使用Source task数据训练之后得到的预训练权重
    parser.add_argument('--T_inf', type=float, default=293.15, help='ambient temperature in Kelvin')
    parser.add_argument('--U', type=float, default=1., help='convection velocity')
    args = parser.parse_args()

    # 设置设备和参数
    device = torch.device(args.device)
    U = args.U

    x_max = np.array([args.xmax, args.ymax, args.zmax, args.tmax])
    x_min = np.array([args.xmin, args.ymin, args.zmin, args.tmin])
    X_max = torch.tensor(x_max, dtype=torch.float).to(device)
    X_min = torch.tensor(x_min, dtype=torch.float).to(device)

    r = args.r
    v = args.v
    t_end = args.t_end
    P = args.P
    eta = args.eta

    T_ref = args.T_ref
    T_range = args.T_range
    T_inf = args.T_inf

    Cp = args.Cp
    k = args.k
    h = args.h
    Rboltz = 5.6704e-14
    emiss = args.emiss
    rho = args.rho

    data = np.load(args.valid)
    test_in = torch.tensor(data[:, 0:4], requires_grad=False, dtype=torch.float).to(device)
    test_out = torch.tensor(data[:, 4:5], requires_grad=False, dtype=torch.float).to(device)

    iterations = 50000
    lr = args.lr

    net = FNN([4, 64, 64, 64, 1], activation=nn.Tanh(), in_tf=input_transform, out_tf=output_transform)
    net.to(device)
    if args.pretrain != 'None':
        net.load_state_dict(torch.load(args.pretrain))

    point_sets, flags = generate_points([], [])
    if args.data != 'None':
        point_sets, flags = load_data(point_sets, flags, args.data, args.data_num)

    inv_params = []
    if args.calib_eta:
        eta = torch.tensor(1e-5, requires_grad=True, device=device)
        inv_params.append(eta)

    if args.calib_material:
        Cp = torch.tensor(1e-5, requires_grad=True, device=device)
        inv_params.append(Cp)
        k = torch.tensor(1e-5, requires_grad=True, device=device)
        inv_params.append(k)

    l_history, err_history, pde_loss_history, bc_loss_history, ic_loss_history, data_loss_history = train(
        net,
        lambda x, y, z, t, net=net: PDE(x, y, z, t, net),
        BC,
        point_sets,
        flags,
        iterations,
        lr=lr,
        info_num=100,
        w=[1., 1e-4, 1., 1e-4],
        inv_params=inv_params,
        test_in=test_in,
        test_out=test_out
    )

    torch.save(net.state_dict(), '../{}.pt')
    np.save('../total_loss.npy', l_history)
    np.save('..total_err.npy', err_history)
    np.save('../pde_loss.npy', pde_loss_history)
    np.save('../bc_loss.npy', bc_loss_history)
    np.save('../ic_loss.npy', ic_loss_history)
    np.save('../data_loss.npy', data_loss_history)












    





























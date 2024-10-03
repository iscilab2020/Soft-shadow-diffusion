import random

import numpy as np
import torch
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import torch.distributed as dist
import os
import torch
import numpy as np
import random
import time
import logging
import logging.handlers
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import LambdaLR
import math
import numbers
import math
import torch
from torchvision.utils import make_grid

from torch.optim.lr_scheduler import LambdaLR

# torch.set_float32_matmul_precision('high')



THOUSAND = 1000
MILLION = 1000000

STANDARD_MEAN = 0.5
STANDARD_STD = 0.5


def reparameterize_gaussian(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(std.size()).to(mean)
    return mean + std * eps


def gaussian_entropy(logvar):
    const = 0.5 * float(logvar.size(1)) * (1. + np.log(np.pi * 2))
    ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
    return ent


def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * np.log(2 * np.pi)
    return log_z - z.pow(2) / 2


def truncated_normal_(tensor, mean=0, std=1, trunc_std=2):
    """
    Taken from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor



def Standard_Normalize(image):
    return (image - STANDARD_MEAN)/STANDARD_STD


def show_point_cloud(point_cloud, axis=False):
    """visual a point cloud
    Args:
        point_cloud (np.ndarray): the coordinates of point cloud
        axis (bool, optional): Hid the coordinate of the matplotlib. Defaults to False.
    """
    ax = plt.figure().add_subplot(projection='3d')
    ax._axis3don = axis
    ax.scatter(xs=point_cloud[:, 0], ys=point_cloud[:, 1], zs=point_cloud[:, 2], s=5)
    plt.show()




def ChamferDistance(x, y):
    """
    The inputs are sets of d-dimensional points:
    x = {x_1, ..., x_n} and y = {y_1, ..., y_m}.
    Arguments:
        x: a float tensor with shape [b, d, n].
        y: a float tensor with shape [b, d, m].
    Returns:
        a float tensor with shape [].
    """
    x = x.unsqueeze(3)  # shape [b, d, n, 1]
    y = y.unsqueeze(2)  # shape [b, d, 1, m]

    # compute pairwise l2-squared distances
    d = torch.pow(x - y, 2)  # shape [b, d, n, m]
    d = d.sum(1)  # shape [b, n, m]

    min_for_each_x_i, _ = d.min(dim=2)  # shape [b, n]
    min_for_each_y_j, _ = d.min(dim=1)  # shape [b, m]

    distance = min_for_each_x_i.sum(1) + min_for_each_y_j.sum(1)  # shape [b]
    return distance.mean(0)

def setup_seed(seed):
    """
    Set the random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def index_points(point_clouds, index):
    """
    Given a batch of tensor and index, select sub-tensor.

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, N, k]
    Return:
        new_points:, indexed points data, [B, N, k, C]
    """
    device = point_clouds.device
    batch_size = point_clouds.shape[0]
    view_shape = list(index.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(index.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = point_clouds[batch_indices, index, :]
    return new_points


def knn(x, k):
    """
    K nearest neighborhood.

    Parameters
    ----------
        x: a tensor with size of (B, C, N)
        k: the number of nearest neighborhoods
    
    Returns
    -------
        idx: indices of the k nearest neighborhoods with size of (B, N, k)
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # (B, N, N)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # (B, 1, N)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (B, 1, N), (B, N, N), (B, N, 1) -> (B, N, N)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (B, N, k)
    return idx


def to_one_hots(y, categories):
    """
    Encode the labels into one-hot coding.

    :param y: labels for a batch data with size (B,)
    :param categories: total number of kinds for the label in the dataset
    :return: (B, categories)
    """
    y_ = torch.eye(categories)[y.data.cpu().numpy()]
    if y.is_cuda:
        y_ = y_.cuda()
    return y_


def kl_loss(mu, logvar):
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()




def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()





# helperss
def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)


def cast(inputs):
    return inputs.type(torch.float32)



def grid(images):
    return make_grid(images, nrow=len(images))


def distChamfer(x, y):
    x = x.unsqueeze(3)  # shape [b, d, n, 1]
    y = y.unsqueeze(2)  # shape [b, d, 1, m]

    # compute pairwise l2-squared distances
    d = torch.pow(x - y, 2)  # shape [b, d, n, m]
    d = d.sum(1)  # shape [b, n, m]

    min_for_each_x_i, _ = d.min(dim=2)  # shape [b, n]
    min_for_each_y_j, _ = d.min(dim=1)  # shape [b, m]

    distance = min_for_each_x_i.sum(1) + min_for_each_y_j.sum(1)  # shape [b]
    return distance.mean(0)



def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_new_log_dir(root='./logs', postfix='', prefix=''):
    log_dir = os.path.join(root, prefix + time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime()) + postfix)
    os.makedirs(log_dir)
    return log_dir


def int_tuple(argstr):
    return tuple(map(int, argstr.split(',')))


def str_tuple(argstr):
    return tuple(argstr.split(','))


def int_list(argstr):
    return list(map(int, argstr.split(',')))


def str_list(argstr):
    return list(argstr.split(','))


def log_hyperparams(writer, args):
    from torch.utils.tensorboard.summary import hparams
    vars_args = {k:v if isinstance(v, str) else repr(v) for k, v in vars(args).items()}
    exp, ssi, sei = hparams(vars_args, {})
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)


def grid(images):
    return make_grid(images, nrow=len(images))



def get_linear_scheduler(optimizer, start_epoch, end_epoch, start_lr, end_lr):
    def lr_func(epoch):
        if epoch <= start_epoch:
            return 1.0
        elif epoch <= end_epoch:
            total = end_epoch - start_epoch
            delta = epoch - start_epoch
            frac = delta / total
            return (1-frac) * 1.0 + frac * (end_lr / start_lr)
        else:
            return end_lr / start_lr
    return LambdaLR(optimizer, lr_lambda=lr_func)


def point_as_occluder(point0, box=None, x=(0.7, 0.55), y=(0.7, 0.6), z=(0.35, 0.15), flip=False):
    x_max, x_min =x; y_max, y_min =y ; z_max, z_min =z
    point0 = torch.Tensor(point0).clone()
    reshape = len(point0.shape) < 3
    if reshape:
        point0 = point0.unsqueeze(0)
    if flip:
        point0 = point0[:, :,  [0, 2, 1]].clone()
       
    minimum = point0.min(1)[0].unsqueeze(1)
    maximum = point0.max(1)[0].unsqueeze(1)
    if box is None:
        val_min = torch.Tensor(np.array([[x_min, y_min, z_min]])).to(point0)
        val_max = torch.Tensor(np.array([[x_max, y_max, z_max]])).to(point0)
       
    else:
        val_min = box[:3].unsqueeze(0).to(point0)
        val_max = box[3:].unsqueeze(0).to(point0)

    point0 = ((point0 - minimum)/(maximum-minimum))*(val_max-val_min) + val_min

    if reshape:
        return point0[0]

    return point0


class LinearTransformation(object):
    # Implemented in https://github.com/luost26/diffusion-point-cloud/blob/main/utils/transform.py
    r"""Transforms node positions with a square transformation matrix computed
    offline.
    Args:
        matrix (Tensor): tensor with shape :math:`[D, D]` where :math:`D`
            corresponds to the dimensionality of node positions.
    """

    def __init__(self, matrix, attr):
        assert matrix.dim() == 2, (
            'Transformation matrix should be two-dimensional.')
        assert matrix.size(0) == matrix.size(1), (
            'Transformation matrix should be square. Got [{} x {}] rectangular'
            'matrix.'.format(*matrix.size()))

        self.matrix = matrix
        self.attr = attr

    def __call__(self, data):

        if self.attr:
            for key in self.attr:
                pos = data[key].view(-1, 1) if data[key].dim() == 1 else data[key]

                assert pos.size(-1) == self.matrix.size(-2), (
                    'Node position matrix and transformation matrix have incompatible '
                    'shape.')

                data[key] = torch.matmul(pos, self.matrix.to(pos.dtype).to(pos.device))
        else:
            pos = data.view(-1, 1) if data.dim() == 1 else data

            assert pos.size(-1) == self.matrix.size(-2), (
                'Node position matrix and transformation matrix have incompatible '
                'shape.')

            data = torch.matmul(pos, self.matrix.to(pos.dtype).to(pos.device))

        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.matrix.tolist())


class RandomRotate(object):

    # Implemented in https://github.com/luost26/diffusion-point-cloud/blob/main/utils/transform.py
    r"""Rotates node positions around a specific axis by a randomly sampled
    factor within a given interval.
    Args:
        degrees (tuple or float): Rotation interval from which the rotation
            angle is sampled. If :obj:`degrees` is a number instead of a
            tuple, the interval is given by :math:`[-\mathrm{degrees},
            \mathrm{degrees}]`.
        axis (int, optional): The rotation axis. (default: :obj:`0`)
    """

    def __init__(self, degrees, attr, axis=0):
        if isinstance(degrees, numbers.Number):
            degrees = (-abs(degrees), abs(degrees))
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
        self.degrees = degrees
        self.axis = axis
        self.attr = attr

    def __call__(self, data):
        degree = math.pi * random.uniform(*self.degrees) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)

        if self.axis == 0:
            matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
        elif self.axis == 1:
            matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
        else:
            matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
        return LinearTransformation(torch.tensor(matrix), attr=self.attr)(data)

    def __repr__(self):
        return '{}({}, axis={})'.format(self.__class__.__name__, self.degrees,
                                        self.axis)

def awgn(s, SNR_min=100, SNR_max=None, L=1, return_snr=False):
    shape = s.shape
    """
    AWGN channel
    Add AWGN noise to input signal. The function adds AWGN noise vector to signal 's' to generate a resulting signal vector 'r' of specified SNR in dB. It also
    returns the noise vector 'n' that is added to the signal 's' and the power spectral density N0 of noise added
    Parameters:
        s : input/transmitted signal vector
        SNRdB : desired signal to noise ratio (expressed in dB) for the received signal
        L : oversampling factor (applicable for waveform simulation) default L = 1.
    Returns:
        r : received signal vector (r=s+n)
    """
    s = torch.Tensor(s)
    device = s.device

    assert len(shape) == 4, f"Expect 4 Dim, Got {shape} Dim to use"

    if SNR_max:
        SNRdB = torch.randint(low=SNR_min, high=SNR_max, size=(shape[0],)).to(device)
    else:
        if isinstance(SNR_min, int):
            return_snr = False

            SNRdB = torch.ones((shape[0],)).to(device)*SNR_min
        else:
            SNRdB = torch.tensor(SNR_min).to(device)

    s = torch.reshape(s, [s.shape[0], -1])
    gamma = 10**(SNRdB/10)


    P = L *torch.sum(torch.abs(s)**2, dim=1)/s.shape[-1]
    N0 = P/gamma
    n = torch.sqrt(N0/2).unsqueeze(1)*torch.rand(s.shape).to(device)
    s = s+n
    if return_snr:
        return torch.reshape(s, shape), SNRdB
    else:
        return torch.reshape(s, shape)


def normalize(s):
    shape = s.shape
    s = torch.reshape(s, (shape[0], -1))
    s = torch.divide(s, s.max(1)[0].unsqueeze(1))
    return torch.reshape(s, shape)



# importing required libraries
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.animation as animation
from IPython import display


def gif_anim(fig, axes=None, filename=None):

    def rotate(angle):
        if axes is None:
            for axs in fig.axes:
                axs.view_init(azim=angle)

        else:
            for axs in axes:
                axs.view_init(azim=angle)


    print("Making animation")
    rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 362, 2), interval=100)
    # converting to an html5 video
    video = rot_animation.to_html5_video()

    # embedding for the video
    html = display.HTML(video)

    # draw the animation
    display.display(html)

    if filename is not None:
        writervideo = animation.FFMpegWriter(fps=20)
        rot_animation.save(filename, writer=writervideo)
    plt.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def is_main_process():
    return get_rank() == 0

def format_step(step):
    if isinstance(step, str):
        return step
    s = ""
    if len(step) > 0:
        s += "Training Epoch: {} ".format(step[0])
    if len(step) > 1:
        s += "Training Iteration: {} ".format(step[1])
    if len(step) > 2:
        s += "Validation Iteration: {} ".format(step[2])
    return s



class ConstantLRSchedule(LambdaLR):
    """ Constant learning rate schedule.
    """
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLRSchedule, self).__init__(optimizer, lambda _: 1.0, last_epoch=last_epoch)


class WarmupConstantSchedule(LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupConstantSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return 1.


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


def calc_iou( gt_bbox, pred_bbox):
    '''
    This function takes the predicted bounding box and ground truth bounding box and 
    return the IoU ratio
    '''
    x_min, y_min, z_min, x_max, y_max, z_max = gt_bbox
    x_min_p, y_min_p, z_min_p, x_max_p, y_max_p, z_max_p = pred_bbox
    
    if (x_min > x_max) or (y_min > y_max) or (z_min > z_max):
        raise AssertionError("Ground Truth Bounding Box is not correct")
    if (x_min_p > x_max_p) or (y_min_p> y_max_p) or (z_min_p > z_max_p):
        raise AssertionError("Predicted Bounding Box is not correct", x_min_p, x_max_p, y_min_p, y_max_p, z_min_p, z_max_p)
        
         
    #if the GT bbox and predcited BBox do not overlap then iou=0
    if(x_max < x_min_p) or (y_max < y_min_p) or (z_max < z_min_p):
        # If bottom right of x-coordinate  GT  bbox is less than or above the top left of x coordinate of  the predicted BBox
        
        return 0.0
    

    # if(x_topleft_gt> x_bottomright_p): # If bottom right of x-coordinate  GT  bbox is greater than or below the bottom right  of x coordinate of  the predcited BBox
        
    #     return 0.0
    # if(y_topleft_gt> y_bottomright_p): # If bottom right of y-coordinate  GT  bbox is greater than or below the bottom right  of y coordinate of  the predcited BBox
        
    #     return 0.0
    
    
    GT_bbox_area = (x_max -  x_min + 1) * (  y_max -y_min + 1) * (  z_max - z_min + 1)
    Pred_bbox_area = (x_max_p -  x_min_p + 1) * (  y_max_p - y_min_p + 1) * (  z_max_p - z_min_p + 1)
    
    x_min_area = np.max([x_min, x_min_p])
    y_min_area = np.max([y_min, y_min_p])
    z_min_area = np.max([z_min, z_min_p])

    x_max_area = np.min([x_max, x_max_p])
    y_max_area = np.min([y_max, y_max_p])
    z_max_area = np.min([z_max, z_max_p])
    
    intersection_area = (x_max_area -  x_min_area + 1) * (  y_max_area - y_min_area + 1) * (  z_max_area - z_min_area + 1)
    
    union_area = (GT_bbox_area + Pred_bbox_area - intersection_area)
   
    return intersection_area/union_area


def get_verts(box):
    if len(box.shape)==1:
        box = box.tolist()
        x_min, y_min, z_min = box[:3]
        x_max, y_max, z_max = box[3:]

        return [[x_min, y_min, z_min],
                  [x_min, y_max, z_min], 
                  [x_min, y_max, z_max],
                  [x_min, y_min, z_max],
                  [x_max, y_min, z_min],
                  [x_max, y_max, z_min],
                  [x_max, y_min, z_max],
                  [x_max, y_max, z_max]]


def cast(inputs):
    return inputs.type(torch.float32)

def preset(prefix):
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    print(f'{prefix}: {os.environ.get("MKL_THREADING_LAYER")}')

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def set_seed(args):
    seed_everything(args.seed)



if __name__ == '__main__':
    pcs = torch.rand(32, 3, 1024)
    knn_index = knn(pcs, 16)
    print(knn_index.size())
    knn_pcs = index_points(pcs.permute(0, 2, 1), knn_index)
    print(knn_pcs.size())

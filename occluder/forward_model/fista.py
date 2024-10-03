import numpy as np
from random import randint
import torch
import torch.linalg as LA
import functools


summ = lambda x: torch.sum(torch.sum(x, dim=-1), dim=-1)



l1_prox = lambda x, weight: torch.sign(x)*torch.max(torch.abs(x) - weight, dim=1)[0][..., None]
l2_prox = lambda x, weight: (1.0 / (weight + 1)) * x
gx = lambda x: TV_norm(x, opt="")
fx = lambda A, b, x, mode: norm2sq(b - A.GetMeasurement(x, mode))

resetshape = lambda x, N1, N2: x.permute(0, 3, 1, 2).reshape(-1, N1, N2, 1)
reverseshape = lambda x, N1, N2, C: x.reshape(-1, C, N1, N2).permute(0, 2, 3, 1)


def norm2sq(x):
    return (1.0 / 2) * torch.linalg.matrix_norm(x, dim=(1, 2)) ** 2

def psnr(ref, recon):
    """
        psnr takes as input the reference (ref) and the estimated (recon) images.
    """
    mse = torch.mean((ref - recon) ** 2)
    return 20 * torch.log10(255.0 / torch.sqrt(mse))

def slice_at_axis(sl, axis):
    """
    Construct tuple of slices to slice an array in the given dimension.

    Parameters
    ----------
    sl : slice
        The slice for the given dimension.
    axis : int
        The axis to which `sl` is applied. All other dimensions are left
        "unsliced".

    Returns
    -------
    sl : tuple of slices
        A tuple with slices matching `shape` in length.

    Examples
    --------
    >>> slice_at_axis(slice(None, 3, -1), 1)
    (slice(None, None, None), slice(None, 3, -1), Ellipsis)
    """
    return (slice(None),) * axis + (sl,) + (...,)

def TV_norm(X, opt=None):
    """
        Computes the TV-norm of image X
        opts = 'iso' for isotropic, else it is the anisotropic TV-norm
    """

    

    b, m, n = X.shape
    P1 = X[:, 0:m - 1, :] - X[:, 1:m, :]
    P2 = X[:, :, 0:n - 1] - X[:, :, 1:n]

    if opt == 'iso':
        D = torch.zeros_like(X)
        D[:, 0:m - 1, :] = P1 ** 2
        D[:, :, 0:n - 1] = D[:, :, 0:n - 1] + P2 ** 2
        tv_out = summ(torch.sqrt(D))
        # tv_out = np.sum(np.sqrt(D))
    else:
        tv_out = summ(torch.abs(P1)) + summ(torch.abs(P2))

    return tv_out




def FISTA(x_initial, gradf, proxg, alpha, num_iteration=5):



    x_k = x_initial
    y_k = x_initial
    t_k = torch.ones((x_initial.shape[0], 1, 1, 1), device=x_initial.device)

    for k in range(num_iteration):
        # Update iterate
        grad = gradf(y_k)
        # print(grad.shape)
        # print(y_k.shape)
     
        prox_argument = y_k - alpha * grad


        
    
        
        x_k_next = proxg(prox_argument, alpha)
      
        t_k_next = (1 + (4 * (t_k ** 2) + 1)**0.5 ) / 2
        y_k_next = x_k_next + ((t_k - 1) / t_k_next) * (x_k_next - x_k)
        y_k = y_k_next
        t_k = t_k_next
        x_k = x_k_next
        # # print(x_k_next)

        # print(torch.isnan(x_k).any(), "Here")


    return x_k






def reconstructTV(A, b, x_initial=None, mode=None, num_iteration=30, lamda=1, prox_Lips=None, proximal_iter=30, eps=1e-5):
    """
        image: undersampled image (mxm) to be reconstructed
        indices: indices of the undersampled locations
        optimizer: method of reconstruction (FISTA/ISTA function handle)
        params:
    """
    B, M1, M2, C = b.shape
    N1, N2, TB = A.hidden_resolution, A.hidden_resolution, C*B
    b = resetshape(b, M1, M2)

    if b.device != A.device:
        b = torch.Tensor(b).to(A.device)

    device = A.device
    alpha = 1/(torch.linalg.norm(A.Model)**2) if prox_Lips is None else 1/prox_Lips
    
    x = x_initial if x_initial is not None else torch.zeros((TB, N1, N2, 1), device=device)


    proxg = lambda x, alpha: denoise_tv_chambolle(x.reshape((TB, -1, 1)),
                                    weight=lamda * alpha, eps=eps,
                                    max_num_iter=proximal_iter, channel_axis=0).reshape((TB, N1, N2, 1))

    # proxg= l1_prox(x.reshape((TB, -1)), weight=lamda * alpha).reshape((TB, N1, N2, 1)) #

    gradf = lambda x: A.GetTranspose(A.get_measurement(x, Model=mode) - b, Model=mode)

    x = FISTA(x, gradf, proxg, alpha, num_iteration, )
    return reverseshape(x, N1, N2, C)



def reconstructL1(A, b, x_initial=None, mode=None, num_iteration=30, lamda=1, prox_Lips=None):
    """
        image: undersampled image (mxm) to be reconstructed
        indices: indices of the undersampled locations
        optimizer: method of reconstruction (FISTA/ISTA function handle)
        params:
    """
    B, M1, M2, C = b.shape
    N1, N2, TB = A.N2, A.N1, C*B
    b = resetshape(b, M1, M2)

    if b.device != A.device:
        b = torch.Tensor(b).to(A.device)

    device = A.device
    alpha = 1/(torch.linalg.norm(A.Model)**2) if prox_Lips is None else 1/prox_Lips
    
    x = x_initial if x_initial is not None else torch.zeros((TB, N1, N2, 1), device=device)


    proxg = lambda x, alpha: l1_prox(x.reshape((TB, -1)), weight=lamda * alpha).reshape((TB, N1, N2, 1)) #

    gradf = lambda x: A.GetTranspose(A.GetMeasurement(x, mode) - b, mode)

    x = FISTA(x, gradf, proxg, alpha, num_iteration, )
    return reverseshape(x, N2, N1, C)



def denoise_tv_chambolle(image, weight=0.1, eps=2.e-4, max_num_iter=200,
                         multichannel=False, *, channel_axis=None):

    if channel_axis is not None:
        channel_axis = channel_axis % image.ndim
        _at = functools.partial(slice_at_axis, axis=channel_axis)
        out = torch.zeros_like(image, device=image.device)
        for c in range(image.shape[channel_axis]):
            out[_at(c)] = denoise_tv_chambolle_nd(image[_at(c)], weight, eps,
                                                   max_num_iter)
    else:
        out = denoise_tv_chambolle_nd(image, weight, eps, max_num_iter)
    return out


def denoise_tv_chambolle_nd(image, weight=0.1, eps=2.e-4, max_num_iter=200, channel_axis=None):
    """Perform total-variation denoising on n-dimensional images.

    Parameters
    ----------
    image : ndarray
        n-D input data to be denoised.
    weight : float, optional
        Denoising weight. The greater `weight`, the more denoising (at
        the expense of fidelity to `input`).
    eps : float, optional
        Relative difference of the value of the cost function that determines
        the stop criterion. The algorithm stops when:

            (E_(n-1) - E_n) < eps * E_0

    max_num_iter : int, optional
        Maximal number of iterations used for the optimization.

    Returns
    -------
    out : ndarray
        Denoised array of floats.

    Notes
    -----
    Rudin, Osher and Fatemi algorithm.
    """

    ndim = image.ndim
    p = torch.zeros((image.ndim, ) + image.shape, device=image.device)
    g = torch.zeros_like(p, device=image.device)
    d = torch.zeros_like(image, device=image.device)
    i = 0
    while i < max_num_iter:
        
        if i > 0:
            # d will be the (negative) divergence of p
            d = -p.sum(0)
            slices_d = [slice(None), ] * ndim
            slices_p = [slice(None), ] * (ndim + 1)
            for ax in range(ndim):
                slices_d[ax] = slice(1, None)
                slices_p[ax+1] = slice(0, -1)
                slices_p[0] = ax
                d[tuple(slices_d)] += p[tuple(slices_p)]
                slices_d[ax] = slice(None)
                slices_p[ax+1] = slice(None)
            out = image + d
        else:
            out = image
        E = (d ** 2).sum()

        # print(torch.isnan(out).any(), "Here")

        # g stores the gradients of out along each axis
        # e.g. g[0] is the first order finite difference along axis 0
        slices_g = [slice(None), ] * (ndim + 1)
        for ax in range(ndim):
            slices_g[ax+1] = slice(0, -1)
            slices_g[0] = ax
            g[tuple(slices_g)] = torch.diff(out, axis=ax)
            slices_g[ax+1] = slice(None)

        norm = torch.sqrt((g ** 2).sum(axis=0))[None, ...]
        E += weight * norm.sum()
        tau = 1. / (2.*ndim)
        norm *= tau / weight
        norm += 1.
        p -= tau * g
        p /= norm
        E /= image.shape[0]
        if i == 0:
            E_init = E
            E_previous = E
        else:
            if torch.abs(E_previous - E) < eps * E_init:
                break
            else:
                E_previous = E
        i += 1
    return out




def gradient_scheme_restart_condition(x_k, x_k_next, y_k):
    """
    Whether to restart
    """
    return (y_k - x_k_next) @ (x_k_next - x_k) > 0






import torch
import numpy as np

def _number_of_params(p):
    '''
    computes the dimension of a parameter (vectorized for matrices)

    Args:
        p: model parameter from which we want to know the dimension

    Out:
        integer, which gives the dimension of the parameter'''
    res = 1
    for s in p.shape:
        res *= s
    return res


def is_freezed(p, freezed):
    '''
    checks if the parameter p is in the list of frozen parameters 'freezed'
    Args:
        p: a model parameter (p in model.parameters())
        freezed (list): list of the frozen parameters of the model
    Out:
         boolean
    '''
    for freezed_p in freezed:
        # print(p.shape, freezed_p.shape)
        try:
            if p is freezed_p:
                return True

        except RuntimeError as m:
            print(m)
            continue
    return False

def prod(iterable):
    '''
    computes the product of all elements in an iterable
    Args:
        iterable: an iterable
    Out:
        product of all elements in the iterable
    '''
    res = 1
    for i in iterable:
        res *= i
    return res

def inverse_by_cholesky(tensor, damping):
    damped = tensor + torch.eye(tensor.shape[-1]) * damping
    cholesky = torch.linalg.cholesky(damped)
    return torch.cholesky_inverse(cholesky)

def grads_to_mat(grads):
    mat = grads.reshape([grads.shape[0], grads.shape[1]*grads.shape[2]*grads.shape[3]])
    return mat

def mat_to_grads(mat):
    grad = mat.reshape([mat.shape[0],int(mat.shape[1]/9), 3,3])
    return grad

def precond(kfac_tuple, grad_p):
    prec_grad = mat_to_grads(kfac_tuple[0] @ grads_to_mat(grad_p) @ kfac_tuple[1])
    return prec_grad

def SingularValues(kernel, input_shape):
    transforms = np.fft.fft2(kernel, input_shape, axes=[0, 1])
    s = np.linalg.svd(transforms, compute_uv=False)
    return s

def max_sing_value(kernel, input_shape):
    s = SingularValues(kernel, input_shape)
    sv = np.ndarray.flatten(s)
    return max(sv)


import torch

from LogSinkhornGPUBackend import BalanceCUDA_32, BalanceCUDA_64, \
    BasicToCompositeCUDA_2D_32, BasicToCompositeCUDA_2D_64, \
    AddWithOffsetsCUDA_2D_32, AddWithOffsetsCUDA_2D_64

def log_dens(a):
    """
    Log of `a`, thresholded at the negative infinities. Taken from `geomloss`.
    """
    a_log = a.log()
    a_log[a <= 0] = -10000.0
    return a_log


def batch_dim(a):
    """
    Batch dimension of tensor.
    """
    return a.shape[0]


def geom_dims(a):
    """
    Dimensions folowing the batch dimension.
    """
    return a.shape[1:]

# DomDec functions


def BalanceCUDA(nu_basic, mass_delta, thresh_step):
    if nu_basic.dtype == torch.float32:
        f = BalanceCUDA_32
    elif nu_basic.dtype == torch.float64:
        f = BalanceCUDA_64
    else: 
        raise NotImplementedError(
            "BalanceCUDA only implemented for float and double"
        )
    return f(nu_basic, mass_delta, thresh_step)
    
def BasicToCompositeCUDA_2D(
    nu_basic, w, h, 
    left_in_composite, left_in_basic, width_basic,
    bottom_in_composite, bottom_in_basic, height_basic
):
    if nu_basic.dtype == torch.float32:
        f = BasicToCompositeCUDA_2D_32
    elif nu_basic.dtype == torch.float64:
        f = BasicToCompositeCUDA_2D_64
    else: 
        raise NotImplementedError(
            "BalanceCUDA only implemented for float and double"
        )
    return f(nu_basic, w, h, 
        left_in_composite, left_in_basic, width_basic,
        bottom_in_composite, bottom_in_basic, height_basic)


def AddWithOffsetsCUDA_2D(
    nu_basic, w, h, 
    weights, sum_indices,
    left_in_composite, left_in_basic, width_basic,
    bottom_in_composite, bottom_in_basic, height_basic
):
    if nu_basic.dtype == torch.float32:
        f = AddWithOffsetsCUDA_2D_32
    elif nu_basic.dtype == torch.float64:
        f = AddWithOffsetsCUDA_2D_64
    else: 
        raise NotImplementedError(
            "BalanceCUDA only implemented for float and double"
        )
    return f(nu_basic, w, h, 
        weights, sum_indices,
        left_in_composite, left_in_basic, width_basic,
        bottom_in_composite, bottom_in_basic, height_basic)
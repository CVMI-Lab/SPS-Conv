import torch
import spconv.pytorch as spconv

def calculate_gemm_flops(x, batch_dict, indice_key, inchannel, outchannel):
    pair_fwd = x.indice_dict[indice_key].pair_fwd
    cur_flops = 2 * (pair_fwd > -1).sum() * inchannel * outchannel - pair_fwd.shape[1]
    return cur_flops
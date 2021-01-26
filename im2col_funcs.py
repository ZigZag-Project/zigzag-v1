from copy import deepcopy
from numpy import prod
from math import ceil


def im2col_layer_transform(layer_info):
    im2col_layer_info = {}
    for layer_index, layer in layer_info.items():
        # TODO support stride under im2col mode
        im2col_layer_info[layer_index] = {'B': 1, 'K': 1, 'C': 1, 'OY': 1, 'OX': 1, 'FY': 1, 'FX': 1, 'SY': 1, 'SX': 1,
                                          'SFY': 1, 'SFX': 1, 'PY': 0, 'PX': 0}
        im2col_layer_info[layer_index]['B'] = layer['B'] * layer['OY'] * layer['OX']
        im2col_layer_info[layer_index]['K'] = layer['K']
        im2col_layer_info[layer_index]['C'] = layer['C'] * layer['FY'] * layer['FX']

    return im2col_layer_info


def im2col_mem_access_correction(layer_origin, layer_im2col, mem_total_access, temporal_loop, spatial_loop,
                                 im2col_top_mem_level):
    # TODO This is just a temporary solution for a constraint architecture (FB-similar),
    #  in which only the top memory level for I may not do im2col, thus may need to be corrected) & stride = 1.
    C_pre_unrolled = 2

    I_mem_level = len(temporal_loop.B['I'])

    B_tot = deepcopy(spatial_loop.Bu['I'])
    K_tot = deepcopy(spatial_loop.Ku['I'])
    C_tot = deepcopy(spatial_loop.Cu['I'])

    for level in range(I_mem_level):
        B_tot[level + 1] *= temporal_loop.B['I'][level]
        K_tot[level + 1] *= temporal_loop.K['I'][level]
        C_tot[level + 1] *= temporal_loop.C['I'][level]

    B_below = prod(B_tot[0:im2col_top_mem_level + 1]).item()
    K_below = prod(K_tot[0:im2col_top_mem_level + 1]).item()
    C_below = prod(C_tot[0:im2col_top_mem_level + 1]).item()

    B_L, OY_L, OX_L = B_col2im_decouple(B_below, layer_origin)
    K_L = K_below
    C_below /= C_pre_unrolled
    C_L, FY_L, FX_L = C_col2im_decouple(C_below, layer_origin)
    C_L *= C_pre_unrolled

    B_H = layer_origin.B / B_L
    K_H = layer_origin.K / K_L
    C_H = layer_origin.C / C_L
    OY_H = layer_origin.OY / OY_L
    OX_H = layer_origin.OX / OX_L
    FY_H = layer_origin.FY / FY_L
    FX_H = layer_origin.FX / FX_L

    cycle_L = B_L * K_L * C_L * OY_L * OX_L * FY_L * FX_L
    I_data_size_L = B_L* C_L * (OY_L + FY_L - 1) * (OX_L + FX_L - 1)
    I_data_reuse_L = cycle_L / I_data_size_L
    I_data_reuse_tot = layer_origin.total_data_reuse['I']
    I_data_reuse_H = I_data_reuse_tot/I_data_reuse_L

    a=1


def B_col2im_decouple(B_below, layer_origin):
    B = 1
    OY = 1
    OX = 1
    B_below_origin = B_below
    if B_below == 1:
        return B, OY, OX
    else:
        if B_below <= layer_origin.OX:
            OX = B_below
            return B, OY, OX
        else:
            OX = layer_origin.OX
            B_below /= OX
            if B_below <= layer_origin.OY:
                OY = ceil(B_below)
                OX = B_below_origin / OY
                return B, OY, OX
            else:
                OY = layer_origin.OY
                B_below /= OY
                if B_below <= layer_origin.B:
                    B = ceil(B_below)
                    OX = B_below_origin / B / OY
                    return B, OY, OX


def C_col2im_decouple(C_below, layer_origin):
    C = 1
    FY = 1
    FX = 1
    C_below_origin = C_below
    if C_below == 1:
        return C, FY, FX
    else:
        if C_below <= layer_origin.FX:
            FX = C_below
            return C, FY, FX
        else:
            FX = layer_origin.FX
            C_below /= FX
            if C_below <= layer_origin.FY:
                FY = ceil(C_below)
                FX = C_below_origin / FY
                return C, FY, FX
            else:
                FY = layer_origin.FY
                C_below /= FY
                if C_below <= layer_origin.C:
                    C = ceil(C_below)
                    FX = C_below_origin / C / FY
                    return C, FY, FX


def su_col2im(mem_scheme, layer_7D_origin, layer_3D_origin, layer_3D_rounded):
    """
    This function updates col2im parameters in mem_scheme, namely,
    col2im_flooring, col2im_fraction_spatial_unrolling, col2im_spatial_unrolling.
    These parameters will later be used to calculate accurate Input access count for
    those Input memory levels above the im2col_top_mem_level (defined in setting file),
    which can get benefit from Input FIFO effect.
    """

    ideal_su = mem_scheme.spatial_unrolling
    fraction_su = mem_scheme.fraction_spatial_unrolling
    flooring = mem_scheme.flooring

    col2im_ideal_su = {'W': [], 'I': [], 'O': []}
    col2im_fraction_su = {'W': [], 'I': [], 'O': []}
    col2im_flooring = {'W': [], 'I': [], 'O': []}

    for ii_su in range(len(ideal_su)):
        for op in ['W', 'I', 'O']:
            for su_per_level in ideal_su[ii_su][op]:
                col2im_ideal_su[op].append([])
                if su_per_level:
                    for su_single in su_per_level:
                        su_type = su_single[0]
                        if su_type == 6:
                            col2im_ideal_su[op][-1].append(su_single)
                        else:
                            su_single_update = su_single_decouple(su_single, layer_7D_origin)
                            col2im_ideal_su[op][-1].append(su_single_update)

    a = 1

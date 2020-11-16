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


# def su_col2im(mem_scheme, layer_7D_origin, layer_3D_origin, layer_3D_rounded):
#     """
#     This function updates col2im parameters in mem_scheme, namely,
#     col2im_flooring, col2im_fraction_spatial_unrolling, col2im_spatial_unrolling.
#     These parameters will later be used to calculate accurate Input access count for
#     those Input memory levels above the im2col_top_mem_level (defined in setting file),
#     which can get benefit from Input FIFO effect.
#     """
#
#     ideal_su = mem_scheme.spatial_unrolling
#     fraction_su = mem_scheme.fraction_spatial_unrolling
#     flooring = mem_scheme.flooring
#
#     col2im_ideal_su = {'W': [], 'I': [], 'O': []}
#     col2im_fraction_su = {'W': [], 'I': [], 'O': []}
#     col2im_flooring = {'W': [], 'I': [], 'O': []}
#
#     for ii_su in range(len(ideal_su)):
#         for op in ['W', 'I', 'O']:
#             for su_per_level in ideal_su[ii_su][op]:
#                 col2im_ideal_su[op].append([])
#                 if su_per_level:
#                     for su_single in su_per_level:
#                         su_type = su_single[0]
#                         if su_type == 6:
#                             col2im_ideal_su[op][-1].append(su_single)
#                         else:
#                             su_single_update = su_single_decouple(su_single, layer_7D_origin)
#                             col2im_ideal_su[op][-1].append(su_single_update)
#
#     a = 1


def pw_layer_col2im(spatial_scheme, flooring, temporal_scheme, original_layer):
    """
    This function change a pointwise layer, which has been auto-transferred (im2col), back to its original shape.
    Recover 3D (B, K, C) back to 5D (B, K, C, OY, OX) in spatial_scheme, flooring, temporal_scheme.
    """

    OX = {'W': original_layer[3], 'I': original_layer[3], 'O': original_layer[3]}
    OY = {'W': original_layer[4], 'I': original_layer[4], 'O': original_layer[4]}
    B = {'W': original_layer[7], 'I': original_layer[7], 'O': original_layer[7]}

    # su_transfer_count is used to convert flooring, 7 -> 3 or 3,4 or 3,4,7
    su_transfer_op = {'W': [], 'I': [], 'O': []}

    spatial_scheme_saved = deepcopy(spatial_scheme)
    for op in ['W', 'I', 'O']:
        for level, su_list in enumerate(spatial_scheme_saved[op]):
            su_transfer_op[op].append([])
            if su_list:
                for idx, su_single in enumerate(su_list):
                    if su_single[0] == 7:
                        su_transfer_op[op][-1].append([])
                        if su_single[1] <= OX[op]:
                            find_7_item = next((x for x in spatial_scheme[op][level] if x[0] == 7), None)
                            find_7_idx = spatial_scheme[op][level].index(find_7_item)
                            OX_position_value = spatial_scheme_saved[op][level][idx][1]
                            spatial_scheme[op][level].insert(find_7_idx, [3, OX_position_value])
                            su_transfer_op[op][-1][-1] = [3]  # B -> OX
                            try:
                                spatial_scheme[op][level].remove((7, su_single[1]))
                            except:
                                spatial_scheme[op][level].remove([7, su_single[1]])
                            OX[op] = round(OX[op] / OX_position_value)

                        elif OX[op] < su_single[1] < OX[op] * OY[op]:
                            if OX[op] > 1:
                                find_7_item = next((x for x in spatial_scheme[op][level] if x[0] == 7), None)
                                find_7_idx = spatial_scheme[op][level].index(find_7_item)
                                spatial_scheme[op][level].insert(find_7_idx, [3, OX[op]])
                                OY_posision_value = round(spatial_scheme_saved[op][level][idx][1] / OX[op])
                                spatial_scheme[op][level].insert(find_7_idx + 1, [4, OY_posision_value])
                                su_transfer_op[op][-1][-1] = [3, 4]  # B -> OX, OY
                                try:
                                    spatial_scheme[op][level].remove((7, su_single[1]))
                                except:
                                    spatial_scheme[op][level].remove([7, su_single[1]])

                            else:
                                find_7_item = next((x for x in spatial_scheme[op][level] if x[0] == 7), None)
                                find_7_idx = spatial_scheme[op][level].index(find_7_item)
                                OY_posision_value = spatial_scheme_saved[op][level][idx][1]
                                spatial_scheme[op][level].insert(find_7_idx, [4, OY_posision_value])
                                su_transfer_op[op][-1][-1] = [4]  # B -> OY
                                try:
                                    spatial_scheme[op][level].remove((7, su_single[1]))
                                except:
                                    spatial_scheme[op][level].remove([7, su_single[1]])
                            OX[op] = 1
                            OY[op] = round(OY[op] / OY_posision_value)

                        elif su_single[1] == OX[op] * OY[op]:
                            if OX[op] > 1:
                                find_7_item = next((x for x in spatial_scheme[op][level] if x[0] == 7), None)
                                find_7_idx = spatial_scheme[op][level].index(find_7_item)
                                spatial_scheme[op][level].insert(find_7_idx, [3, OX[op]])
                                spatial_scheme[op][level].insert(find_7_idx + 1, [4, OY[op]])
                                su_transfer_op[op][-1][-1] = [3, 4]  # B -> OX, OY
                                try:
                                    spatial_scheme[op][level].remove((7, su_single[1]))
                                except:
                                    spatial_scheme[op][level].remove([7, su_single[1]])

                            else:
                                find_7_item = next((x for x in spatial_scheme[op][level] if x[0] == 7), None)
                                find_7_idx = spatial_scheme[op][level].index(find_7_item)
                                spatial_scheme[op][level].insert(find_7_idx, [4, OY[op]])
                                su_transfer_op[op][-1][-1] = [4]  # B -> OY
                                try:
                                    spatial_scheme[op][level].remove((7, su_single[1]))
                                except:
                                    spatial_scheme[op][level].remove([7, su_single[1]])
                            OX[op] = 1
                            OY[op] = 1

                        elif su_single[1] > OX[op] * OY[op]:
                            if OX[op] > 1 and OY[op] > 1:
                                find_7_item = next((x for x in spatial_scheme[op][level] if x[0] == 7), None)
                                find_7_idx = spatial_scheme[op][level].index(find_7_item)
                                spatial_scheme[op][level].insert(find_7_idx, [3, OX[op]])
                                spatial_scheme[op][level].insert(find_7_idx + 1, [4, OY[op]])
                                B_posision_value = round(spatial_scheme_saved[op][level][idx][1] / OX[op] / OY[op])
                                spatial_scheme[op][level].insert(find_7_idx + 2, [7, B_posision_value])
                                su_transfer_op[op][-1][-1] = [3, 4, 7]  # B -> OX, OY, B
                                try:
                                    spatial_scheme[op][level].remove((7, su_single[1]))
                                except:
                                    spatial_scheme[op][level].remove([7, su_single[1]])

                            elif OX[op] == 1 and OY[op] > 1:
                                find_7_item = next((x for x in spatial_scheme[op][level] if x[0] == 7), None)
                                find_7_idx = spatial_scheme[op][level].index(find_7_item)
                                spatial_scheme[op][level].insert(find_7_idx, [4, OY[op]])
                                B_posision_value = round(spatial_scheme_saved[op][level][idx][1] / OY[op])
                                spatial_scheme[op][level].insert(find_7_idx + 1, [7, B_posision_value])
                                su_transfer_op[op][-1][-1] = [4, 7]  # B -> OY, B
                                try:
                                    spatial_scheme[op][level].remove((7, su_single[1]))
                                except:
                                    spatial_scheme[op][level].remove([7, su_single[1]])

                            elif OX[op] == 1 and OY[op] == 1:
                                B_posision_value = spatial_scheme_saved[op][level][idx][1]
                                su_transfer_op[op][-1][-1] = [7]  # B -> B
                            else:
                                raise ValueError('ERROR 1 (su)')
                            OX[op] = 1
                            OY[op] = 1
                            B[op] = round(B[op] / B_posision_value)

                        else:
                            raise ValueError('ERROR 2 (su)')

    if B['W'] != B['I'] != B['O'] or OY['W'] != OY['I'] != OY['O'] or OX['W'] != OX['I'] != OX['O']:
        raise ValueError('ERROR 3')

    flooring_saved = deepcopy(flooring)
    for op in ['W', 'I', 'O']:
        for level, floor_list in enumerate(flooring_saved[op]):
            i = 0
            for XY, floor_XY in enumerate(floor_list):
                for floor_single in floor_XY:
                    if floor_single == 7:
                        find_7_idx = flooring[op][level][XY].index(7)
                        for x in reversed(su_transfer_op[op][level][i]):
                            flooring[op][level][XY].insert(find_7_idx, x)
                        i += 1
                        flooring[op][level][XY].remove(7)

    temporal_scheme_saved = deepcopy(temporal_scheme)
    for op in ['W', 'I', 'O']:
        for level, loop_list in enumerate(temporal_scheme_saved[op]):
            # su_transfer_op[op].append([])
            if loop_list:
                for idx, loop_single in enumerate(loop_list):
                    if loop_single[0] == 7:
                        # su_transfer_op[op][-1].append([])
                        if loop_single[1] <= OX[op]:
                            find_7_item = next((x for x in temporal_scheme[op][level] if x[0] == 7), None)
                            find_7_idx = temporal_scheme[op][level].index(find_7_item)
                            OX_position_value = temporal_scheme_saved[op][level][idx][1]
                            temporal_scheme[op][level].insert(find_7_idx, (3, OX_position_value))
                            # su_transfer_op[op][-1][-1] = [3]  # B -> OX
                            temporal_scheme[op][level].remove((7, loop_single[1]))
                            OX[op] = round(OX[op] / OX_position_value)

                        elif OX[op] < loop_single[1] < OX[op] * OY[op]:
                            if OX[op] > 1:
                                find_7_item = next((x for x in temporal_scheme[op][level] if x[0] == 7), None)
                                find_7_idx = temporal_scheme[op][level].index(find_7_item)
                                temporal_scheme[op][level].insert(find_7_idx, (3, OX[op]))
                                OY_posision_value = round(temporal_scheme_saved[op][level][idx][1] / OX[op])
                                temporal_scheme[op][level].insert(find_7_idx + 1, (4, OY_posision_value))
                                # su_transfer_op[op][-1][-1] = [3, 4]  # B -> OX, OY
                                temporal_scheme[op][level].remove((7, loop_single[1]))
                            else:
                                find_7_item = next((x for x in temporal_scheme[op][level] if x[0] == 7), None)
                                find_7_idx = temporal_scheme[op][level].index(find_7_item)
                                OY_posision_value = temporal_scheme_saved[op][level][idx][1]
                                temporal_scheme[op][level].insert(find_7_idx, (4, OY_posision_value))
                                # su_transfer_op[op][-1][-1] = [4]  # B -> OY
                                temporal_scheme[op][level].remove((7, loop_single[1]))
                            OX[op] = 1
                            OY[op] = round(OY[op] / OY_posision_value)

                        elif loop_single[1] == OX[op] * OY[op]:
                            if OX[op] > 1:
                                find_7_item = next((x for x in temporal_scheme[op][level] if x[0] == 7), None)
                                find_7_idx = temporal_scheme[op][level].index(find_7_item)
                                temporal_scheme[op][level].insert(find_7_idx, (3, OX[op]))
                                temporal_scheme[op][level].insert(find_7_idx + 1, (4, OY[op]))
                                # su_transfer_op[op][-1][-1] = [3, 4]  # B -> OX, OY
                                temporal_scheme[op][level].remove((7, loop_single[1]))
                            else:
                                find_7_item = next((x for x in temporal_scheme[op][level] if x[0] == 7), None)
                                find_7_idx = temporal_scheme[op][level].index(find_7_item)
                                temporal_scheme[op][level].insert(find_7_idx, (4, OY[op]))
                                # su_transfer_op[op][-1][-1] = [4]  # B -> OY
                                temporal_scheme[op][level].remove((7, loop_single[1]))
                            OX[op] = 1
                            OY[op] = 1

                        elif loop_single[1] > OX[op] * OY[op]:
                            if OX[op] > 1 and OY[op] > 1:
                                find_7_item = next((x for x in temporal_scheme[op][level] if x[0] == 7), None)
                                find_7_idx = temporal_scheme[op][level].index(find_7_item)
                                temporal_scheme[op][level].insert(find_7_idx, (3, OX[op]))
                                temporal_scheme[op][level].insert(find_7_idx + 1, (4, OY[op]))
                                B_posision_value = round(temporal_scheme_saved[op][level][idx][1] / OX[op] / OY[op])
                                temporal_scheme[op][level].insert(find_7_idx + 2, (7, B_posision_value))
                                # su_transfer_op[op][-1][-1] = [3, 4, 7]  # B -> OX, OY, B
                                temporal_scheme[op][level].remove((7, loop_single[1]))
                            elif OX[op] == 1 and OY[op] > 1:
                                find_7_item = next((x for x in temporal_scheme[op][level] if x[0] == 7), None)
                                find_7_idx = temporal_scheme[op][level].index(find_7_item)
                                temporal_scheme[op][level].insert(find_7_idx, (4, OY[op]))
                                B_posision_value = round(temporal_scheme_saved[op][level][idx][1] / OY[op])
                                temporal_scheme[op][level].insert(find_7_idx + 1, (7, B_posision_value))
                                # su_transfer_op[op][-1][-1] = [4, 7]  # B -> OY, B
                                temporal_scheme[op][level].remove((7, loop_single[1]))
                            elif OX[op] == 1 and OY[op] == 1:
                                B_posision_value = temporal_scheme_saved[op][level][idx][1]
                                # su_transfer_op[op][-1][-1] = [7]  # B -> B
                            else:
                                raise ValueError('ERROR 1 (tm)')
                            OX[op] = 1
                            OY[op] = 1
                            B[op] = round(B[op] / B_posision_value)

                        else:
                            raise ValueError('ERROR 2 (tm)')

    if not (B['W'] == B['I'] == B['O'] == 1 and OY['W'] == OY['I'] == OY['O'] == 1 and OX['W'] == OX['I'] == OX['O'] == 1):
        raise ValueError('ERROR 4')

    return spatial_scheme, flooring, temporal_scheme
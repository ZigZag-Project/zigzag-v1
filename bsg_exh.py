import math
import numpy as np
from itertools import combinations
from itertools import permutations
import time
import copy
import bsgutils as su
import classes as cls
import sys
import pickle
from collections import Counter

'''
The blocking scheme generator generates the optimal blocking schemes given the memory hierarchy

The sequence of operations that are done are 
    1. blocking scheme generation, 
    2. data reuse cleanup
    3. loop order combinations

The first function to be called is bsg, followed by data_reuse_cleanup and finally loop_order_combinations

The output of the loop_order_combinations function is a list of temporal loops ready to be used by the cost model
'''


def data_reuse_cleanup(layer_loop_info, list_LPF_schemes, spatial_unrolling, mem_size, precision):
    '''
    @param layer_loop_info: The size of the layer loops
    @type layer_loop_info: dict
    @param list_LPF_schemes: A list of blocking schemes
    @param spatial_unrolling: The spatial unrolling given by the memory scheme
    @return: a list_LPF_schemes without all the schemes that exhibited data reuse equal to 1 in intermediate levels

    Function steps
    ==============
        1. define the data reuse of WEIGHTS AND OUTPUTS. INPUTs are not considered as of now, will be implemented later
        2. look for intermediate levels (threfore excluding level 0 and last level) that have data reuse == 1. If it is the case, the scheme is discarded
        3. if no data reuse == 1 in intermediate levels, append the scheme to the good_list list
    '''
    good_list = []

    spatial_loop = cls.SpatialLoop.extract_loop_info(spatial_unrolling, layer_loop_info)
    layer = cls.Layer.extract_layer_info(layer_loop_info)

    # i_tot = layer.total_data_size['I']
    # w_tot = layer.total_data_size['W']
    # o_tot = layer.total_data_size['O']
    
    data_reuse_pruning_levels = {'W':[],'I':[],'O':[]}
    for op in ['W','I','O']:
        for level in mem_size[op]:
            if level <= precision[op] or level >= layer.total_data_size[op] * precision[op]:
                data_reuse_pruning_levels[op].append(False)
            else:
                data_reuse_pruning_levels[op].append(True)
    for bs in list_LPF_schemes:
        temporal_loop = cls.TemporalLoop.extract_loop_info(layer, bs, spatial_loop)
        loop = cls.Loop.extract_loop_info(layer, temporal_loop, spatial_loop, precision, False)

        # DATA REUSE CLEANING IS DONE ONLY FOR WEIGHT AND OUTPUT
        discard = False

        for operand in ['W', 'O']:
            for ii_dr, dr in enumerate(loop.data_reuse[operand][1:]):
                if data_reuse_pruning_levels[operand][ii_dr]:
                    if dr == 1 and np.prod([d for d in loop.data_reuse[operand][:ii_dr+1]])/layer.total_data_reuse[operand] != 1:
                        discard = True
                        break
            if discard:
                break
            # for level, level_reuse in enumerate(loop.data_reuse[operand][1:-1]):
            #     if level_reuse == 1:
            #         discard = True
            #         break
            # if discard:
            #     break
        if not discard:
            good_list.append(bs)
    # print('\n  |-> data reuse cleanup:', len(list_LPF_schemes), ' to ', len(good_list))
    return good_list


def even_uneven(LPF_scheme):
    '''
        @param LPF_scheme: Single LPF scheme
        @type LPF_scheme : dict
        @return: a list of valid permutations of the LPF scheme

        Function steps
        ==============
            Iteratively split the LPF scheme in virtual memory level, starting from the innermost set of LPFs
            1. Find innermost set of common LPFs among different operands
            2. Generate possible permuations for the found set of LPFs
            3. For each permutation append it to the partial scheme. For the first iteration the partial scheme is a single empty one
            4. Each partial scheme is appended to a list of partial schemes. The set of LPFs appended is removed from the original LPF scheme
            - For the subsequent iterations, each virtual level is appended on top of the previous partial schemes.
            - Steps 1 to 4 are repeated until all common sets of LPFs (virtual levels) have been appended.
        '''
    bs_next = []
    finished = False
    # tmp_LPF_scheme = copy.deepcopy(LPF_scheme)
    tmp_LPF_scheme = {'W': [], 'I': [], 'O': []}
    for op in LPF_scheme:
        for ii_lev, lev in enumerate(LPF_scheme[op]):
            tmp_LPF_scheme[op].append([])
            for pf in lev:
                tmp_LPF_scheme[op][ii_lev].append(tuple([pf[0], pf[1], ii_lev]))
    lo_ok = True
    operand_irrelevant = {'W': [3, 4, 7], 'I': [6], 'O': [1, 2, 5]}
    length_bs = {
        0: len(LPF_scheme['W']),
        1: len(LPF_scheme['I']),
        2: len(LPF_scheme['O']),
    }
    bsx = {
        'W': [[]],
        'I': [[]],
        'O': [[]]
    }

    # bs_next is a list of partial schemes that gets updated after each iteration
    # bsx is an empty scheme
    bs_next.append(bsx)
    even = True
    # The while loop is looped through until all virtual levels are assigned and the schemes are complete
    while not finished:
        # The first step is the identification of the virtual memory level
        # Basically corresponds to comparing for each operand what is the number of LPFs in the innermost level of the
        # *TMP_LPF_SCHEME* (which is different from the LPF scheme which is initially fed to the function and is a copy of the
        # former that gets updated after each iteration by removing the assigned the virtual level) and setting as
        # virtual memory level the one which has the lowest number of LPFs
        if any(tmp_LPF_scheme[op] == [[]] for op in tmp_LPF_scheme):
            min_virtual_roof = 0
        else:
            min_virtual_roof = min([len(tmp_LPF_scheme[op][0]) for op in tmp_LPF_scheme])
        min_virtual_roof_list = []

        for operand in tmp_LPF_scheme:
            if tmp_LPF_scheme[operand]:
                if len(tmp_LPF_scheme[operand][0]) == min_virtual_roof:
                    min_virtual_roof_list.append([operand, len(tmp_LPF_scheme[operand][0])])

        virtual_level = {'W': [], 'I': [], 'O': []}
        for operand in virtual_level:
            for ii_pf in range(0, min_virtual_roof):
                virtual_level[operand].append(tmp_LPF_scheme[operand][0][ii_pf])

        pf = list(zip(*[virtual_level['W'], virtual_level['I'], virtual_level['O']]))
        # print()

        for pf_op in pf:
            for ii_pfx, pfx in enumerate(pf_op):
                op_in = {0: 'W', 1: 'I', 2: 'O'}

                if any([pfx[2] < pfy[2] and pfx[2] < length_bs[ii_pfx] - 1 for pfy in pf_op]):
                    # print()
                    # for op in ['W','I','O']:
                    #     print(LPF_scheme[op])
                    # print(length_bs)
                    # print(virtual_level)
                    # print(pf)
                    even = False
                    break
            if not even:
                break
        if not even:
            break
        for operand in tmp_LPF_scheme:
            for pf in virtual_level[operand]:
                tmp_LPF_scheme[operand][0].remove(pf)

        # Merge LPFs in the virtual memory level with the same loop type
        # EG (6, 2), (3, 3), (6, 5) -> (6, 10), (3, 3)

        # Remove the virtual level from tmp_LPF_scheme
        for operand in ['W', 'I', 'O']:
            if tmp_LPF_scheme[operand]:
                if not tmp_LPF_scheme[operand][0]:
                    tmp_LPF_scheme[operand].remove(tmp_LPF_scheme[operand][0])

        # The assignment process of different virtual levels terminates when tmp_LPF_scheme is an empty scheme,
        # having had all its virtual levels removed in previous iterations
        if all(not tmp_LPF_scheme[op] for op in tmp_LPF_scheme):
            finished = True
        # print(virtual_level)
        # print(pf)

    return even


def st_loop_orders_gen(tmp_virtual_level_aux, O_st, I_st, W_st):
    vl_order_list = []
    for st_list in O_st + I_st + W_st:
        order = []
        for l in tmp_virtual_level_aux:
            order.append(st_list[l[0]])
        vl = [x for _, x in sorted(zip(order, tmp_virtual_level_aux))]
        if vl not in vl_order_list:
            vl_order_list.append(vl)
    return vl_order_list


def loop_order_combinations_stationary_v2(LPF_scheme):
    '''
        @param LPF_scheme: Single LPF scheme
        @type LPF_scheme : dict
        @return: a list of valid permutations of the LPF scheme

        Function steps
        ==============
            Iteratively split the LPF scheme in virtual memory level, starting from the innermost set of LPFs
            1. Find innermost set of common LPFs among different operands
            2. Generate possible permuations for the found set of LPFs
            3. For each permutation append it to the partial scheme. For the first iteration the partial scheme is a single empty one
            4. Each partial scheme is appended to a list of partial schemes. The set of LPFs appended is removed from the original LPF scheme
            - For the subsequent iterations, each virtual level is appended on top of the previous partial schemes.
            - Steps 1 to 4 are repeated until all common sets of LPFs (virtual levels) have been appended.
        '''
    bs_next = []
    finished = False
    tmp_LPF_scheme = copy.deepcopy(LPF_scheme)
    lo_ok = True
    operand_irrelevant = {'W': [3, 4, 7], 'I': [6], 'O': [1, 2, 5]}
    O_st = [
        [None, 1, 2, 4, 5, 3, 6, 7],
        [None, 2, 1, 4, 5, 3, 6, 7]
    ]
    I_st = [
        [None, 2, 3, 4, 5, 6, 1, 7],
        [None, 3, 2, 4, 5, 6, 1, 7],
        [None, 3, 4, 2, 5, 6, 1, 7],
        [None, 3, 4, 5, 2, 6, 1, 7]
    ]
    W_st = [
        [None, 4, 5, 1, 2, 6, 7, 3],
        [None, 4, 5, 2, 1, 6, 7, 3]
    ]
    length_bs = {
        'W': len(LPF_scheme['W']),
        'I': len(LPF_scheme['I']),
        'O': len(LPF_scheme['O']),
    }
    bsx = {
        'W': [[]],
        'I': [[]],
        'O': [[]]
    }

    # bs_next is a list of partial schemes that gets updated after each iteration
    # bsx is an empty scheme
    bs_next.append(bsx)

    # The while loop is looped through until all virtual levels are assigned and the schemes are complete
    while not finished:
        # The first step is the identification of the virtual memory level
        # Basically corresponds to comparing for each operand what is the number of LPFs in the innermost level of the
        # *TMP_LPF_SCHEME* (which is different from the LPF scheme which is initially fed to the function and is a copy of the
        # former that gets updated after each iteration by removing the assigned the virtual level) and setting as
        # virtual memory level the one which has the lowest number of LPFs
        if any(tmp_LPF_scheme[op] == [[]] for op in tmp_LPF_scheme):
            min_virtual_roof = 0
        else:
            try:
                min_virtual_roof = min(
                    [len(tmp_LPF_scheme['W'][0]), len(tmp_LPF_scheme['I'][0]), len(tmp_LPF_scheme['O'][0])])
            except IndexError:
                print('tmp_LPF_scheme', tmp_LPF_scheme)
                min_virtual_roof = 0
        min_virtual_roof_list = []

        for operand in ['W', 'I', 'O']:
            if tmp_LPF_scheme[operand]:
                if len(tmp_LPF_scheme[operand][0]) == min_virtual_roof:
                    min_virtual_roof_list.append([operand, len(tmp_LPF_scheme[operand][0])])
        virtual_level = copy.deepcopy(tmp_LPF_scheme[min_virtual_roof_list[0][0]][0])

        for operand in ['W', 'I', 'O']:
            for pf in virtual_level:
                try:
                    tmp_LPF_scheme[operand][0].remove(pf)
                except IndexError:
                    lo_ok = False
                    return lo_ok, bs_next
                except ValueError:
                    lo_ok = False
                    return lo_ok, bs_next

        # Merge LPFs in the virtual memory level with the same loop type
        # EG (6, 2), (3, 3), (6, 5) -> (6, 10), (3, 3)
        tmp_virtual_level = []
        for pf in virtual_level:
            if pf[0] not in [x[0] for x in tmp_virtual_level]:
                c = np.prod([x[1] for x in virtual_level if x[0] == pf[0]])
                tmp_virtual_level.append(tuple([pf[0], c]))

        bs_old = copy.deepcopy(bs_next)
        bs_next = []
        if tmp_virtual_level == []:
            tmp_bs = copy.deepcopy(bs_old[0])
            for op in ['W', 'I', 'O']:
                if op in [opx[0] for opx in min_virtual_roof_list]:
                    if len(tmp_bs[op]) != length_bs[op]:
                        tmp_bs[op].append([])
            bs_next.append(tmp_bs)
        else:
            tmp_virtual_level_aux = copy.deepcopy(tmp_virtual_level)
            for bs in bs_old:
                tmp_virtual_level_fully_pmt = st_loop_orders_gen(tmp_virtual_level_aux, O_st, I_st, W_st)
                for vl in tmp_virtual_level_fully_pmt:
                    tmp_bs = copy.deepcopy(bs)
                    for op in ['W', 'I', 'O']:
                        tmp_bs[op][len(bs[op]) - 1] += list(vl)
                        if op in [opx[0] for opx in min_virtual_roof_list]:
                            if len(tmp_bs[op]) != length_bs[op]:
                                tmp_bs[op].append([])
                    if tmp_bs not in bs_next:
                        bs_next.append(tmp_bs)

        # Remove the virtual level from tmp_LPF_scheme
        for operand in tmp_LPF_scheme:
            if tmp_LPF_scheme[operand]:
                if not tmp_LPF_scheme[operand][0]:
                    tmp_LPF_scheme[operand].remove(tmp_LPF_scheme[operand][0])

        # The assignement process of different virtual levels terminates when tmp_LPF_scheme is an empty scheme,
        # having had all its virtual levels removed in previous iterations
        if tmp_LPF_scheme['W'] or tmp_LPF_scheme['I'] or tmp_LPF_scheme['O']:
            finished = False
        else:
            finished = True

    return lo_ok, bs_next


def loop_order_combinations_stationary(LPF_scheme):
    '''
        @param LPF_scheme: Single LPF scheme
        @type LPF_scheme : dict
        @return: a list of valid permutations of the LPF scheme

        Function steps
        ==============
            Iteratively split the LPF scheme in virtual memory level, starting from the innermost set of LPFs
            1. Find innermost set of common LPFs among different operands
            2. Generate possible permuations for the found set of LPFs
            3. For each permutation append it to the partial scheme. For the first iteration the partial scheme is a single empty one
            4. Each partial scheme is appended to a list of partial schemes. The set of LPFs appended is removed from the original LPF scheme
            - For the subsequent iterations, each virtual level is appended on top of the previous partial schemes.
            - Steps 1 to 4 are repeated until all common sets of LPFs (virtual levels) have been appended.
        '''
    bs_next = []
    finished = False
    tmp_LPF_scheme = copy.deepcopy(LPF_scheme)
    lo_ok = True
    operand_irrelevant = {'W': [3, 4, 7], 'I': [6], 'O': [1, 2, 5]}
    length_bs = {
        'W': len(LPF_scheme['W']),
        'I': len(LPF_scheme['I']),
        'O': len(LPF_scheme['O']),
    }
    bsx = {
        'W': [[]],
        'I': [[]],
        'O': [[]]
    }

    # bs_next is a list of partial schemes that gets updated after each iteration
    # bsx is an empty scheme
    bs_next.append(bsx)

    # The while loop is looped through until all virtual levels are assigned and the schemes are complete
    while not finished:
        # The first step is the identification of the virtual memory level
        # Basically corresponds to comparing for each operand what is the number of LPFs in the innermost level of the
        # *TMP_LPF_SCHEME* (which is different from the LPF scheme which is initially fed to the function and is a copy of the
        # former that gets updated after each iteration by removing the assigned the virtual level) and setting as
        # virtual memory level the one which has the lowest number of LPFs
        if any(tmp_LPF_scheme[op] == [[]] for op in tmp_LPF_scheme):
            min_virtual_roof = 0
        else:
            try:
                min_virtual_roof = min(
                    [len(tmp_LPF_scheme['W'][0]), len(tmp_LPF_scheme['I'][0]), len(tmp_LPF_scheme['O'][0])])
            except IndexError:
                print('tmp_LPF_scheme', tmp_LPF_scheme)
                min_virtual_roof = 0
        min_virtual_roof_list = []

        for operand in ['W', 'I', 'O']:
            if tmp_LPF_scheme[operand]:
                if len(tmp_LPF_scheme[operand][0]) == min_virtual_roof:
                    min_virtual_roof_list.append([operand, len(tmp_LPF_scheme[operand][0])])
        virtual_level = copy.deepcopy(tmp_LPF_scheme[min_virtual_roof_list[0][0]][0])

        for operand in ['W', 'I', 'O']:
            for pf in virtual_level:
                try:
                    tmp_LPF_scheme[operand][0].remove(pf)
                except IndexError:
                    lo_ok = False
                    return lo_ok, bs_next
                except ValueError:
                    lo_ok = False
                    return lo_ok, bs_next

        # Merge LPFs in the virtual memory level with the same loop type
        # EG (6, 2), (3, 3), (6, 5) -> (6, 10), (3, 3)
        tmp_virtual_level = []
        for pf in virtual_level:
            if pf[0] not in [x[0] for x in tmp_virtual_level]:
                c = np.prod([x[1] for x in virtual_level if x[0] == pf[0]])
                tmp_virtual_level.append(tuple([pf[0], c]))

        bs_old = copy.deepcopy(bs_next)
        bs_next = []
        if tmp_virtual_level == []:
            tmp_bs = copy.deepcopy(bs_old[0])
            bs_next.append(tmp_bs)
            for op in ['W', 'I', 'O']:
                if op in [opx[0] for opx in min_virtual_roof_list]:
                    if len(tmp_bs[op]) != length_bs[op]:
                        tmp_bs[op].append([])
        else:
            tmp_virtual_level_aux = copy.deepcopy(tmp_virtual_level)
            tmp_virtual_level_rel = {
                'W': [lpf for lpf in tmp_virtual_level_aux if lpf[0] not in operand_irrelevant['W']],
                'I': [lpf for lpf in tmp_virtual_level_aux if lpf[0] not in operand_irrelevant['I']],
                'O': [lpf for lpf in tmp_virtual_level_aux if lpf[0] not in operand_irrelevant['O']]}
            operand_short_cut = None
            for operand in ['W', 'I', 'O']:
                if not tmp_virtual_level_rel[operand]:
                    operand_short_cut = operand
                    break
            for bs in bs_old:
                # --- shortcut for fully permutation when all loop sets at current vm level are ir for one operand.
                if operand_short_cut:
                    tmp_virtual_level_irrel = tmp_virtual_level_aux
                    virtual_level_irrel_perm = list(permutations(tmp_virtual_level_irrel))
                    for vl in virtual_level_irrel_perm:
                        tmp_bs = copy.deepcopy(bs)
                        for op in ['W', 'I', 'O']:
                            tmp_bs[op][len(bs[op]) - 1] += list(vl)
                            if op in [opx[0] for opx in min_virtual_roof_list]:
                                if len(tmp_bs[op]) != length_bs[op]:
                                    tmp_bs[op].append([])
                        if tmp_bs not in bs_next:
                            bs_next.append(tmp_bs)
                # ------------------------------------------------------------------------------------------------
                else:
                    for operand in ['W', 'I', 'O']:
                        tmp_virtual_level_irrel = [lpf for lpf in tmp_virtual_level_aux if
                                                   lpf[0] in operand_irrelevant[operand]]
                        virtual_level_irrel_perm = list(permutations(tmp_virtual_level_irrel))
                        for vl in virtual_level_irrel_perm:
                            tmp_bs = copy.deepcopy(bs)
                            for op in ['W', 'I', 'O']:
                                tmp_bs[op][len(bs[op]) - 1] += list(vl) + tmp_virtual_level_rel[operand]
                                if op in [opx[0] for opx in min_virtual_roof_list]:
                                    if len(tmp_bs[op]) != length_bs[op]:
                                        tmp_bs[op].append([])
                            if tmp_bs not in bs_next:
                                bs_next.append(tmp_bs)

        # Remove the virtual level from tmp_LPF_scheme
        for operand in tmp_LPF_scheme:
            if tmp_LPF_scheme[operand]:
                if not tmp_LPF_scheme[operand][0]:
                    tmp_LPF_scheme[operand].remove(tmp_LPF_scheme[operand][0])

        # The assignement process of different virtual levels terminates when tmp_LPF_scheme is an empty scheme,
        # having had all its virtual levels removed in previous iterations
        if tmp_LPF_scheme['W'] or tmp_LPF_scheme['I'] or tmp_LPF_scheme['O']:
            finished = False
        else:
            finished = True

    return lo_ok, bs_next


def loop_order_combinations_exhaustive(blocking_scheme):
    """
    @param LPF_scheme: Single LPF scheme
    @type LPF_scheme : dict
    @return: a list of valid permutations of the LPF scheme

    Function steps
    ==============
        Iteratively split the LPF scheme in virtual memory level, starting from the innermost set of LPFs
        1. Find innermost set of common LPFs among different operands
        2. Generate possible permuations for the found set of LPFs
        3. For each permutation append it to the partial scheme. For the first iteration the partial scheme is a single empty one
        4. Each partial scheme is appended to a list of partial schemes. The set of LPFs appended is removed from the original LPF scheme
        - For the subsequent iterations, each virtual level is appended on top of the previous partial schemes.
        - Steps 1 to 4 are repeated until all common sets of LPFs (virtual levels) have been appended.
    """
    bs_next = []
    finished = False
    tmp_blocking_scheme = copy.deepcopy(blocking_scheme)
    lo_ok = True

    length_bs = {
        'W': len(blocking_scheme['W']),
        'I': len(blocking_scheme['I']),
        'O': len(blocking_scheme['O']),
    }
    bsx = {
        'W': [[]],
        'I': [[]],
        'O': [[]]
    }

    # bs_next is a list of partial schemes that gets updated after each iteration
    # bsx is an empty scheme
    bs_next.append(bsx)

    # The while loop is looped through until all virtual levels are assigned and the schemes are complete
    while not finished:
        if any(tmp_blocking_scheme[op] == [[]] for op in tmp_blocking_scheme):
            min_virtual_roof = 0
        else:
            try:
                min_virtual_roof = min([len(tmp_blocking_scheme[op][0]) for op in tmp_blocking_scheme])
            except IndexError:
                print(tmp_blocking_scheme)
                min_virtual_roof = 0
        min_virtual_roof_list = []
        # Find lowest memory level to be virtualized

        for operand in tmp_blocking_scheme:
            if tmp_blocking_scheme[operand]:
                if len(tmp_blocking_scheme[operand][0]) == min_virtual_roof:
                    min_virtual_roof_list.append([operand, len(tmp_blocking_scheme[operand][0])])
        virtual_level = copy.deepcopy(tmp_blocking_scheme[min_virtual_roof_list[0][0]][0])

        for operand in tmp_blocking_scheme:
            for pf in virtual_level:
                try:
                    tmp_blocking_scheme[operand][0].remove(pf)
                except IndexError:
                    # print(blocking_scheme)
                    lo_ok = False
                    return lo_ok, bs_next
                except ValueError:
                    # print(blocking_scheme)
                    lo_ok = False
                    return lo_ok, bs_next

        # Merge LPFs in the virtual memory level with the same loop type
        # EG (6, 2), (3, 3), (6, 5) -> (6, 10), (3, 3)
        tmp_virtual_level = []
        for pf in virtual_level:
            if pf[0] not in [x[0] for x in tmp_virtual_level]:
                c = np.prod([x[1] for x in virtual_level if x[0] == pf[0]])
                tmp_virtual_level.append(tuple([pf[0], c]))

        # Generate all possible permutations within the virtual level
        virtual_level_comb = list(permutations(tmp_virtual_level))

        # Append the permutations found in the virtual level to the partial schemes found previously, contained in bs_next
        # Create a new bs_next list that contains the new schemes
        bs_old = copy.deepcopy(bs_next)
        bs_next = []
        for bs in bs_old:
            for vl in virtual_level_comb:
                tmp_bs = copy.deepcopy(bs)
                for operand in ['W', 'I', 'O']:
                    tmp_bs[operand][len(bs[operand]) - 1] += vl
                    if operand in [opx[0] for opx in min_virtual_roof_list]:
                        if len(tmp_bs[operand]) != length_bs[operand]:
                            tmp_bs[operand].append([])
                # if tmp_bs not in bs_next:
                #     bs_next.append(tmp_bs)
                bs_next.append(tmp_bs)

        for operand in tmp_blocking_scheme:
            if tmp_blocking_scheme[operand]:
                if not tmp_blocking_scheme[operand][0]:
                    tmp_blocking_scheme[operand].remove(tmp_blocking_scheme[operand][0])

        # The assignment process of different virtual levels terminates when tmp_LPF_scheme is an empty scheme,
        # having had all its virtual levels removed in previous iterations
        if all(not tmp_blocking_scheme[op] for op in tmp_blocking_scheme):
            finished = True

    return lo_ok, bs_next


def bsg(mem_size, mem_share, precision, utilization_rate, layer_loop_info, layer_index, spatial_unrolling, drc_enabled,
        stationary_enable, tmg_scheme_hint=[]):
    operand_irrelevant = {
        'W': [3, 4, 7],
        'O': [1, 2, 5],
        'I': [6]
    }

    # loops_pf contains prime factors for each loop type.
    # After each loop assignment the relative list is updated
    loops_pf = {
        7: [],
        6: [],
        5: [],
        4: [],
        3: [],
        2: [],
        1: []
    }

    # For each operand and memory level defines the effective size of blockings that it can contain
    mem_block_size = {
        'W': [],
        'O': [],
        'I': []
    }

    m_size = max([len(mem_size['I']), len(mem_size['W']), len(mem_size['O'])])

    # Auxiliary term that stores the temporary roof values.
    # For each operand it is defined to what memory level the roof belongs and how much space is still left to be assigned
    # 'operand' : [memory_level, roof_value]
    roof = {
        'O': [0, 0],
        'I': [0, 0],
        'W': [0, 0]
    }

    # Init blocking scheme
    bs = {
        'W': [[] for i in range(len(mem_size['W']))],
        'O': [[] for i in range(len(mem_size['O']))],
        'I': [[] for i in range(len(mem_size['I']))],
    }

    # Append those LPFs that are relative to the spatial unrollings to the memory level where they have to be stored
    for op in ['W', 'I', 'O']:
        for level in range(0, len(spatial_unrolling[op])):
            if spatial_unrolling[op][level]:
                for unroll in range(0, len(spatial_unrolling[op][level])):
                    sp = su.prime_factors(spatial_unrolling[op][level][unroll][1])
                    bs[op][level] += [tuple([spatial_unrolling[op][level][unroll][0], j]) for j in sp]

    # Assign prime factors for each loop type
    ll = {1: 'FX', 2: 'FY', 3: 'OX', 4: 'OY', 5: 'C', 6: 'K', 7: 'B'}

    for loop_type in loops_pf:
        loops_pf[loop_type] = su.prime_factors(layer_loop_info[ll[loop_type]])

    loops_pf_irrelevant_unrolled = {'W': [], 'I': [], 'O': []}

    # Remove the LPFs relative to the spatial unrollings from loops_pf
    for level in range(0, len(spatial_unrolling['W'])):
        for unroll in range(0, len(spatial_unrolling['W'][level])):
            sp = su.prime_factors(spatial_unrolling['W'][level][unroll][1])
            for j in sp:
                try:
                    loops_pf[spatial_unrolling['W'][level][unroll][0]].remove(j)
                except:
                    print('in bsg, spatial_unrolling', spatial_unrolling)
                    print(j)

    # Assign relative memory sizes for each operand at each memory level
    for operand in ['W', 'I', 'O']:
        for level in range(0, len(mem_size[operand])):
            mem_block_size[operand].append(math.floor(mem_size[operand][level] / precision[operand]))

    # Initialize roof values
    for operand in ['W', 'I', 'O']:
        roof[operand][1] = mem_block_size[operand][0]
    roof = su.update_roof(bs, spatial_unrolling, [], roof, mem_share, mem_size, precision, operand_irrelevant, loops_pf,
                          layer_loop_info)
    r_op, r_lev = roof.keys(), roof.values()
    r_lev, r_size = zip(*r_lev)

    min_roof_aux = ['', 0, max(r_size)]
    next_min_roof = ['', 0, max(r_size)]
    last_roof = ['', 0, max(r_size)]

    # The root_node is a SchedulerNode object. It corresponds to the very initial partial scheme, which is empty and with no
    # temporal LPF assigned (only the spatial LPF are present in bs). Each SchedulerNode object is defined by
    # - Its partial scheme of LPFs (bs in this case)
    # - The set of LPFs still to be assigned (loops_pf)
    # - The roof variable which identifies at which memory level in the hierarchy for each operand the assignments must be carried out
    root_node = su.SchedulerNode(bs, roof, loops_pf)
    next_partial_LPF_schemes_list = [root_node]
    final_LPF_schemes_list = []
    finished = False

    old_clean_LPF_schemes_list = []
    rep = 0
    # finished = True

    while not finished:
        # finished = True
        # Check if all the nodes in next_partial_LPF_schemes_list are leaf nodes
        # The cleaner eliminates duplicate partial schemes
        clean_LPF_schemes_list = su.cleaner(next_partial_LPF_schemes_list)
        # The following if condition checks whether the previous assignment step was unsuccessful.
        # If one (originally three) consecutive LPF assignments are unsuccessful, the assignment process is terminated (finished = True),
        # and only those schemes that are finished (leaf.over == True) are considered for successive loop ordering step
        if clean_LPF_schemes_list:
            if all(any(old_node == node for old_node in old_clean_LPF_schemes_list) for node in
                   clean_LPF_schemes_list):
                rep += 1
                if rep == 1:  # rep == 3
                    final_LPF_schemes_list = [bn for bn in clean_LPF_schemes_list if bn.leaf_over == True]
                    # if not final_LPF_schemes_list:
                    # print('no fitting found!')
                    finished = True
                    # print('case2')
                    continue
            else:
                rep = 0
        old_clean_LPF_schemes_list = copy.deepcopy(clean_LPF_schemes_list)
        # The following if condition checks whether all the partial schemes in the cleaned list of schemes are finished.
        # If all partial schemes are finished (leaf_over == True) then the while loop is broken (finished = True) and the
        # scheme are considered for the reordering step
        if all(nodes.leaf_over == True for nodes in clean_LPF_schemes_list) and clean_LPF_schemes_list.__len__() > 0:
            final_LPF_schemes_list = [bn for bn in clean_LPF_schemes_list if bn.leaf_over == True]
            finished = True
            # print('case3', clean_LPF_schemes_list.__len__())
            continue

        partial_LPF_schemes_list = copy.deepcopy(clean_LPF_schemes_list)
        next_partial_LPF_schemes_list = []
        # partial_LPF_schemes_list is a copy of the cleaned list of partial schemes obtained from previous assignments.
        # Each partial scheme is analyzed *individually*: a set of fitting LPF combinations will be found that can be stacked
        # for each partial scheme. Each of these sets will contribute to a new partial scheme, that will be appended to
        # next_partial_LPF_schemes_list
        for z in range(0, len(partial_LPF_schemes_list)):
            # The following if condition checks whether the partial scheme is in fact a complete scheme
            # A complete scheme is a scheme where all the LPF have been assigned
            # If it is a complete scheme (leaf_over == True), no assignement is needed and the scheme is appended to next_partial_LPF_schemes_list
            if partial_LPF_schemes_list[z].leaf_over:
                leaf_node = su.SchedulerNode(partial_LPF_schemes_list[z].LPF_scheme, partial_LPF_schemes_list[z].roof,
                                             partial_LPF_schemes_list[z].loops_pf)
                leaf_node.set_leaf_over()
                next_partial_LPF_schemes_list.append(leaf_node)
                continue
            roof = copy.deepcopy(partial_LPF_schemes_list[z].roof)
            LPF_scheme = copy.deepcopy(partial_LPF_schemes_list[z].LPF_scheme)
            loops_pf = copy.deepcopy(partial_LPF_schemes_list[z].loops_pf)

            r_op, r_lev = roof.keys(), roof.values()
            r_lev, r_size = zip(*r_lev)

            min_roof_aux = ['', m_size, max(r_size)]
            tmp_min_roof = ['', m_size, max(r_size)]

            # Given the partial scheme and its relative roof value, a list of minimum roofs is found.
            # A minimum roof correspond to a roof value that has the lowest level in the mem hierarchy and/or
            # the smallest amount of blockings space available if equivalent level in the hierarchy
            # EG between W : [0, 5] and I : [2, 245] the min roof will be the 'W' one, since it belongs to a lower level in the hierarchy
            # EG between W : [0, 5] and I : [0, 2] the min roof will be the 'I' one, since it has less blockings space available
            min_roof_list = []
            for operand in roof:
                if any((roof[roof_op][0] < roof[operand][0]) for roof_op in roof) and \
                        not all([roof[opx][0] == len(mem_size[opx]) - 1 for opx in ['W', 'I', 'O']]):
                    if roof[operand][0] == (len(mem_size[operand]) - 1):
                        continue
                if (roof[operand][0] == (len(mem_size[operand]) - 1) and
                        (roof[operand][1] == 1 or roof[operand][1] == 0)):
                    continue
                else:
                    if roof[operand][1] <= min_roof_aux[2]:
                        min_roof_aux[0] = operand
                        min_roof_aux[1] = roof[operand][0]
                        min_roof_aux[2] = roof[operand][1]
            for operand in roof:
                if roof[operand][0] == min_roof_aux[1] and roof[operand][1] == min_roof_aux[2]:
                    tmp_min_roof = ['', m_size, r_size]
                    tmp_min_roof[0] = operand
                    tmp_min_roof[1] = roof[operand][0]
                    tmp_min_roof[2] = roof[operand][1]
                    min_roof_list.append(tmp_min_roof)

            mr_list_operand = [mr[0] for mr in min_roof_list]

            # The assignment process is carried out for *each min roof separately*
            # Each min roof will have a different set of
            for min_roof in reversed(min_roof_list):
                # Check if min roof belongs to a shared level of memory. If so, create a list with all shared memory levels related to min roof
                shared_min_roof_levels = [tuple([min_roof[0], min_roof[1]])]
                for shared_set in mem_share:
                    if tuple([min_roof[0], min_roof[1]]) in mem_share[shared_set]:
                        shared_min_roof_levels = mem_share[shared_set]

                fitting_combination = []
                min_list_fitting_combinations = []
                loop_blocks = []

                # List LPFs that can be assigned to min roof operand
                for loop_type in loops_pf:
                    for i in range(0, len(loops_pf[loop_type])):
                        loop_blocks.append(tuple([loop_type, loops_pf[loop_type][i]]))
                roof_list = [[rf, roof[rf][0], roof[rf][1]] for rf in roof]

                # Definition of k_range: Each LPF combination corresponds to a combination of LPFs drawn from loop_block
                # Normally this combinations can have range (defined by k) from 0 to the amount of LPFs to be assigned.
                # If the level where the min roof belongs to is the last level in the hierarchy then the k should be equal only
                # to all the LPF still to be assigned, in order to avoid redundant computations.
                if all([roof[rf][0] == len(mem_size[rf]) - 1 for rf in roof]):
                    k_range = range(len(loop_blocks), len(loop_blocks) + 1)
                else:
                    k_range = range(0, len(loop_blocks) + 1)

                # For each k in k_range:
                # 1. Generate all possible combinations of loop blocks with length k
                # 2. For each combination generated check if they fit within the roof (check_comb_fit function)
                # 3. If they fit, append the combination to fitting_combination

                for k in k_range:
                    tmp_comb = combinations(loop_blocks, k)
                    comb = []
                    for x in tmp_comb:
                        if sorted(x) not in comb:
                            comb.append(sorted(x))
                    mlft = len(fitting_combination)
                    for j in range(0, len(comb)):
                        is_fit = True
                        for r in roof_list:
                            is_min_roof = False
                            if r[0] == min_roof[0]: # in mr_list_operand:
                                is_min_roof = True
                            is_fit = su.check_comb_fit(LPF_scheme, spatial_unrolling, comb[j], r, mem_size, mem_share,
                                                       utilization_rate, precision, operand_irrelevant, is_min_roof,
                                                       layer_loop_info)
                            if not is_fit:
                                break
                        if not is_fit:
                            continue
                        fitting_combination.append(comb[j])
                    if len(fitting_combination) > 0 and len(fitting_combination) == mlft:
                        break

                # Clean up all duplicate combinations from fitting_combination (order is not important yet)
                # The sanitized list of fitting combinations of LPF within the roof is contained in min_list_fitting_combinations
                # ------------------------------------------------------------------------------Here--------------------
                # fitting_combination.append([])
                for ii_fit_comb in range(0, len(fitting_combination)):
                    s = sorted(fitting_combination[ii_fit_comb])
                    if s not in min_list_fitting_combinations:
                        min_list_fitting_combinations.append(s)
                # remove all fitting combs that are subsets
                if all([roof[op][0] == (len(mem_block_size[op]) - 1) for op in ['W', 'I', 'O']]):
                    for lfc in reversed(min_list_fitting_combinations):
                        if any([all(x in mlfc for x in lfc) for mlfc in min_list_fitting_combinations if mlfc != lfc]):
                            min_list_fitting_combinations.remove(lfc)
                # The following if condition is for the case when, given a specific roof, *no LPF combination* is found to fit
                # In this case the only operation to be done is to update the roof value by going up one level in the min roof
                # The loops_pf will remain the same (no comb is found and therefore no LPF has to be removed from loops_pf)
                # The LPF_scheme will remain the same (no comb is found that can be stacked on top of the previous one)
                # The roof value is updated (update level of current min_roof and then call update_roof)
                if not min_list_fitting_combinations:

                    # No need to update loops_pf
                    new_loops_pf = copy.deepcopy(loops_pf)
                    # tt = sum([len(x) for x in new_loops_pf.values()])

                    # Update roof level of current min_roof.
                    # If min roof is shared, go one memory level up for all roofs that are shared with min_roof
                    new_tmp_roof = copy.deepcopy(roof)
                    for p in range(0, len(shared_min_roof_levels)):
                        level_up = 1
                        if shared_min_roof_levels[p][1] == len(mem_block_size[shared_min_roof_levels[p][0]]) - 1:
                            level_up = 0
                        new_tmp_roof[shared_min_roof_levels[p][0]][0] = shared_min_roof_levels[p][1] + level_up
                    new_roof = su.update_roof(LPF_scheme, spatial_unrolling, [], new_tmp_roof, mem_share, mem_size,
                                              precision,
                                              operand_irrelevant, new_loops_pf, layer_loop_info)
                    isgood = su.check_node(LPF_scheme, mem_size, operand_irrelevant, mem_share, precision,
                                           layer_loop_info, utilization_rate)
                    # No need to update LPF scheme
                    new_LPF_scheme = copy.deepcopy(LPF_scheme)

                    # Generate new node in the blocking scheme tree, add it to the list
                    blocking_node = su.SchedulerNode(new_LPF_scheme, new_roof, new_loops_pf)
                    if isgood:
                        next_partial_LPF_schemes_list.append(blocking_node)
                    # print('\r bs',f'{tt:3d}','/',f'{lpf2a:3d}',' ', f'{(z+1)/len(partial_LPF_schemes_list)*100:3.0f}','%, : f ',finished_lpf_scheme,' r ',len(next_partial_LPF_schemes_list), end='')

                # If there list of fitting combinations of LPFs within the roof is NOT empty, proceed in creating new partial schemes
                # For each fitting combination a new SchedulerNode object is created that contains:
                # - An updated loops_pf in which the LPF in the fitting combination are removed
                # - An updated roof value
                # - A new partial scheme with the fitting combination stacked on top of the previous partial scheme
                else:
                    for k in range(0, len(min_list_fitting_combinations)):

                        # Generate different tmp roof and tmp loops pf since for each min roof fitting combination different roofs and loops pf will be defined
                        new_loops_pf = copy.deepcopy(loops_pf)

                        # Given remaining loop prime factors, remove those assigned in this combination to min roof
                        for i in range(0, len(min_list_fitting_combinations[k])):
                            new_loops_pf[min_list_fitting_combinations[k][i][0]].remove(
                                min_list_fitting_combinations[k][i][1])
                        # tt = sum([len(x) for x in new_loops_pf.values()])

                        # Set temporary roof with selected fitting combination
                        # This temporary roof is only used for updating the level in the min roof
                        # It is not the one that is ultimately saved in the SchedulerNode object
                        tmp_roof = su.update_roof(LPF_scheme, spatial_unrolling, min_list_fitting_combinations[k], roof,
                                                  mem_share,
                                                  mem_size, precision, operand_irrelevant, new_loops_pf,
                                                  layer_loop_info)

                        new_tmp_roof = copy.deepcopy(tmp_roof)
                        new_LPF_scheme = copy.deepcopy(LPF_scheme)
                        level0 = copy.deepcopy(min_list_fitting_combinations[k])

                        for rf in roof:
                            new_LPF_scheme[rf][roof[rf][0]] += copy.deepcopy(level0)
                        for p in range(0, len(shared_min_roof_levels)):
                            level_up = 1
                            if shared_min_roof_levels[p][1] == len(
                                    mem_block_size[shared_min_roof_levels[p][0]]) - 1:
                                level_up = 0
                            new_tmp_roof[shared_min_roof_levels[p][0]][0] = shared_min_roof_levels[p][1] + level_up

                        # The new_roof variable contains the final updated value of the new roof, with the updated blocking space available
                        # for each operand in the roof
                        new_roof = su.update_roof(new_LPF_scheme, spatial_unrolling, [], new_tmp_roof, mem_share,
                                                  mem_size,
                                                  precision,
                                                  operand_irrelevant, new_loops_pf, layer_loop_info)
                        isgood = su.check_node(new_LPF_scheme, mem_size, operand_irrelevant, mem_share, precision,
                                               layer_loop_info, utilization_rate)

                        if isgood:
                            # Generate new node in the blocking scheme tree, add it to the list
                            blocking_node = su.SchedulerNode(new_LPF_scheme, new_roof, new_loops_pf)

                            # The following if condition checks whether all the LPFs have been assigned.
                            # If so, the partial scheme is considere a complete scheme and the SchedulerNode will have leaf_over == True
                            over = False
                            if all(new_loops_pf[loop_types] == [] for loop_types in new_loops_pf):
                                over = True
                                blocking_node.set_leaf_over()
                            next_partial_LPF_schemes_list.append(blocking_node)
                        else:
                            # No need to update loops_pf
                            new_loops_pf = copy.deepcopy(loops_pf)
                            # tt = sum([len(x) for x in new_loops_pf.values()])

                            # Update roof level of current min_roof.
                            # If min roof is shared, go one memory level up for all roofs that are shared with min_roof
                            new_tmp_roof = copy.deepcopy(roof)
                            for p in range(0, len(shared_min_roof_levels)):
                                level_up = 1
                                if shared_min_roof_levels[p][1] == len(
                                        mem_block_size[shared_min_roof_levels[p][0]]) - 1:
                                    level_up = 0
                                new_tmp_roof[shared_min_roof_levels[p][0]][0] = shared_min_roof_levels[p][1] + level_up
                            new_roof = su.update_roof(LPF_scheme, spatial_unrolling, [], new_tmp_roof, mem_share,
                                                      mem_size,
                                                      precision,
                                                      operand_irrelevant, new_loops_pf, layer_loop_info)
                            isgood = su.check_node(LPF_scheme, mem_size, operand_irrelevant, mem_share, precision,
                                                   layer_loop_info, utilization_rate)

                            if isgood:
                                # No need to update LPF scheme
                                new_LPF_scheme = copy.deepcopy(LPF_scheme)
                                # Generate new node in the blocking scheme tree, add it to the list
                                blocking_node = su.SchedulerNode(new_LPF_scheme, new_roof, new_loops_pf)
                                next_partial_LPF_schemes_list.append(blocking_node)

    list_LPF_schemes = [scheme_node.LPF_scheme for scheme_node in final_LPF_schemes_list]

    # Remove the LPFs which correspond to the spatial unrollings that were previously assigned
    # for each LPF scheme in list_LPF_schemes
    for bs in list_LPF_schemes:
        for op in spatial_unrolling:
            for level in range(0, len(spatial_unrolling[op])):
                if spatial_unrolling[op][level]:
                    for unroll in range(0, len(spatial_unrolling[op][level])):
                        sp = su.prime_factors(spatial_unrolling[op][level][unroll][1])
                        for i in range(0, len(sp)):
                            try:
                                bs[op][level].remove(tuple([spatial_unrolling[op][level][unroll][0], sp[i]]))
                            except IndexError:
                                continue

    # Data reuse cleanup for the LPF schemes
    # If you want this cleanup to be disabled, set drc_enabled = 0 in the input file
    if drc_enabled:
        list_LPF_schemes = data_reuse_cleanup(layer_loop_info, list_LPF_schemes, spatial_unrolling, mem_size, precision)
        # print('\n  |-> drc: ', len(list_LPF_schemes),'/',len(list_scheme_nodes))

    # print('Loop blocking finished. Loop ordering starts.')
    total = []
    for ii_bs, bs in enumerate(list_LPF_schemes):
        # print('\r  |-> ordering: ', ii_bs, '/', len(list_LPF_schemes), end='', flush=True)
        if stationary_enable is True:
            lo_ok, lo = loop_order_combinations_stationary(bs)
            # lo_ok, lo = loop_order_combinations_stationary_v2(bs)
        else:
            lo_ok, lo = loop_order_combinations_exhaustive(bs)
        if lo_ok:
            total.extend(lo)
    # print('  |-> loop order combinations: ', len(total))
    return (total)


def bsg_fixed_order(loop_order, mem_size, mem_share, precision, utilization_rate, layer_loop_info, layer_index,
                    spatial_unrolling):
    operand_irrelevant = {
        'W': [3, 4, 7],
        'O': [1, 2, 5],
        'I': [6]
    }

    # loops_pf contains prime factors for each loop type.
    # After each loop assignment the relative list is updated
    loops_pf = {
        7: [],
        6: [],
        5: [],
        4: [],
        3: [],
        2: [],
        1: []
    }

    # For each operand and memory level defines the effective size of blockings that it can contain
    mem_block_size = {
        'W': [],
        'O': [],
        'I': []
    }

    # Auxiliary term that stores the temporary roof values.
    # For each operand it is defined to what memory level the roof belongs and how much space is still left to be assigned
    # 'operand' : [memory_level, roof_value]
    roof = {
        'O': [0, 0],
        'I': [0, 0],
        'W': [0, 0]
    }

    # Init blocking scheme
    bs = {
        'W': [[] for i in range(len(mem_size['W']))],
        'O': [[] for i in range(len(mem_size['O']))],
        'I': [[] for i in range(len(mem_size['I']))],
    }

    # Append those LPFs that are relative to the spatial unrollings to the memory level where they have to be stored
    for op in ['W', 'I', 'O']:
        for level in range(0, len(spatial_unrolling[op])):
            if spatial_unrolling[op][level]:
                for unroll in range(0, len(spatial_unrolling[op][level])):
                    sp = su.prime_factors(spatial_unrolling[op][level][unroll][1])
                    bs[op][level] += [tuple([spatial_unrolling[op][level][unroll][0], j]) for j in sp]

    # Assign prime factors for each loop type
    ll = {1: 'FX', 2: 'FY', 3: 'OX', 4: 'OY', 5: 'C', 6: 'K', 7: 'B'}

    for loop_type in loops_pf:
        loops_pf[loop_type] = su.prime_factors(layer_loop_info[ll[loop_type]])

    loops_pf_irrelevant_unrolled = {'W': [], 'I': [], 'O': []}

    # Remove the LPFs relative to the spatial unrollings from loops_pf
    for level in range(0, len(spatial_unrolling['W'])):
        for unroll in range(0, len(spatial_unrolling['W'][level])):
            sp = su.prime_factors(spatial_unrolling['W'][level][unroll][1])
            for j in sp:
                try:
                    loops_pf[spatial_unrolling['W'][level][unroll][0]].remove(j)
                except:
                    print('in bsg, spatial_unrolling', spatial_unrolling)
                    print(j)

    # Assign relative memory sizes for each operand at each memory level
    for operand in ['W', 'I', 'O']:
        for level in range(0, len(mem_size[operand])):
            mem_block_size[operand].append(math.floor(mem_size[operand][level] / precision[operand]))

    # Initialize roof values
    for operand in ['W', 'I', 'O']:
        roof[operand][1] = mem_block_size[operand][0]
    roof = su.update_roof(bs, spatial_unrolling, [], roof, mem_share, mem_size, precision, operand_irrelevant, loops_pf,
                          layer_loop_info)
    r_op, r_lev = roof.keys(), roof.values()
    r_lev, r_size = zip(*r_lev)

    min_roof_aux = ['', 0, max(r_size)]
    next_min_roof = ['', 0, max(r_size)]
    last_roof = ['', 0, max(r_size)]

    # The root_node is a SchedulerNode object. It corresponds to the very initial partial scheme, which is empty and with no
    # temporal LPF assigned (only the spatial LPF are present in bs). Each SchedulerNode object is defined by
    # - Its partial scheme of LPFs (bs in this case)
    # - The set of LPFs still to be assigned (loops_pf)
    # - The roof variable which identifies at which memory level in the hierarchy for each operand the assignments must be carried out
    root_node = su.SchedulerNode(bs, roof, loops_pf)
    next_partial_LPF_schemes_list = [root_node]
    final_LPF_schemes_list = []
    finished = False

    old_clean_LPF_schemes_list = []
    rep = 0
    # finished = True

    while not finished:
        # finished = True
        # Check if all the nodes in next_partial_LPF_schemes_list are leaf nodes
        # The cleaner eliminates duplicate partial schemes
        clean_LPF_schemes_list = su.cleaner(next_partial_LPF_schemes_list)
        # The following if condition checks whether the previous assignment step was unsuccessful.
        # If three consecutive LPF assignments are unsuccessful the assignement process is terminated (finished = True),
        # and only those schemes that are finished (leaf.over == True) are considered for successive reordering
        if clean_LPF_schemes_list:
            if all(any(old_node == node for old_node in old_clean_LPF_schemes_list) for node in
                   clean_LPF_schemes_list):
                rep += 1
                if rep == 3:
                    final_LPF_schemes_list = [bn for bn in clean_LPF_schemes_list if bn.leaf_over == True]
                    # if not final_LPF_schemes_list:
                    # print('no fitting found!')
                    finished = True
                    # print('case2')
                    continue
            else:
                rep = 0
        old_clean_LPF_schemes_list = copy.deepcopy(clean_LPF_schemes_list)
        # The following if condition checks whether all the partial schemes in the cleaned list of schemes are finished.
        # If all partial schemes are finished (leaf_over == True) then the while loop is broken (finished = True) and the
        # scheme are considered for the reordering step
        if all(nodes.leaf_over == True for nodes in clean_LPF_schemes_list) and clean_LPF_schemes_list.__len__() > 0:
            final_LPF_schemes_list = [bn for bn in clean_LPF_schemes_list if bn.leaf_over == True]
            finished = True
            # print('case3', clean_LPF_schemes_list.__len__())
            continue

        partial_LPF_schemes_list = copy.deepcopy(clean_LPF_schemes_list)
        next_partial_LPF_schemes_list = []
        # partial_LPF_schemes_list is a copy of the cleaned list of partial schemes obtained from previous assignements.
        # Each partial scheme is analyzed *individually*: a set of fitting LPF combinations will be found that can be stacked
        # for each partial scheme. Each of these sets will contribute to a new partial scheme, that will be appended to
        # next_partial_LPF_schemes_list
        for z in range(0, len(partial_LPF_schemes_list)):
            # The following if condition checks whether the partial scheme is in fact a complete scheme
            # A complete scheme is a scheme where all the LPF have been assigned
            # If it is a complete scheme (leaf_over == True), no assignement is needed and the scheme is appended to next_partial_LPF_schemes_list
            if partial_LPF_schemes_list[z].leaf_over:
                leaf_node = su.SchedulerNode(partial_LPF_schemes_list[z].LPF_scheme, partial_LPF_schemes_list[z].roof,
                                             partial_LPF_schemes_list[z].loops_pf)
                leaf_node.set_leaf_over()
                next_partial_LPF_schemes_list.append(leaf_node)
                continue
            roof = copy.deepcopy(partial_LPF_schemes_list[z].roof)
            LPF_scheme = copy.deepcopy(partial_LPF_schemes_list[z].LPF_scheme)
            loops_pf = copy.deepcopy(partial_LPF_schemes_list[z].loops_pf)
            r_op, r_lev = roof.keys(), roof.values()
            r_lev, r_size = zip(*r_lev)
            m_size = max([len(mem_size['I']), len(mem_size['W']), len(mem_size['O'])])

            min_roof_aux = ['', m_size, max(r_size)]
            tmp_min_roof = ['', m_size, max(r_size)]
            # Given the partial scheme and its relative roof value, a list of minimum roofs is found.
            # A minimum roof correspond to a roof value that has the lowest level in the mem hierarchy and/or
            # the smallest amount of blockings space available if equivalent level in the hierarchy
            # EG between W : [0, 5] and I : [2, 245] the min roof will be the 'W' one, since it belongs to a lower level in the hierarchy
            # EG between W : [0, 5] and I : [0, 2] the min roof will be the 'I' one, since it has less blockings space available
            min_roof_list = []
            for operand in roof:
                if any((roof[roof_op][0] < roof[operand][0]) for roof_op in
                       roof) and not all([roof[opx][0] == len(mem_size[opx]) - 1 for opx in ['W', 'I', 'O']]):
                    if roof[operand][0] == (len(mem_size[operand]) - 1):
                        continue
                if (roof[operand][0] == (len(mem_size[operand]) - 1) and (
                        roof[operand][1] == 1 or roof[operand][1] == 0)):
                    continue
                else:
                    if roof[operand][1] <= min_roof_aux[2]:
                        min_roof_aux[0] = operand
                        min_roof_aux[1] = roof[operand][0]
                        min_roof_aux[2] = roof[operand][1]

            for operand in roof:
                if roof[operand][0] == min_roof_aux[1] and roof[operand][1] == min_roof_aux[2]:
                    tmp_min_roof = ['', m_size, r_size]
                    tmp_min_roof[0] = operand
                    tmp_min_roof[1] = roof[operand][0]
                    tmp_min_roof[2] = roof[operand][1]
                    min_roof_list.append(tmp_min_roof)

            mr_list_operand = [mr[0] for mr in min_roof_list]
            # The assignment process is carried out for *each min roof separately*
            # Each min roof will have a different set of
            for min_roof in min_roof_list:
                # Check if min roof belongs to a shared level of memory. If so, create a list with all shared memory levels related to min roof
                shared_min_roof_levels = [tuple([min_roof[0], min_roof[1]])]
                for shared_set in mem_share:
                    if tuple([min_roof[0], min_roof[1]]) in mem_share[shared_set]:
                        shared_min_roof_levels = mem_share[shared_set]

                fitting_combination = []
                min_list_fitting_combinations = []
                loop_blocks = []

                # List LPFs that can be assigned to min roof operand
                # for loop_type in loops_pf:
                for loop_type in loops_pf:
                    for i in range(0, len(loops_pf[loop_type])):
                        loop_blocks.append(tuple([loop_type, loops_pf[loop_type][i]]))
                roof_list = [[rf, roof[rf][0], roof[rf][1]] for rf in roof]
                # Definition of k_range: Each LPF combination corresponds to a combination of LPFs drawn from loop_block
                # Normally this combinations can have range (defined by k) from 0 to the amount of LPFs to be assigned.
                # If the level where the min roof belongs to is the last level in the hierarchy then the k should be equal only
                # to all the LPF still to be assigned, in order to avoid redundant computations.
                if all([roof[rf][0] == len(mem_size[rf]) - 1 for rf in roof]):
                    k_range = range(len(loop_blocks), len(loop_blocks) + 1)
                else:
                    k_range = range(0, len(loop_blocks) + 1)

                k_range = list(k_range)
                k_range.reverse()

                loop_blocks_lpfs = {}
                for lp in loop_blocks:
                    loop_blocks_lpfs[lp[0]] = len([x for x in loop_blocks if x[0] == lp[0]])
                # For each k in k_range:
                # 1. Generate all possible combinations of loop blocks with length k
                # 2. For each combination generated check if they fit within the roof (check_comb_fit function)
                # 3. If they fit, append the combination to fitting_combination
                for k in k_range:
                    tmp_comb = combinations(loop_blocks, k)
                    aix = [loop_order.index(i[0]) for i in loop_blocks]
                    min_x = min(aix)
                    comb = []
                    for x in tmp_comb:
                        if x:
                            loop_typesx, loop_sizesx = zip(*x)
                            loop_typesa = list(set(loop_typesx))
                            loop_typesy = [loop_order.index(i) for i in loop_typesa]
                            goodcomb = True
                            loop_typesa = sorted(loop_typesa, key=lambda i: loop_order.index(i))
                            for lt in loop_typesa[:-1]:
                                if loop_blocks_lpfs[lt] != len([lpf for lpf in x if lpf[0] == lt]):
                                    goodcomb = False
                            if min(loop_typesy) == min_x and max(loop_typesy) - min(loop_typesy) == len(
                                    loop_typesy) - 1 and goodcomb:
                                x = sorted(x, key=lambda i: loop_order.index(i[0]))
                                if x not in comb:
                                    comb.append(sorted(x, key=lambda i: loop_order.index(i[0])))
                    mlft = len(fitting_combination)
                    for j in range(0, len(comb)):
                        is_fit = True
                        for r in roof_list:
                            is_min_roof = False
                            if r[0] in mr_list_operand:
                                is_min_roof = True
                            is_fit = su.check_comb_fit(LPF_scheme, spatial_unrolling, comb[j], r, mem_size, mem_share,
                                                       utilization_rate, precision, operand_irrelevant, is_min_roof,
                                                       layer_loop_info)
                            # print('FIT ', is_fit)
                            if not is_fit:
                                break
                        if not is_fit:
                            continue
                        fitting_combination.append(comb[j])
                        break
                    if len(fitting_combination) > 0 and len(fitting_combination) == mlft:
                        break
                    if is_fit:
                        break
                # Clean up all duplicate combinations from fitting_combination (order is not important yet)
                # The sanitized list of fitting combinations of LPF within the roof is contained in min_list_fitting_combinations
                # ------------------------------------------------------------------------------Here--------------------
                # fitting_combination.append([])
                for ii_fit_comb in range(0, len(fitting_combination)):
                    s = list(fitting_combination[ii_fit_comb])
                    if s not in min_list_fitting_combinations:
                        min_list_fitting_combinations.append(s)
                # remove all fitting combs that are subsets
                if all([roof[op][0] == (len(mem_block_size[op]) - 1) for op in ['W', 'I', 'O']]):
                    for lfc in reversed(min_list_fitting_combinations):
                        if any([all(x in mlfc for x in lfc) for mlfc in min_list_fitting_combinations if mlfc != lfc]):
                            min_list_fitting_combinations.remove(lfc)
                # The following if condition is for the case when, given a specific roof, *no LPF combination* is found to fit
                # In this case the only operation to be done is to update the roof value by going up one level in the min roof
                # The loops_pf will remain the same (no comb is found and therefore no LPF has to be removed from loops_pf)
                # The LPF_scheme will remain the same (no comb is found that can be stacked on top of the previous one)
                # The roof value is updated (update level of current min_roof and then call update_roof)
                if not min_list_fitting_combinations:
                    # print('wtf')
                    # No need to update loops_pf
                    new_loops_pf = copy.deepcopy(loops_pf)
                    tt = sum([len(x) for x in new_loops_pf.values()])

                    # Update roof level of current min_roof.
                    # If min roof is shared, go one memory level up for all roofs that are shared with min_roof
                    new_tmp_roof = copy.deepcopy(roof)
                    for p in range(0, len(shared_min_roof_levels)):
                        level_up = 1
                        if shared_min_roof_levels[p][1] == len(mem_block_size[shared_min_roof_levels[p][0]]) - 1:
                            level_up = 0
                        new_tmp_roof[shared_min_roof_levels[p][0]][0] = shared_min_roof_levels[p][1] + level_up
                    new_roof = su.update_roof(LPF_scheme, spatial_unrolling, [], new_tmp_roof, mem_share, mem_size,
                                              precision,
                                              operand_irrelevant, new_loops_pf, layer_loop_info)
                    isgood = su.check_node(LPF_scheme, mem_size, operand_irrelevant, mem_share, precision,
                                           layer_loop_info, utilization_rate)
                    # No need to update LPF scheme
                    new_LPF_scheme = copy.deepcopy(LPF_scheme)

                    # Generate new node in the blocking scheme tree, add it to the list
                    blocking_node = su.SchedulerNode(new_LPF_scheme, new_roof, new_loops_pf)
                    if isgood:
                        next_partial_LPF_schemes_list.append(blocking_node)


                # If there list of fitting combinations of LPFs within the roof is NOT empty, proceed in creating new partial schemes
                # For each fitting combination a new SchedulerNode object is created that contains:
                # - An updated loops_pf in which the LPF in the fitting combination are removed
                # - An updated roof value
                # - A new partial scheme with the fitting combination stacked on top of the previous partial scheme
                else:
                    for k in range(0, len(min_list_fitting_combinations)):

                        # Generate different tmp roof and tmp loops pf since for each min roof fitting combination different roofs and loops pf will be defined
                        new_loops_pf = copy.deepcopy(loops_pf)
                        # Given remaining loop prime factors, remove those assigned in this combination to min roof
                        for i in range(0, len(min_list_fitting_combinations[k])):
                            new_loops_pf[min_list_fitting_combinations[k][i][0]].remove(
                                min_list_fitting_combinations[k][i][1])
                        tt = sum([len(x) for x in new_loops_pf.values()])

                        # Set temporary roof with selected fitting combination
                        # This temporary roof is only used for updating the level in the min roof
                        # It is not the one that is ultimately saved in the SchedulerNode object
                        tmp_roof = su.update_roof(LPF_scheme, spatial_unrolling, min_list_fitting_combinations[k], roof,
                                                  mem_share,
                                                  mem_size, precision, operand_irrelevant, new_loops_pf,
                                                  layer_loop_info)

                        new_tmp_roof = copy.deepcopy(tmp_roof)
                        new_LPF_scheme = copy.deepcopy(LPF_scheme)
                        level0 = copy.deepcopy(min_list_fitting_combinations[k])

                        for rf in roof:
                            new_LPF_scheme[rf][roof[rf][0]] += copy.deepcopy(level0)
                        for p in range(0, len(shared_min_roof_levels)):
                            level_up = 1
                            if shared_min_roof_levels[p][1] == len(
                                    mem_block_size[shared_min_roof_levels[p][0]]) - 1:
                                level_up = 0
                            new_tmp_roof[shared_min_roof_levels[p][0]][0] = shared_min_roof_levels[p][1] + level_up

                        # The new_roof variable contains the final updated value of the new roof, with the updated blocking space available
                        # for each operand in the roof
                        new_roof = su.update_roof(new_LPF_scheme, spatial_unrolling, [], new_tmp_roof, mem_share,
                                                  mem_size,
                                                  precision,
                                                  operand_irrelevant, new_loops_pf, layer_loop_info)

                        isgood = su.check_node(new_LPF_scheme, mem_size, operand_irrelevant, mem_share, precision,
                                               layer_loop_info, utilization_rate)
                        if isgood:
                            # Generate new node in the blocking scheme tree, add it to the list
                            blocking_node = su.SchedulerNode(new_LPF_scheme, new_roof, new_loops_pf)

                            # The following if condition checks whether all the LPFs have been assigned.
                            # If so, the partial scheme is considere a complete scheme and the SchedulerNode will have leaf_over == True
                            over = False
                            if all(new_loops_pf[loop_types] == [] for loop_types in new_loops_pf):
                                over = True
                            if over:
                                blocking_node.set_leaf_over()
                            next_partial_LPF_schemes_list.append(blocking_node)
                        else:
                            # No need to update loops_pf
                            new_loops_pf = copy.deepcopy(loops_pf)
                            tt = sum([len(x) for x in new_loops_pf.values()])

                            # Update roof level of current min_roof.
                            # If min roof is shared, go one memory level up for all roofs that are shared with min_roof
                            new_tmp_roof = copy.deepcopy(roof)
                            for p in range(0, len(shared_min_roof_levels)):
                                level_up = 1
                                if shared_min_roof_levels[p][1] == len(
                                        mem_block_size[shared_min_roof_levels[p][0]]) - 1:
                                    level_up = 0
                                new_tmp_roof[shared_min_roof_levels[p][0]][0] = shared_min_roof_levels[p][1] + level_up
                            new_roof = su.update_roof(LPF_scheme, spatial_unrolling, [], new_tmp_roof, mem_share,
                                                      mem_size,
                                                      precision,
                                                      operand_irrelevant, new_loops_pf, layer_loop_info)
                            isgood = su.check_node(LPF_scheme, mem_size, operand_irrelevant, mem_share, precision,
                                                   layer_loop_info, utilization_rate)

                            if isgood:
                                # No need to update LPF scheme
                                new_LPF_scheme = copy.deepcopy(LPF_scheme)
                                # Generate new node in the blocking scheme tree, add it to the list
                                blocking_node = su.SchedulerNode(new_LPF_scheme, new_roof, new_loops_pf)
                                next_partial_LPF_schemes_list.append(blocking_node)

    list_LPF_schemes = [scheme_node.LPF_scheme for scheme_node in final_LPF_schemes_list]

    # Remove the LPFs which correspond to the spatial unrollings that were previously assigned
    # for each LPF scheme in list_LPF_schemes
    for bs in list_LPF_schemes:
        for op in spatial_unrolling:
            for level in range(0, len(spatial_unrolling[op])):
                if spatial_unrolling[op][level]:
                    for unroll in range(0, len(spatial_unrolling[op][level])):
                        sp = su.prime_factors(spatial_unrolling[op][level][unroll][1])
                        for i in range(0, len(sp)):
                            try:
                                bs[op][level].remove(tuple([spatial_unrolling[op][level][unroll][0], sp[i]]))
                            except IndexError:
                                continue

    for i, sch in enumerate(list_LPF_schemes):
        list_LPF_schemes[i] = loop_same_term_merge(sch)

    return list_LPF_schemes


def loop_same_term_merge(loop_unmerged):
    loop_merged = {'W': [], 'I': [], 'O': []}

    # change data format from tuple to list
    for operand in ['W', 'I', 'O']:
        for level_list in loop_unmerged[operand]:
            loop_merged[operand].append([])
            if not level_list:
                continue
            else:
                for loop_elem in level_list:
                    loop_merged[operand][-1].append(list(loop_elem))

    # merge same type loops within each memory level
    for operand in ['W', 'I', 'O']:
        for level, level_list in enumerate(loop_unmerged[operand]):
            if len(level_list) in [1, 0]:
                continue
            else:
                va_clean_idx = 0
                for va_idx in range(1, len(level_list)):
                    if level_list[va_idx - 1][0] == level_list[va_idx][0]:
                        loop_merged[operand][level][va_clean_idx][1] *= level_list[va_idx][1]
                        loop_merged[operand][level].remove(list(level_list[va_idx]))
                        va_clean_idx -= 1
                    va_clean_idx += 1

    return loop_merged

"""
Layer size rounding regarding array dimension, to support greedy mapping.
"""
import copy
from itertools import permutations

# factor_pool is a global variable.
factor_pool = []


def factorsListFunc(first, each_prod, n, single_result_list):
    if (first > n or each_prod > n):
        return
    if (each_prod == n):
        # print(*single_result_list)
        global factor_pool
        factor_pool.append(copy.deepcopy(single_result_list))
        return
    for i in range(first, n):
        if (i * each_prod > n):
            break
        if (n % i == 0):
            single_result_list.append(i)
            factorsListFunc(i, i * each_prod, n, single_result_list)
            single_result_list.remove(single_result_list[-1])


def factComb(n):
    """
    This function factorizes a number n with all possible combinations.
    """
    single_result_list = []
    factorsListFunc(2, 1, n, single_result_list)


def array_size_extend(array_size, unrolling_scheme_list):
    """
    This function finds all combinations of array 1D unrolling replication,
    i.e. unrolling multiple loop dimensions on 1D PE array dimension.
    """
    array_size_update_list = []
    aux_layer_to_su_hint_table = []
    for i, unroll_scheme in enumerate(unrolling_scheme_list):
        array_size_update_list.append([])
        array_size_update_list_1D = {0: [], 1: []}
        for j, unroll_1D in enumerate(unroll_scheme):
            if len(unroll_1D) == 1:
                array_size_update_list_1D[j].append((array_size[j],))
            else:
                factComb(array_size[j])
                for factor in factor_pool:
                    if len(factor) == len(unroll_1D):
                        for factor_permu in list(set(permutations(factor))):
                            array_size_update_list_1D[j].append(factor_permu)
                listOfGlobals = globals()
                listOfGlobals['factor_pool'] = []
        for D2 in array_size_update_list_1D[1]:
            for D1 in array_size_update_list_1D[0]:
                array_size_update_list[-1].append([D1,D2])
                aux_layer_to_su_hint_table.append(i)
    return array_size_update_list, aux_layer_to_su_hint_table


class LayerRound(object):

    def __init__(self, layer_spec_raw, array_size, unrolling_scheme_list):

        array_size, aux_layer_to_su_hint_table = array_size_extend(array_size, unrolling_scheme_list)

        ll = {1: 'FX', 2: 'FY', 3: 'OX', 4: 'OY', 5: 'C', 6: 'K', 7: 'B'}
        round_layer_info = []
        footer_info = []
        fraction_spatial_unrolling = []
        ideal_spatial_unrolling = []
        for i in range(len(unrolling_scheme_list)):
            su_list_flatten = [item for sublist in unrolling_scheme_list[i] for item in sublist]
            for j in range(len(array_size[i])):
                round_layer_info.append(copy.deepcopy(layer_spec_raw))
                footer_info.append({'B': 0, 'K': 0, 'C': 0, 'OY': 0, 'OX': 0, 'FY': 0, 'FX': 0})
                fraction_spatial_unrolling.append([])
                ideal_spatial_unrolling.append([])
                array_size_single = array_size[i][j]
                array_size_single_flatten = [item for sublist in array_size_single for item in sublist]
                for su_dim, su_type in enumerate(su_list_flatten):
                    footer = layer_spec_raw[ll[su_type]] % array_size_single_flatten[su_dim]
                    if footer != 0:
                        round_layer_info[-1][ll[su_type]] += array_size_single_flatten[su_dim] - footer
                        footer_info[-1][ll[su_type]] = footer
                        if layer_spec_raw[ll[su_type]] > array_size_single_flatten[su_dim]:
                            mapping_time = max(1,round_layer_info[-1][ll[su_type]] // array_size_single_flatten[su_dim])
                        else:
                            mapping_time = 1
                        fraction = (array_size_single_flatten[su_dim] * (mapping_time - 1) + footer) / mapping_time
                        fraction_spatial_unrolling[-1].append([su_type, fraction])
                    else:
                        footer_info[-1][ll[su_type]] = 0
                        fraction_spatial_unrolling[-1].append([su_type, array_size_single_flatten[su_dim]])
                    ideal_spatial_unrolling[-1].append([su_type, array_size_single_flatten[su_dim]])
        self.round_layer_info = round_layer_info
        self.footer_info = footer_info
        self.fraction_su = fraction_spatial_unrolling
        self.ideal_su = ideal_spatial_unrolling
        self.aux_layer_to_su_hint_table = aux_layer_to_su_hint_table

    @classmethod
    def layer_rounding(cls, layer_spec_raw, array_size, unrolling_scheme_list):
        return cls(layer_spec_raw, array_size, unrolling_scheme_list)

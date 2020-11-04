"""
Layer size rounding regarding array dimension, to support greedy mapping.
"""
from copy import deepcopy
from itertools import permutations
from itertools import combinations

# factor_pool is a global variable.
factor_pool = []


def factorsListFunc(first, each_prod, n, single_result_list):
    if (first > n or each_prod > n):
        return
    if (each_prod == n):
        # print(*single_result_list)
        global factor_pool
        factor_pool.append(deepcopy(single_result_list))
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
    return array_size_update_list


def unrolling_scheme_clean(unrolling_bundle, unrolling_scheme_list, layer_spec_raw):
    ll = {1: 'FX', 2: 'FY', 3: 'OX', 4: 'OY', 5: 'C', 6: 'K', 7: 'B'}
    unrolling_bundle_new = []
    unrolling_scheme_list_new = []
    unrolling_bundle_save = deepcopy(unrolling_bundle)

    # remove the size 1 dimension in every unrolling scheme.
    for idx1, unroll_scheme in enumerate(unrolling_scheme_list):
        for idx2, unroll in enumerate(unroll_scheme):
            if len(unroll) != 1:
                unroll_save = deepcopy(unroll)
                for idx3, elem in enumerate(unroll_save):
                    if layer_spec_raw[ll[elem]] == 1:
                        unroll.remove(elem)
                        unrolling_bundle[idx1][idx2].remove(unrolling_bundle_save[idx1][idx2][idx3])

    # remove repetitive unrolling scheme
    for idx, unroll_scheme in enumerate(unrolling_bundle):
        if unroll_scheme not in unrolling_bundle_new:
            unrolling_bundle_new.append(unroll_scheme)
            unrolling_scheme_list_new.append(unrolling_scheme_list[idx])

    return unrolling_bundle_new, unrolling_scheme_list_new


class LayerRound(object):

    def __init__(self, layer_spec_raw, array_size, unrolling_scheme_list, unrolling_size_list, spatial_utilization_TH):

        # step 0: combine unrolling_scheme_list and unrolling_size_list so as to process them in a bundle
        unrolling_bundle = []
        for idx1, unroll_scheme in enumerate(unrolling_scheme_list):
            unrolling_bundle.append([])
            for idx2, unroll in enumerate(unroll_scheme):
                unrolling_bundle[-1].append(list(zip(unroll, unrolling_size_list[idx1][idx2])))

        # step 1: remove the layer dimension which is 1 in the unrolling_scheme_list
        unrolling_bundle, unrolling_scheme_list = unrolling_scheme_clean(unrolling_bundle, unrolling_scheme_list, layer_spec_raw)

        # step 2: represent array_size with replication
        array_size = array_size_extend(array_size, unrolling_scheme_list)

        # step 3: generate rounded layer info for each array scheme in each unrolling scheme.
        ll = {1: 'FX', 2: 'FY', 3: 'OX', 4: 'OY', 5: 'C', 6: 'K', 7: 'B'}
        round_layer_info = []
        footer_info = []
        fraction_spatial_unrolling = []
        ideal_spatial_unrolling = []
        aux_layer_to_su_hint_table = []
        for i in range(len(unrolling_scheme_list)):
            su_list_flatten = [item for sublist in unrolling_scheme_list[i] for item in sublist]
            unrolling_bundle_flatten = [item for sublist in unrolling_bundle[i] for item in sublist]
            for j in range(len(array_size[i])):
                array_size_single = array_size[i][j]
                array_size_single_flatten = [item for sublist in array_size_single for item in sublist]

                # If user has defined the unrolled size (not necessary), check whether it is met.
                # Only pass the met one (flag = True) to the next stage.
                for su_dim, su_type in enumerate(su_list_flatten):
                    flag = True
                    if (unrolling_bundle_flatten[su_dim][1] is None) or \
                            (unrolling_bundle_flatten[su_dim][1] == array_size_single_flatten[su_dim]):
                        continue
                    else:
                        flag = False
                        break

                if flag:
                    aux_layer_to_su_hint_table.append(i)
                    round_layer_info.append(deepcopy(layer_spec_raw))
                    footer_info.append({'B': 0, 'K': 0, 'C': 0, 'OY': 0, 'OX': 0, 'FY': 0, 'FX': 0})
                    fraction_spatial_unrolling.append([])
                    ideal_spatial_unrolling.append([])
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

        # step 4: remove repetitive unrollings and unrollings that cannot meet PE array spatial utilization TH
        round_layer_info0 = []
        footer_info0 = []
        fraction_spatial_unrolling0 = []
        ideal_spatial_unrolling0 = []
        aux_layer_to_su_hint_table0 = []
        spatial_utilization_list = []
        greedy_mapping_flag = []
        for idx1, su in enumerate(ideal_spatial_unrolling):
            spatial_utilization = (layer_spec_raw['B']/round_layer_info[idx1]['B']) * \
                                  (layer_spec_raw['K']/round_layer_info[idx1]['K']) * \
                                  (layer_spec_raw['C']/round_layer_info[idx1]['C']) * \
                                  (layer_spec_raw['OY']/round_layer_info[idx1]['OY']) * \
                                  (layer_spec_raw['OX']/round_layer_info[idx1]['OX']) * \
                                  (layer_spec_raw['FY']/round_layer_info[idx1]['FY']) * \
                                  (layer_spec_raw['FX']/round_layer_info[idx1]['FX'])
            if spatial_utilization < spatial_utilization_TH:
                pass
            elif su not in ideal_spatial_unrolling0:
                if spatial_utilization != 1:
                    greedy_mapping_flag.append(True)
                else:
                    greedy_mapping_flag.append(False)
                round_layer_info0.append(round_layer_info[idx1])
                footer_info0.append(footer_info[idx1])
                fraction_spatial_unrolling0.append(fraction_spatial_unrolling[idx1])
                ideal_spatial_unrolling0.append(su)
                aux_layer_to_su_hint_table0.append(aux_layer_to_su_hint_table[idx1])
                spatial_utilization_list.append(spatial_utilization)

        self.round_layer_info = round_layer_info0
        self.footer_info = footer_info0
        self.fraction_su = fraction_spatial_unrolling0
        self.ideal_su = ideal_spatial_unrolling0
        self.aux_layer_to_su_hint_table = aux_layer_to_su_hint_table0
        self.spatial_utilization_list = spatial_utilization_list
        self.greedy_mapping_flag = greedy_mapping_flag

    @classmethod
    def layer_rounding(cls, layer_spec_raw, array_size, unrolling_scheme_list, spatial_utilization_TH):
        return cls(layer_spec_raw, array_size, unrolling_scheme_list, spatial_utilization_TH)


def generate_unrolling_scheme_list(non_one_dim, symmetrical_array):
    l = non_one_dim
    unrolling_scheme_list = []
    for i in range(1, len(l)):
        comb1 = combinations(l, i)
        for D1 in comb1:
            l_change = deepcopy(l)
            for d1 in D1:
                l_change.remove(d1)
            for j in range(1, len(l_change) + 1):
                comb2 = combinations(l_change, j)
                for D2 in comb2:
                    if symmetrical_array:
                        if [list(D2), list(D1)] not in unrolling_scheme_list:
                            unrolling_scheme_list.append([list(D1), list(D2)])
                    else:
                        unrolling_scheme_list.append([list(D1), list(D2)])

    return unrolling_scheme_list


class LayerRound2(object):

    """ LayerRound2 doesn't require spatial unrolling hint from user (LayerRound does) """

    def __init__(self, layer_spec_raw, array_size, spatial_utilization_TH):

        # step 0: generate unrolling_scheme_list
        symmetrical_array = (array_size[0] == array_size[1])
        non_one_dim = []
        for idx, dim in enumerate(['FX', 'FY', 'OX', 'OY', 'C', 'K', 'B']):
            if layer_spec_raw[dim] != 1:
                non_one_dim.append(idx+1)
        unrolling_scheme_list = generate_unrolling_scheme_list(non_one_dim, symmetrical_array)

        # step 1: represent array_size with replication
        array_size = array_size_extend(array_size, unrolling_scheme_list)

        # step 2: generate rounded layer info for each array scheme in each unrolling scheme.
        ll = {1: 'FX', 2: 'FY', 3: 'OX', 4: 'OY', 5: 'C', 6: 'K', 7: 'B'}
        round_layer_info = []
        footer_info = []
        fraction_spatial_unrolling = []
        ideal_spatial_unrolling = []
        aux_layer_to_su_hint_table = []
        for i in range(len(unrolling_scheme_list)):
            su_list_flatten = [item for sublist in unrolling_scheme_list[i] for item in sublist]
            for j in range(len(array_size[i])):
                array_size_single = array_size[i][j]
                array_size_single_flatten = [item for sublist in array_size_single for item in sublist]

                aux_layer_to_su_hint_table.append(i)
                round_layer_info.append(deepcopy(layer_spec_raw))
                footer_info.append({'B': 0, 'K': 0, 'C': 0, 'OY': 0, 'OX': 0, 'FY': 0, 'FX': 0})
                fraction_spatial_unrolling.append([])
                ideal_spatial_unrolling.append([])
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

        # step 3: remove repetitive unrollings and unrollings that cannot meet PE array spatial utilization TH
        round_layer_info0 = []
        footer_info0 = []
        fraction_spatial_unrolling0 = []
        ideal_spatial_unrolling0 = []
        aux_layer_to_su_hint_table0 = []
        spatial_utilization_list = []
        greedy_mapping_flag = []
        for idx1, su in enumerate(ideal_spatial_unrolling):
            spatial_utilization = (layer_spec_raw['B']/round_layer_info[idx1]['B']) * \
                                  (layer_spec_raw['K']/round_layer_info[idx1]['K']) * \
                                  (layer_spec_raw['C']/round_layer_info[idx1]['C']) * \
                                  (layer_spec_raw['OY']/round_layer_info[idx1]['OY']) * \
                                  (layer_spec_raw['OX']/round_layer_info[idx1]['OX']) * \
                                  (layer_spec_raw['FY']/round_layer_info[idx1]['FY']) * \
                                  (layer_spec_raw['FX']/round_layer_info[idx1]['FX'])
            if spatial_utilization < spatial_utilization_TH:
                pass
            elif su not in ideal_spatial_unrolling0:
                if spatial_utilization != 1:
                    greedy_mapping_flag.append(True)
                else:
                    greedy_mapping_flag.append(False)
                round_layer_info0.append(round_layer_info[idx1])
                footer_info0.append(footer_info[idx1])
                fraction_spatial_unrolling0.append(fraction_spatial_unrolling[idx1])
                ideal_spatial_unrolling0.append(su)
                aux_layer_to_su_hint_table0.append(aux_layer_to_su_hint_table[idx1])
                spatial_utilization_list.append(spatial_utilization)\

        self.unrolling_scheme_list = unrolling_scheme_list
        self.round_layer_info = round_layer_info0
        self.footer_info = footer_info0
        self.fraction_su = fraction_spatial_unrolling0
        self.ideal_su = ideal_spatial_unrolling0
        self.aux_layer_to_su_hint_table = aux_layer_to_su_hint_table0
        self.spatial_utilization_list = spatial_utilization_list
        self.greedy_mapping_flag = greedy_mapping_flag

    @classmethod
    def layer_rounding(cls, layer_spec_raw, array_size, unrolling_scheme_list, spatial_utilization_TH):
        return cls(layer_spec_raw, array_size, unrolling_scheme_list, spatial_utilization_TH)


def mem_access_count_correct(loop_fractional, loop):
    """
    This function penalize the memory which feed data into MAC array under greedy mapping
    when there is no energy benefit from memory BW under-utilization.
    """
    for op in ['W', 'I', 'O']:
        for level, mem_count in enumerate(loop.req_mem_count[op]):
            if mem_count == 1 and op == 'W':
                loop_fractional.mem_access_elem['W'][level][0] = loop.mem_access_elem['W'][level][0]
                break
            if mem_count == 1 and op == 'I':
                loop_fractional.mem_access_elem['I'][level][0] = loop.mem_access_elem['I'][level][0]
                loop_fractional.mem_access_elem['I_base'][level][0] = loop.mem_access_elem['I_base'][level][0]
                break
            if mem_count == 1 and op == 'O':
                loop_fractional.mem_access_elem['O'][level][0] = loop.mem_access_elem['O'][level][0]
                loop_fractional.mem_access_elem['O_final'][level][0] = loop.mem_access_elem['O_final'][level][0]
                loop_fractional.mem_access_elem['O_partial'][level][0] = loop.mem_access_elem['O_partial'][level][0]
                break
    return loop_fractional




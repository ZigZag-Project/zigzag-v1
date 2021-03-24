import itertools
import operator
from copy import deepcopy

import numpy as np
from sympy.ntheory import factorint

import classes as cls
import cost_model_funcs as cmf
import msg
import output_funcs as of
from classes.order import Order
from loma_utils import permutations

loop_types_list = ["FX", "FY", "OX", "OY", "C", "K", "B"]
# Corresponding number for each loop_type {"FX": 1, "FY": 2, "OX": 3, "OY": 4, "C": 5, "K": 6, "B": 7}
loop_type_to_ids = {key: value + 1 for value, key in enumerate(loop_types_list)}
# Corresponding number for each loop_type: {1: "FX", 2: "FY", 3: "OX", 4: "OY", 5: "C", 6: "K", 7: "B"}
ids_to_loop_type = {value: key for key, value in loop_type_to_ids.items()}
operand_cost_types = ['W', 'I', 'O']
operand_cost_template = {key: [] for key in operand_cost_types}


def get_cost_model_output(allocated_order, input_settings, mem_scheme, layer_comb, spatial_loop_comb, ii_su=0):
    """
    Return the cost model output of an ordering.

    Arguments
    =========
    allocated_order:  Allocated order
    input_settings: The input settings
    mem_scheme:     The memory scheme
    layer_comb:          The Layer class for the evaluated layer
    spatial_loop_comb:
    ii_su:
    """
    ii = 0
    schedule_info = 0  # not used right now
    [layer_origin, layer_rounded] = layer_comb
    [spatial_loop, spatial_loop_fractional] = spatial_loop_comb
    try:
        greedy_mapping_flag = mem_scheme.greedy_mapping_flag[ii_su]
        footer_info = mem_scheme.footer_info[ii_su]
    except IndexError:
        greedy_mapping_flag = False
        footer_info = None
    # memory allocation part
    temporal_loop, loop = create_loop_objects(layer_rounded, allocated_order, spatial_loop, input_settings)
    loop_fractional = perform_greedy_mapping(layer_origin, allocated_order, spatial_loop_fractional, loop,
                                             input_settings)
    # utilization part
    utilization = get_utilization(layer_rounded, temporal_loop, spatial_loop_comb, loop, input_settings, mem_scheme)
    active_mac_cost = cmf.get_active_mac_cost(layer_origin, input_settings.mac_array_info['single_mac_energy'])
    idle_mac_cost = cmf.get_idle_mac_cost(layer_origin, layer_rounded, input_settings.mac_array_info['array_size'],
                                          input_settings.mac_array_info['idle_mac_energy'],
                                          mem_scheme.spatial_unrolling)
    total_cost_layer = find_total_cost_layer(allocated_order, loop_fractional, utilization, active_mac_cost,
                                             idle_mac_cost[ii_su], mem_scheme, input_settings, schedule_info, ii)
    # TODO MAC area (multiplier and adder) is not included.
    # occupied_area format: [total_area, active_area]
    occupied_area = msg.get_mem_scheme_area2(mem_scheme, spatial_loop.unit_count, utilization.mac_utilize_spatial)
    cost_model_output = get_cost_model(total_cost_layer, active_mac_cost, idle_mac_cost, ii_su, temporal_loop,
                                       mem_scheme, loop_fractional, spatial_loop, greedy_mapping_flag, footer_info,
                                       occupied_area, utilization, ii)
    return cost_model_output


def get_cost_model(total_cost_layer, active_mac_cost, idle_mac_cost, ii_su, temporal_loop, mem_scheme,
                   loop_fractional, spatial_loop, greedy_mapping_flag, footer_info, occupied_area, utilization, ii):
    # Get CostModelOutput
    operand_cost = deepcopy(operand_cost_template)
    cost_model_output = of.CostModelOutput(total_cost_layer, deepcopy(operand_cost),
                                           (active_mac_cost, idle_mac_cost[ii_su]),
                                           deepcopy(temporal_loop.temporal_loop),
                                           deepcopy(mem_scheme.spatial_unrolling[ii_su]),
                                           mem_scheme.flooring[ii_su],
                                           deepcopy(loop_fractional), deepcopy(spatial_loop),
                                           greedy_mapping_flag, footer_info,
                                           deepcopy(temporal_loop), occupied_area,
                                           utilization, ii)
    return cost_model_output


def combine_orderings(ordering_1, ordering_2):
    """
    Function to combine two orderings.
    Example 1:
    ordering_1 = ((7,2), 'X')
    ordering_2 = ((6,5),)
    combined_ordering = ((7,2),(6,5))

    Example 2:
    ordering_1 = ((7,2), 'X', 'X')
    ordering_2 = ((6,5), 'X')
    combined_ordering = ((7,2),(6,5), 'X')

    Example 3:
    ordering_1 = ('X', (7,2), 'X')
    ordering_2 = ((6,5), 'X')
    combined_ordering = ((6,5),(7,2), 'X')
    """

    if ordering_1 is None:
        return ordering_2

    if ordering_2 is None:
        return ordering_1

    idx_2 = 0
    combined_ordering = []
    for idx_1, elem in enumerate(ordering_1):
        if elem == "X":
            combined_ordering.append(ordering_2[idx_2])
            idx_2 += 1
        else:
            combined_ordering.append(ordering_1[idx_1])
    return combined_ordering


def combine_orderings_list(orderings_list):
    combined_ordering = deepcopy(orderings_list)[-1]
    for ordering in range(len(orderings_list) - 2, -1, -1):
        combined_ordering = combine_orderings(combined_ordering, orderings_list[ordering])
    return combined_ordering


def create_multiset_format(n, rs, pfs):
    """
    Function that converts the given n, rs and pfs into a multiset format,
            ex.:  [('X', 'X', 'X', 'X', 'X', 'X', 'X', (1, 3))]

    Arguments
    =========
    n: Total length of the permutations
    rs: How many of each non-X elements should be present within each permutation
    pfs: The value of each non-X prime factor to order

    Example:
    n = 6, rs = (1,3), pfs = (3,5)
    multiset = [3,5,5,5,X,X]
    The set is represented as a list here, because the permutation generatior expects this.
    """

    multiset = []

    # Extend the multiset with given r's
    for i, r in enumerate(rs):
        multiset.extend([str(pfs[i])] * r)

    # If sum(rs) < n we append with 'X' until len(multiset) == n
    if sum(rs) < n:
        diff = int(n - sum(rs))
        multiset.extend(["X"] * diff)
    if sum(rs) > n:
        raise NotImplementedError

    return multiset


def get_permutations(multiset: list, loop_type: dict):
    """
    Function that generates all unique non-merged permutations of a multiset.

    Arguments
    =========
    multiset: A list of elements X and non-X.
    loop_type: A loop type {"FX": 1, "FY": 2, "OX": 3, "OY": 4, "C": 5, "K": 6, "B": 7}

    Returns
    =======
    permutations_list: A list containing all the permutations of the multiset with the loop_type_number added to each
                       non-X element, ex.:  [('X', 'X', 'X', 'X', 'X', 'X', 'X', (1, 3))]
    n_permutations: permutation number
    """

    # Get the loop_type number for the given loop_type
    loop_type_number = loop_type_to_ids[loop_type]

    permutations_list = []
    n_permutations = 0
    for permutation in permutations(multiset):
        p = []
        for dimension in permutation:
            if dimension == "X":
                p.append("X")
            else:
                p.append((loop_type_number, int(dimension)))
        permutations_list.append(tuple(p))
        n_permutations += 1

    return permutations_list, n_permutations


def limit_loop_prime_factors(layer_spec_pf, layer_spec_pf_count, layer_spec_pf_count_sum, loop_prime_factor_limit):
    """

    Function to limit the total number of loop prime factors.
    This function scans the loop prime factors and while the number of loop prime factors is greater than the lpf_limit it:
    - picks the loop type that has the most loop prime factors
    - merges the smallest two loop prime factors of that loop type (multiplying their values)

    Parameters
    ----------
    layer_spec_pf:  contains for each loop_type the different prime factors
    layer_spec_pf_count: contains for each loop_type the different multiplicities of the prime factors
    layer_spec_pf_count_sum: the total amount of LPFs across all loop_types
    loop_prime_factor_limit: limit of the total number of the loop prime factors

    Returns
    -------

    """
    prime_factor_number = sum(layer_spec_pf_count_sum.values())

    if prime_factor_number <= loop_prime_factor_limit:
        # print("No prime factor limiting performed, n_pf =", n_pf)
        return layer_spec_pf, layer_spec_pf_count, prime_factor_number

    while prime_factor_number > loop_prime_factor_limit:
        max_loop_type = max(layer_spec_pf_count_sum.items(), key=operator.itemgetter(1))[
            0]  # a loop type with highest number of pfs
        max_loop_primary_factors = list(layer_spec_pf[max_loop_type])  # pfs for max_loop_type
        max_counts = list(layer_spec_pf_count[max_loop_type])  # counts for different pfs in max_loop_type

        if max_counts[0] == 1:  # multiplicity of smallest pf is 1
            new_factor = max_loop_primary_factors[0] * max_loop_primary_factors[1]
            max_counts[0] -= 1
            max_counts[1] -= 1
        else:  # multiplicity of smallest pf > 1
            new_factor = max_loop_primary_factors[0] * max_loop_primary_factors[0]
            max_counts[0] -= 2

        if new_factor in max_loop_primary_factors:  # possible if not first iteration of while loop
            new_factor_idx = max_loop_primary_factors.index(new_factor)
            max_counts[new_factor_idx] += 1
        else:
            new_factor_idx = len([pf for pf in max_loop_primary_factors if pf < new_factor])
            max_loop_primary_factors.insert(new_factor_idx, new_factor)
            max_counts.insert(new_factor_idx, 1)  # count of new factor is 1

        # Sanitize max_loop_primary_factors and max_counts to remove all elements with multiplicity 0
        non_zero_idxs = [idx for idx in range(len(max_counts)) if max_counts[idx] != 0]
        max_loop_primary_factors = [max_loop_primary_factors[non_zero_idx] for non_zero_idx in non_zero_idxs]
        max_counts = [max_counts[non_zero_idx] for non_zero_idx in non_zero_idxs]

        # Update the layer_spec_pf, layer_spec_pf_count and layer_spec_pf_count_sum with updated factors/counts
        layer_spec_pf[max_loop_type] = tuple(max_loop_primary_factors)
        layer_spec_pf_count[max_loop_type] = tuple(max_counts)
        layer_spec_pf_count_sum[max_loop_type] -= 1

        # Decrease the total number of pfs (actually not all prime anymore) by 1
        prime_factor_number -= 1

    # print("Prime factor limit performed, n_pf:", n_pf_start, "-->", n_pf)
    return layer_spec_pf, layer_spec_pf_count, prime_factor_number


def get_prime_factors(layer_spec: dict, loop_prime_factor_limit: int):
    """

    Parameters
    ----------
    layer_spec: all specifications of the layer
    loop_prime_factor_limit: limit of the total number of the loop prime factors

    Returns
    -------
    layer_spec_pf:  contains for each loop_type the different prime factors
    layer_spec_pf_count: contains for each loop_type the different multiplicities of the prime factors
    total_lpf_count: gives the total amount of LPFs across all loop_types
    """
    layer_spec_pf = {}
    layer_spec_pf_count = {}
    layer_spec_pf_count_sum = {}
    for loop_type, loop_dimension in layer_spec.items():
        if loop_dimension == 0 or loop_dimension == 1:
            continue
        factors = factorint(loop_dimension)
        primary_factors = []
        counts = []
        for primary_factor, count in factors.items():
            primary_factors.append(primary_factor)
            counts.append(count)
        layer_spec_pf[loop_type] = tuple(primary_factors)
        layer_spec_pf_count[loop_type] = tuple(counts)
        layer_spec_pf_count_sum[loop_type] = sum(counts)
    total_lpf_count = sum(layer_spec_pf_count_sum.values())
    layer_spec_pf, layer_spec_pf_count, total_lpf_count = limit_loop_prime_factors(layer_spec_pf, layer_spec_pf_count,
                                                                                   layer_spec_pf_count_sum,
                                                                                   loop_prime_factor_limit)
    return layer_spec_pf, layer_spec_pf_count, total_lpf_count


def get_smallest_prime_factor(temporal_loop):
    """
    Check the temporal mapping ordering and return the smallest prime factor.

    Parameters
    ----------
    temporal_loop

    """
    if temporal_loop is None:
        return None

    smallest_prime_factor = float("inf")
    for loop_prime_factor in temporal_loop:
        if loop_prime_factor == "X":
            continue
        else:
            prime_factor = loop_prime_factor[1]
            if prime_factor < smallest_prime_factor:
                smallest_prime_factor = prime_factor
    assert smallest_prime_factor != float("inf"), "Only Xs are present within the given loop type ordering"
    return smallest_prime_factor


def get_non_trivial_loop_types_spec(layer_spec: dict) -> dict:
    """
    Add all non-trivial loop_types of layer_spec to layer_spec_temporal

    Parameters
    ----------
    layer_spec: all specifications of the layer

    Returns
    -------

    """
    layer_spec_temporal = {}
    for loop_type, loop_factor in layer_spec.items():
        if (loop_factor != 0 or loop_factor != 1) and (loop_type in loop_types_list):
            layer_spec_temporal[loop_type] = loop_factor
    return layer_spec_temporal


def update_temporal_layer_spec(spatial_unrolling: dict, layer_spec_temporal: dict) -> dict:
    """
    Update the temporal layer spec to remove the already spatially unrolled dimensions.

    Parameters
    ----------
    spatial_unrolling: spatial_mapping_fixed (mapping.yaml) in the dict format with the loop type names as ids
    layer_spec_temporal: non-trivial loop_types of the layer specifications

    Returns
    -------

    """
    for level in range(0, len(spatial_unrolling["W"])):
        for [loop_type_number, su_factor] in spatial_unrolling["W"][level]:
            loop_type = ids_to_loop_type[loop_type_number]
            try:
                primary_factor = layer_spec_temporal[loop_type]
            except:
                continue
            quotient, remainder = divmod(primary_factor, su_factor)
            assert remainder == 0  # pf/su_factor should have remainder 0
            layer_spec_temporal[loop_type] = quotient
    return layer_spec_temporal


def get_all_orderings_list(layer_spec_pf: dict, layer_spec_pf_count: dict, total_lpf_count: int):
    """
    Get the list of all orderings for each loop type

    Parameters
    ----------
    layer_spec_pf:  contains for each loop_type the different prime factors
    layer_spec_pf_count: contains for each loop_type the different multiplicities of the prime factors
    total_lpf_count:  the total amount of LPFs across all loop_types

    Returns
    -------
    count_dict: a dict with the number of permutations (size) per each layer, ex.: {'FX': 8, 'OX': 7, 'C': 60, 'K': 3}
    loop_type_order_list: a loop order list, ex.: ['FX', 'OX', 'C', 'K']
    tl_dict: a dict of the possible orderings of the lpfs and placeholders, ex.:
            {'FX': [('X', 'X', 'X', 'X', 'X', 'X', 'X', (1, 3)).. }
    total_count: total number of permutations (total size)
    """
    lpf_processed_count = 0
    loop_type_order_list = []
    count_dict = {}
    total_count = 1
    tl_dict = {}
    for loop_type in list(reversed(loop_types_list)):  # always loop in fixed order
        try:
            prime_factors = layer_spec_pf[loop_type]
        except:
            continue  # if this loop_type is not in layer_spec_pf, move to next loop_type
        primary_factors_count = layer_spec_pf_count[loop_type]
        permutations_length = total_lpf_count - lpf_processed_count
        multiset = create_multiset_format(permutations_length, primary_factors_count, prime_factors)
        # Increment the lpf_processed_count by the number of LPFs processed in this step
        lpf_processed_count += sum(primary_factors_count)
        # Get the different possible orderings for the created multiset for this loop_type
        permutations_list, count = get_permutations(multiset, loop_type)
        loop_type_order_list.append(loop_type)
        count_dict[loop_type] = count
        total_count *= count
        # Add all the found permutations for this loop_type to tl_dict
        tl_dict[loop_type] = permutations_list
    return count_dict, loop_type_order_list, tl_dict, total_count


def merge_loops(order, smallest_primary_factors):
    """
    The function for merging.

    Parameters
    ----------
    order
    smallest_primary_factors

    Returns
    -------

    """
    merged_order = []
    (previous_lt_number, previous_factors) = order[0]
    first_of_this_lt = True
    for (loop_type_number, factor) in order[1:]:
        if loop_type_number == previous_lt_number:
            if first_of_this_lt and previous_factors == smallest_primary_factors[previous_lt_number]:
                merged_order.append((previous_lt_number, previous_factors))
                previous_factors = factor
                first_of_this_lt = False
            else:
                previous_factors *= factor
        elif previous_factors != 1:
            merged_order.append((previous_lt_number, previous_factors))
            previous_lt_number = loop_type_number
            previous_factors = factor
            first_of_this_lt = True
    merged_order.append((previous_lt_number, previous_factors))
    return tuple(merged_order)


def fulfill_temporal_loop_dict(temporal_loop_to_orderings):
    """
    Get the orderings for all possible loop_types (some might be non-existent)
    If loop type doesn't exist in the dict, add it with None value

    Parameters
    ----------
    temporal_loop_to_orderings: a temporal loop dict of the possible orderings of the lpfs and placeholders
                                ex.: {"K": ((6, 64), 'X', (1, 2)), ...}

    """
    temporal_loop_to_none = {key: [None] for key in loop_types_list}
    temporal_loop_to_orderings = {**temporal_loop_to_none, **temporal_loop_to_orderings}
    return temporal_loop_to_orderings


def generate_the_smallest_pfs_dict(temporal_loop_to_orderings):
    """
    Get the smallest prime factor for each loop type (required for loop merging)

    Parameters
    ----------
    temporal_loop_to_orderings: a temporal loop dict of the possible orderings of the lpfs and placeholders

    Returns
    -------

    """
    smallest_pfs = {}
    for key, value in sorted(ids_to_loop_type.items(), key=operator.itemgetter(0), reverse=True):
        # print(value, temporal_loop_to_orderings[value])
        smallest_pfs[key] = get_smallest_prime_factor(temporal_loop_to_orderings[value][0])
    return smallest_pfs


def get_temporal_loop_merged_ordering(temporal_loop_orderings_list, smallest_pfs):
    """
    Creates merged temporal loop orderings
    Parameters
    ----------
    temporal_loop_orderings_list: a list of orderings with placeholders
    smallest_pfs: smallest primary factors

    Returns
    -------
    merged_order: list of tuples: ((list_type_id, value), ..)
                  ex.: ((3, 13), (6, 4), (5, 32), (6, 4), (5, 4), (1, 3), (6, 12))
    """
    temporal_loop_orderings_list = list(temporal_loop_orderings_list)
    # Final order with all X's filled in
    nonmerged_order = combine_orderings_list(temporal_loop_orderings_list)
    # Merge loops of same type
    merged_order = merge_loops(nonmerged_order, smallest_pfs)
    return merged_order


class HashAlreadyExists(Exception):
    pass


def is_order_already_proceed(merged_order, merged_set, skipped):
    hashed = hash(merged_order)
    if hashed in merged_set:
        skipped += 1
        raise HashAlreadyExists()
    else:
        merged_set.add(hashed)


def allocate_memory_for_tl_order(merged_order, spatial_loop, layer_origin, input_settings, nodes):
    """
    Memory allocation object.

    Parameters
    ----------
    merged_order: temporal mapping ordering list, tuple: [(list_type_id, value), ...]
    spatial_loop: mem_scheme.spatial_unrolling SpatialLoop objects
    layer_origin: the original 3D/7D layer
    input_settings: InputSettings object
    nodes: mem_scheme.nodes

    Returns
    -------

    """
    # Get the different MemoryNodes we need to allocate
    n_mem_levels = len(nodes)
    # Initialize Order object
    order = Order(merged_order, spatial_loop, layer_origin, input_settings, n_mem_levels)
    # Loop through all the nodes in each level to allocate the LPFs to the memories
    for level in range(n_mem_levels):
        if level == n_mem_levels - 1:
            # If the level is the last level in the hierarchy, allocate all remaning LPFs.
            allocated_order = order.allocate_remaining()
            break
        for node in nodes[level]:
            order.allocate_memory(node, level)

    # print(merged_order)
    # print('W\t', allocated_order['W'])
    # print('I\t', allocated_order['I'])
    # print('O\t', allocated_order['O'])

    # if merged_order == ((5, 2), (5, 288), (4, 7), (6, 6)):
    #     print(allocated_order['I'])
    return allocated_order


def create_loop_objects(layer_rounded, allocated_order, spatial_loop, input_settings):
    """

    Parameters
    ----------
    layer_rounded: rounded 3D/7D layer
    allocated_order: an Order object with the allocated memory
    spatial_loop: mem_scheme.spatial_unrolling SpatialLoop objects
    input_settings: InputSettings object

    """
    # temporal_loop = TemporalLoopLight(layer_rounded, allocated_order, spatial_loop, order.loop_cycles, order.irrelevant_loop)
    # loop = LoopLight(layer_rounded, temporal_loop, spatial_loop, input_settings.precision,
    #                 input_settings.fixed_temporal_mapping)
    temporal_loop = cls.TemporalLoop.extract_loop_info(
        layer_rounded, allocated_order, spatial_loop
    )
    loop = cls.Loop.extract_loop_info(
        layer_rounded,
        temporal_loop,
        spatial_loop,
        input_settings.precision,
        input_settings.fixed_temporal_mapping,
    )
    return temporal_loop, loop


def perform_greedy_mapping(layer_origin, allocated_order, spatial_loop_fractional, loop, input_settings):
    """
    Greedy mapping: loop_fractional required

    Parameters
    ----------
    layer_origin: the original 3D/7D layer
    allocated_order: an Order object with the allocated memory
    spatial_loop_fractional: mem_scheme.fraction_spatial_unrolling SpatialLoop objects
    loop: fixed temporal looping Loop object
    input_settings: InputSettings object

    Returns
    -------

    """
    if input_settings.spatial_unrolling_mode in [4, 5]:
        ############# Advanced User Configuration #############
        # mem_energy_saving_when_BW_under_utilized = True
        #######################################################
        temporal_loop_fractional = cls.TemporalLoop.extract_loop_info(
            layer_origin, allocated_order, spatial_loop_fractional
        )
        loop_fractional = cls.Loop.extract_loop_info(
            layer_origin,
            temporal_loop_fractional,
            spatial_loop_fractional,
            input_settings.precision,
            input_settings.fixed_temporal_mapping,
        )
        # if mem_energy_saving_when_BW_under_utilized is False:
        #     loop_fractional = mem_access_count_correct(loop_fractional, loop)

    else:
        loop_fractional = loop
    return loop_fractional


def find_total_cost_layer(allocated_order, loop_fractional, utilization, active_mac_cost, idle_mac_cost, mem_scheme,
                          input_settings, schedule_info=0, ii=False):
    """
    Return total energy cost for all layers.
    """
    operand_cost = deepcopy(operand_cost_template)
    total_cost_layer = 0
    for operand in operand_cost_types:
        for level in range(0, len(allocated_order[operand])):
            operand_cost[operand].append(
                cmf.get_operand_level_energy_cost(operand, level, mem_scheme.mem_cost,
                                                  input_settings.mac_array_info,
                                                  schedule_info, loop_fractional,
                                                  mem_scheme.mem_fifo, mem_scheme, input_settings.precision,
                                                  utilization, ii))
            # TODO
            # loop.array_wire_distance[operand].append(
            #     cmf.get_operand_level_wire_distance(operand, level,
            #                                         schedule_info,
            #                                         input_settings.mac_array_info, loop,
            #                                         msc.mem_fifo))
        total_cost_layer += np.sum(operand_cost[operand])

    total_cost_layer += active_mac_cost + idle_mac_cost
    return total_cost_layer


def get_utilization(layer_rounded, temporal_loop, spatial_loop_comb, loop, input_settings, mem_scheme):
    return cls.Utilization.get_utilization(
        layer_rounded,
        temporal_loop,
        spatial_loop_comb,
        loop,
        input_settings.mac_array_info,
        mem_scheme.mem_size,
        mem_scheme.mem_share,
        mem_scheme.mem_type,
        input_settings.mac_array_stall,
        input_settings.precision,
        mem_scheme.mem_bw,
    )


def compare_total_results(allocated_order, utilization, total_cost_layer, min_en, min_en_ut, min_en_order, max_ut,
                          max_ut_en, max_ut_order):
    en = total_cost_layer
    ut = utilization.mac_utilize_no_load
    if (en < min_en) or (en == min_en and ut > min_en_ut):
        min_en = en
        min_en_ut = ut
        min_en_order = allocated_order
    if (ut > max_ut) or (ut == max_ut and en < max_ut_en):
        max_ut = ut
        max_ut_en = en
        max_ut_order = allocated_order
    return min_en, min_en_ut, min_en_order, max_ut, max_ut_en, max_ut_order


def collect_memory(utilization, total_cost_layer, save_all_tm, energy_collect, utilization_collect, latency_collect):
    """

    If input_settings.tm_search_result_saving is True, save all results.

    save_all_tm: input_settings.tm_search_result_saving
    """
    if save_all_tm:
        energy_collect.append(int(total_cost_layer))
        utilization_collect.append(utilization.mac_utilize_no_load)
        latency_collect.append(utilization.latency_no_load)
    return energy_collect, utilization_collect, latency_collect


# def generate_tm_orderings(layer_spec: dict, spatial_unrolling: dict, lpf_limit: int):
def og(layer_spec: dict, spatial_unrolling: dict, lpf_limit: int):
    """
    Description
    ----------
    This function generates the truly exhaustive temporal mapping orderings up to a lpf_limit.
    The orderings are generated through multisets for each loop type individually using placeholders (X).
    The returned tl_dict will contain, for each loop type, the possible orderings of the lpfs and placeholders.

    Parameters
    ----------
    layer_spec: all specifications of the layer, ex.:
                {'B': 1, 'K': 384, 'C': 256, 'OY': 13, 'OX': 13, 'FY': 3, 'FX': 3,
                 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 0, 'PX': 0, 'G': 1}
    spatial_unrolling: spatial_mapping_fixed (mapping.yaml) in the dict format with the loop type names as ids
                {'W': [[], [(4, 13), (2, 3), (5, 2), (6, 2)], []], 'I': [[], [(4, 13), (2, 3), (5, 2), (6, 2)], [], []],
                'O': [[], [(4, 13), (2, 3), (5, 2), (6, 2)], [], []]}
    lpf_limit: limit of the total number of the loop prime factors

    Returns
    -------
    count_dict: a dict with the number of permutations (size) per each layer, ex.: {'FX': 8, 'OX': 7, 'C': 60, 'K': 3}
    loop_type_order_list: a loop order list, ex.: ['FX', 'OX', 'C', 'K']
    tl_dict: a dict of the possible orderings of the lpfs and placeholders, ex.:
            {'FX': [('X', 'X', 'X', 'X', 'X', 'X', 'X', (1, 3)).. }
    total_count: total number of permutations (total size)

    Examples
    -------

    B=6 (prime factorization: 3x2), K=2 (prime factorization: 2):
    tl_dict =
    {
    B:
        (
        ((B,3),(B,2),X),((B,3),X,(B,2)),(X,(B,3),(B,2)),
        ((B,2),(B,3),X),((B,2),X,(B,3)),(X,(B,2),(B,3))
        ),
    K:
        (
        ((K,2))
        )
    }
    """
    layer_spec_temporal = get_non_trivial_loop_types_spec(layer_spec)
    # Update the temporal layer spec to remove the already spatially unrolled dimensions
    layer_spec_temporal = update_temporal_layer_spec(spatial_unrolling, layer_spec_temporal)
    # Get all prime factorizations for the loop_types in loop_spec_temporal
    layer_spec_pf, layer_spec_pf_count, total_lpf_count = get_prime_factors(layer_spec_temporal, lpf_limit)
    # Get the list of all orderings for each loop type
    count_dict, loop_type_order_list, tl_dict, total_count = get_all_orderings_list(layer_spec_pf, layer_spec_pf_count,
                                                                                    total_lpf_count)
    return tl_dict, count_dict, loop_type_order_list, total_count


def tl_worker_new(temporal_loop_list, merged_count_dict, loop_type_order, total_merged_count, input_settings,
                  spatial_loop_comb, mem_scheme, precision, layer, mac_costs, ):
    """
    New tl_worker function to handle the multiset loop orderings.
    These orderings still require a memory allocation, followed by a cost model evaluation.

    Parameters
    ----------
    temporal_loop_list: a temporal loop list of the possible orderings of the lpfs and placeholders
    merged_count_dict: a dict with the number of permutations (size) per each layer, ex.: {'FX': 8, 'OX': 7, 'C': 60, 'K': 3}
    loop_type_order: a loop order list, ex.: ['FX', 'OX', 'C', 'K']
    total_merged_count: total number of permutations (total size)
    input_settings: InputSettings object
    spatial_loop_comb: a list of spatial_loop, spatial_loop_fractional SpatialLoop objects
    mem_scheme: MemorySchema object
    precision: input_settings.precision
    layer: a list of layer_origin(the original 3D/7D layer), layer_rounded(rounded 3D/7D layer), depending on im2col_enable
    mac_costs: [active_mac_cost, idle_mac_cost]

    Returns
    -------
    """
    # Get the active and idle MAC cost
    [active_mac_cost, idle_mac_cost] = mac_costs
    # Layer
    [layer_origin, layer_rounded] = layer
    # Spatial unrolling
    [spatial_loop, spatial_loop_fractional] = spatial_loop_comb
    # Adjust the merged_count_dict for the first loop_type B in the loop_type_order, as this got chunkedin caller function
    merged_count_dict[loop_type_order[0]] = len(temporal_loop_list[loop_type_order[0]])
    # Get the orderings for all possible loop_types (some might be non-existent)
    all_tl_to_orderings = fulfill_temporal_loop_dict(temporal_loop_list)
    temporal_loop_list = list(all_tl_to_orderings.values())
    # Get the smallest prime factor for each loop type (required for loop merging)
    smallest_pfs = generate_the_smallest_pfs_dict(all_tl_to_orderings)

    # Init minimal energy and max utilization results
    min_en = float("inf")
    min_en_ut = 0
    max_ut_en = float("inf")
    max_ut = 0
    min_en_order = {}
    max_ut_order = {}
    # Init energy,latency,utilization collect
    save_all_tm = input_settings.tm_search_result_saving
    energy_collect = []
    utilization_collect = []
    latency_collect = []
    # Loop through all the number of elements in each tl_list element
    ctr = 0
    skipped = 0
    merged_set = set()

    for temporal_loop_orderings_list in itertools.product(*temporal_loop_list):
        ctr += 1
        merged_order = get_temporal_loop_merged_ordering(temporal_loop_orderings_list, smallest_pfs)
        # Check if merged order was already processed
        try:
            is_order_already_proceed(merged_order, merged_set, skipped)
        except HashAlreadyExists:
            continue
        # memory allocation part
        allocated_order = allocate_memory_for_tl_order(merged_order, spatial_loop, layer_origin, input_settings,
                                                       mem_scheme.nodes)
        temporal_loop, loop = create_loop_objects(layer_rounded, allocated_order, spatial_loop, input_settings)
        loop_fractional = perform_greedy_mapping(layer_origin, allocated_order, spatial_loop_fractional, loop,
                                                 input_settings)
        # utilization part
        utilization = get_utilization(layer_rounded, temporal_loop, spatial_loop_comb, loop, input_settings, mem_scheme)
        total_cost_layer = find_total_cost_layer(allocated_order, loop_fractional, utilization, active_mac_cost,
                                                 idle_mac_cost, mem_scheme, input_settings)
        # result comparing part
        total_results = compare_total_results(allocated_order, utilization, total_cost_layer, min_en, min_en_ut,
                                              min_en_order, max_ut, max_ut_en, max_ut_order)
        min_en, min_en_ut, min_en_order, max_ut, max_ut_en, max_ut_order = total_results
        energy_collect, utilization_collect, latency_collect = collect_memory(utilization, total_cost_layer,
                                                                              save_all_tm, energy_collect,
                                                                              utilization_collect, latency_collect)
        # if ctr % 1000 == 0:
        #     print(ctr, "Execution time =", time.time()-t_start)
        #     t_start = time.time()
    return min_en, min_en_ut, min_en_order, max_ut_en, max_ut, max_ut_order, energy_collect, utilization_collect, latency_collect,
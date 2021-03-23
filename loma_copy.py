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
    [layer_origin, layer_rounded] = layer_comb
    [spatial_loop, spatial_loop_fractional] = spatial_loop_comb
    msc = mem_scheme
    temporal_loop = cls.TemporalLoop.extract_loop_info(layer_rounded, allocated_order, spatial_loop)
    loop = cls.Loop.extract_loop_info(layer_rounded, temporal_loop, spatial_loop, input_settings.precision,
                                      input_settings.fixed_temporal_mapping)

    # Fractional loop for greedy mapping
    if input_settings.spatial_unrolling_mode in [4, 5]:
        ############# Advanced User Configuration #############
        # mem_energy_saving_when_BW_under_utilized = True
        #######################################################
        temporal_loop_fractional = cls.TemporalLoop.extract_loop_info(layer_origin, allocated_order,
                                                                      spatial_loop_fractional)
        loop_fractional = cls.Loop.extract_loop_info(layer_origin, temporal_loop_fractional, spatial_loop_fractional,
                                                     input_settings.precision,
                                                     input_settings.fixed_temporal_mapping)
        # if mem_energy_saving_when_BW_under_utilized is False:
        #     loop_fractional = mem_access_count_correct(loop_fractional, loop)
    else:
        loop_fractional = loop

    utilization = cls.Utilization.get_utilization(layer_rounded, temporal_loop, spatial_loop_comb, loop,
                                                  input_settings.mac_array_info, msc.mem_size,
                                                  msc.mem_share, msc.mem_type,
                                                  input_settings.mac_array_stall,
                                                  input_settings.precision, msc.mem_bw)

    total_cost_layer = 0
    # loop.array_wire_distance = {'W': [], 'I': [], 'O': []}
    operand_cost = {'W': [], 'I': [], 'O': []}
    schedule_info = 0  # not used right now

    active_mac_cost = cmf.get_active_mac_cost(layer_origin, input_settings.mac_array_info['single_mac_energy'])
    idle_mac_cost = cmf.get_idle_mac_cost(layer_origin, layer_rounded, input_settings.mac_array_info['array_size'],
                                          input_settings.mac_array_info['idle_mac_energy'],
                                          msc.spatial_unrolling)

    for operand in ['W', 'I', 'O']:
        for level in range(0, len(allocated_order[operand])):
            operand_cost[operand].append(
                cmf.get_operand_level_energy_cost(operand, level, msc.mem_cost,
                                                  input_settings.mac_array_info,
                                                  schedule_info, loop_fractional,
                                                  msc.mem_fifo, msc, input_settings.precision,
                                                  utilization, ii))
            # TODO
            # loop.array_wire_distance[operand].append(
            #     cmf.get_operand_level_wire_distance(operand, level,
            #                                         schedule_info,
            #                                         input_settings.mac_array_info, loop,
            #                                         msc.mem_fifo))
        total_cost_layer += np.sum(operand_cost[operand])
    total_cost_layer += active_mac_cost + idle_mac_cost[ii_su]

    try:
        greedy_mapping_flag = mem_scheme.greedy_mapping_flag[ii_su]
        footer_info = mem_scheme.footer_info[ii_su]
    except:
        greedy_mapping_flag = False
        footer_info = None

    # TODO MAC area (multiplier and adder) is not included.
    # occupied_area format: [total_area, active_area]
    occupied_area = msg.get_mem_scheme_area2(msc, spatial_loop.unit_count, utilization.mac_utilize_spatial)

    # Get CostModelOutput
    flooring = mem_scheme.flooring
    cost_model_output = of.CostModelOutput(total_cost_layer, deepcopy(operand_cost),
                                           (active_mac_cost, idle_mac_cost[ii_su]),
                                           deepcopy(temporal_loop.temporal_loop),
                                           deepcopy(mem_scheme.spatial_unrolling[ii_su]),
                                           flooring[ii_su],
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


def create_multiset_format(n, rs, pfs):
    """
    Function that converts the given n, rs and pfs into a multiset format.

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


def limit_lpf(layer_spec_pf, layer_spec_pf_count, layer_spec_pf_count_sum, lpf_limit):
    """
    Function to limit the total number of loop prime factors.
    This function scans the lpfs and while the number of lpfs is greater than the lpf_limit it:
    - picks the loop type that has the most lpfs
    - merges the smallest two lpfs of that loop type (multiplying their values)
    """
    limit = lpf_limit

    n_pf = sum(layer_spec_pf_count_sum.values())
    if n_pf <= limit:
        # print("No prime factor limiting performed, n_pf =", n_pf)
        return layer_spec_pf, layer_spec_pf_count, n_pf

    while n_pf > limit:

        max_lt = max(layer_spec_pf_count_sum.items(), key=operator.itemgetter(1))[0]  # lt with highest # of pfs
        max_pfs = list(layer_spec_pf[max_lt])  # pfs for max_lt
        max_counts = list(layer_spec_pf_count[max_lt])  # counts for different pfs in max_lt

        if max_counts[0] == 1:  # multiplicity of smallest pf is 1
            new_factor = max_pfs[0] * max_pfs[1]
            max_counts[0] -= 1
            max_counts[1] -= 1

        else:  # multiplicity of smallest pf > 1
            new_factor = max_pfs[0] * max_pfs[0]
            max_counts[0] -= 2

        if new_factor in max_pfs:  # possible if not first iteration of while loop
            new_factor_idx = max_pfs.index(new_factor)
            max_counts[new_factor_idx] += 1
        else:
            new_factor_idx = len([pf for pf in max_pfs if pf < new_factor])
            max_pfs.insert(new_factor_idx, new_factor)
            max_counts.insert(new_factor_idx, 1)  # count of new factor is 1

        # Sanitize max_pfs and max_counts to remove all elements with multiplicity 0
        non_zero_idxs = [idx for idx in range(len(max_counts)) if max_counts[idx] != 0]
        max_pfs = [max_pfs[non_zero_idx] for non_zero_idx in non_zero_idxs]
        max_counts = [max_counts[non_zero_idx] for non_zero_idx in non_zero_idxs]

        # Update the layer_spec_pf, layer_spec_pf_count and layer_spec_pf_count_sum with updated factors/counts
        layer_spec_pf[max_lt] = tuple(max_pfs)
        layer_spec_pf_count[max_lt] = tuple(max_counts)
        layer_spec_pf_count_sum[max_lt] -= 1

        # Decrease the total number of pfs (actually not all prime anymore) by 1
        n_pf -= 1

    # print("Prime factor limit performed, n_pf:", n_pf_start, "-->", n_pf)
    return layer_spec_pf, layer_spec_pf_count, n_pf


def get_prime_factors(layer_spec: dict, lpf_limit: int):
    """

    Parameters
    ----------
    layer_spec
    lpf_limit

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
        pfs = []
        counts = []
        for pf, count in factors.items():
            pfs.append(pf)
            counts.append(count)
        layer_spec_pf[loop_type] = tuple(pfs)
        layer_spec_pf_count[loop_type] = tuple(counts)
        layer_spec_pf_count_sum[loop_type] = sum(counts)

    total_lpf_count = sum(layer_spec_pf_count_sum.values())

    layer_spec_pf, layer_spec_pf_count, total_lpf_count = limit_lpf(
        layer_spec_pf, layer_spec_pf_count, layer_spec_pf_count_sum, lpf_limit
    )

    return layer_spec_pf, layer_spec_pf_count, total_lpf_count


def get_smallest_pf(tl):
    if tl is None:
        return None

    smallest_pf = float("inf")
    for lpf in tl:
        if lpf == "X":
            continue
        else:
            pf = lpf[1]
            if pf < smallest_pf:
                smallest_pf = pf
    assert smallest_pf != float("inf"), "Only Xs are present within the given loop type ordering"

    return smallest_pf


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
    for loop_type in loop_types_list:  # always loop in fixed order
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
        tl_dict[loop_type] = permutations_list  # Add all the found permutations for this loop_type to tl_dict
    return count_dict, loop_type_order_list, tl_dict, total_count


def generate_tm_orderings(layer_spec: dict, spatial_unrolling: dict, lpf_limit: int):
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
    count_dict, loop_type_order, tl_dict, total_count = get_all_orderings_list(layer_spec_pf, layer_spec_pf_count, total_lpf_count)
    return tl_dict, count_dict, loop_type_order, total_count


def merge_loops(order, smallest_pfs):
    merged_order = []
    (previous_lt_number, previous_factors) = order[0]
    first_of_this_lt = True
    for (lt_number, factor) in order[1:]:
        if lt_number == previous_lt_number:
            if first_of_this_lt and previous_factors == smallest_pfs[previous_lt_number]:
                merged_order.append((previous_lt_number, previous_factors))
                previous_factors = factor
                first_of_this_lt = False
            else:
                previous_factors *= factor
        elif previous_factors != 1:
            merged_order.append((previous_lt_number, previous_factors))
            previous_lt_number = lt_number
            previous_factors = factor
            first_of_this_lt = True
    merged_order.append((previous_lt_number, previous_factors))

    return tuple(merged_order)


def tl_worker_new(
        tl_list,
        merged_count_dict,
        loop_type_order,
        total_merged_count,
        input_settings,
        spatial_loop_comb,
        mem_scheme,
        precision,
        layer,
        mac_costs,
):
    """
    New tl_worker function to handle the multiset loop orderings.

    These orderings still require a memory allocation, followed by a cost model evaluation.
    """

    # Get the different MemoryNodes we need to allocate
    nodes = mem_scheme.nodes
    n_mem_levels = len(nodes)

    # Layer
    [layer_origin, layer_rounded] = layer

    # Spatial unrolling
    [spatial_loop, spatial_loop_fractional] = spatial_loop_comb

    # Get the active and idle MAC cost
    [active_mac_cost, idle_mac_cost] = mac_costs

    # loop_type order ['B','K','C','OY','OX','FY','FX']
    # Adjust the merged_count_dict for the first loop_type, as this got chunked in caller function
    first_loop_type = loop_type_order[0]
    merged_count_dict[first_loop_type] = len(tl_list[first_loop_type])

    # Get the orderings for all possible loop_types (some might be non-existent)
    tl_list_B = tl_list.get("B", [None])
    tl_list_K = tl_list.get("K", [None])
    tl_list_C = tl_list.get("C", [None])
    tl_list_OY = tl_list.get("OY", [None])
    tl_list_OX = tl_list.get("OX", [None])
    tl_list_FY = tl_list.get("FY", [None])
    tl_list_FX = tl_list.get("FX", [None])

    # Get the smallest prime factor for each loop type (required for loop merging)
    smallest_pfs = {
        7: get_smallest_pf(tl_list_B[0]),
        6: get_smallest_pf(tl_list_K[0]),
        5: get_smallest_pf(tl_list_C[0]),
        4: get_smallest_pf(tl_list_OY[0]),
        3: get_smallest_pf(tl_list_OX[0]),
        2: get_smallest_pf(tl_list_FY[0]),
        1: get_smallest_pf(tl_list_FX[0]),
    }

    # Init minimal energy and max utilization results
    min_en = float("inf")
    min_en_ut = 0
    max_ut_en = float("inf")
    max_ut = 0

    # Init energy,latency,utilization collect
    energy_collect = None
    utilization_collect = None
    latency_collect = None
    save_all_tm = input_settings.tm_search_result_saving
    if save_all_tm:
        energy_collect = []
        utilization_collect = []
        latency_collect = []

    # Loop through all the number of elements in each tl_list element
    ctr = 0
    skipped = 0
    merged_set = set()
    for order_B in tl_list_B:
        for order_K in tl_list_K:
            order_B_K = combine_orderings(order_B, order_K)
            for order_C in tl_list_C:
                order_B_K_C = combine_orderings(order_B_K, order_C)
                for order_OY in tl_list_OY:
                    order_B_K_C_OY = combine_orderings(order_B_K_C, order_OY)
                    for order_OX in tl_list_OX:
                        order_B_K_C_OY_OX = combine_orderings(order_B_K_C_OY, order_OX)
                        for order_FY in tl_list_FY:
                            order_B_K_C_OY_OX_FY = combine_orderings(order_B_K_C_OY_OX, order_FY)
                            for order_FX in tl_list_FX:
                                ctr += 1

                                # Final order with all X's filled in
                                nonmerged_order = combine_orderings(order_B_K_C_OY_OX_FY, order_FX)

                                # Merge loops of same type
                                merged_order = merge_loops(nonmerged_order, smallest_pfs)

                                # Check if merged order was already processed
                                hashed = hash(merged_order)
                                if hashed in merged_set:
                                    skipped += 1
                                    continue
                                else:
                                    merged_set.add(hashed)

                                ################################## MEMORY ALLOCATION ##################################

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

                                ################################## COST MODEL EVALUATION ##################################
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

                                # Greedy mapping: loop_fractional required
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

                                utilization = cls.Utilization.get_utilization(
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
                                operand_cost = {"W": [], "I": [], "O": []}
                                total_cost_layer = 0
                                for operand in ["W", "I", "O"]:
                                    for level in range(0, len(allocated_order[operand])):
                                        operand_cost[operand].append(
                                            cmf.get_operand_level_energy_cost(
                                                operand,
                                                level,
                                                mem_scheme.mem_cost,
                                                input_settings.mac_array_info,
                                                0,
                                                loop_fractional,
                                                mem_scheme.mem_fifo,
                                                mem_scheme,
                                                input_settings.precision,
                                                utilization,
                                                False,
                                            )
                                        )
                                    total_cost_layer += np.sum(operand_cost[operand])
                                total_cost_layer += active_mac_cost + idle_mac_cost

                                ############################# COMPARISON WITH BEST SO FAR #############################
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
                                if save_all_tm:
                                    energy_collect.append(int(en))
                                    utilization_collect.append(ut)
                                    latency_collect.append(utilization.latency_no_load)

                                # if ctr % 1000 == 0:
                                #     print(ctr, "Execution time =", time.time()-t_start)
                                #     t_start = time.time()

    return (
        min_en,
        min_en_ut,
        min_en_order,
        max_ut_en,
        max_ut,
        max_ut_order,
        energy_collect,
        utilization_collect,
        latency_collect,
    )

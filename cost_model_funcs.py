import numpy as np
import copy
import sys
import math
from numpy import prod

"""

This file includes all the functions used in the cost model.

"""


def get_operand_level_energy_cost(operand, level, mem_word_cost, mac_array_info, schedule_info, loop, mem_fifo,
                                  mem_scheme, precision, utilization, sum_shared_bw):
    # mac_cost = get_mac_cost(layer, mac_array_info['single_mac_energy'])
    #
    # if level < len(schedule_info['temporal'][operand]) - 1:
    #     wire_cost = get_operand_level_wire_cost(operand, level, schedule_info, mac_array_info, loop, mem_fifo)
    # else:
    #     wire_cost = 0
    wire_cost = 0

    mem_cost_dy = get_operand_level_dynamic_mem_cost(operand, level, loop, mem_word_cost, mem_scheme, precision,
                                                     utilization, sum_shared_bw)

    mem_cost_st = get_static_mem_cost()

    return [wire_cost, mem_cost_dy, mem_cost_st]


def get_active_mac_cost(layer, single_mac_energy):
    return layer.total_MAC_op * single_mac_energy


def get_idle_mac_cost(layer, layer_rounded, array_size, idle_mac_energy, spatial_unrolling):
    idle_mac_cost = []
    for su in spatial_unrolling:
        active_mac_count = 1
        for level_list in su['W']:
            if level_list:
                for su_unit in level_list:
                    active_mac_count *= su_unit[1]
        total_mapping_count = math.ceil(layer_rounded.total_MAC_op/active_mac_count)
        ideal_mac_count = total_mapping_count * array_size[0] * array_size[1]
        idle_mac_count = ideal_mac_count - layer.total_MAC_op
        idle_mac_cost.append(idle_mac_count * idle_mac_energy)
    return idle_mac_cost


def get_operand_level_inter_pe_distance(op, operand_partitions, input_temporal_loops, is_fifo):
    """
    The function computes the worst inter-PE distance to be covered by each variable accessed at level above of array for
    each operand across the listed dimensions.
    If two dimensions, first dimension is assumed to be the unrolling/flooring of operand across columns
                        second dimension is assumed to be the unrolling/flooring of operand across rows

    Across each dimension, if flooring is present, each flooring block has to be reached via an offset distance, where
    offset is a list that contains the offsets for each flooring block.
    Inside each flooring block, given the unrolling scheme, an inter-block distance has to be covered.

    The distance for each array dimension is computed as the sum(distance) * len(offset) + sum(offset)

    :param op: Operand type, char variable in ['I','W','O']
    :param operand_partitions: Unrolling and flooring scheme for all partitions at a given level, expressed as
        operand_partitions[operand][dimension][flooring block type][flooring block size]
    :return: distance covered by the given operand type with the given operand partition scheme
    """

    '''
    operand_irrelevant_types contains the irrelevant loop for each operand as enum integers (ref. layer.py)
        operand_irrelevant_types[operand][irrelevant types]
    '''
    output_irrelevant = [1, 2, 5, 7]
    input_irrelevant = [6]
    weight_irrelevant = [3, 4, 7]
    operand_irrelevant_types = {'W': weight_irrelevant, 'I': input_irrelevant, 'O': output_irrelevant}
    opit_distance = [0]
    # Count array cost for each operand type
    operand_partitions = operand_partitions[op]
    opit = 0
    total_distance_opit = 0
    length = []
    lengthx = [0]
    distance = []
    count = []
    dim_distance = []

    '''
    If the operand_partitions at the given level list is empty, the distance covered in the array will be zero since
    there's no unrolling, so returns [0] 
    '''
    if not operand_partitions:
        return [0]
    else:
        for dim in range(len(operand_partitions)):
            '''
            INTER-BLOCK DISTANCE
            Inside each flooring block, the unrolling scheme is considered. The inter-block distance is the sum of 
            jumps between elements of the array times the length of each jump. 
            TODO rest of commenting :P
            '''
            distance = []
            for i in range(0, len(operand_partitions[dim])):
                if not operand_partitions[dim][i]:
                    continue
                if operand_partitions[dim][i][0] in operand_irrelevant_types[op]:
                    # COUNT OF JUMPS
                    n = operand_partitions[dim][i][1]
                    nx = 1
                    for j in range(len(operand_partitions[dim]) - 2, i, -1):
                        if operand_partitions[dim][j][0] in operand_irrelevant_types[op]:
                            nx = nx * operand_partitions[dim][j][1]
                    n = n * nx
                    count.append(n)
                    # LENGTH OF EACH JUMP
                    size_partitions_below = 1
                    for k in range(0, i):
                        size_partitions_below = size_partitions_below * operand_partitions[dim][k][1]
                    length.append(size_partitions_below - lengthx[-1])
                    lengthx.append(size_partitions_below)
                    distance.append(length[-1] * count[-1])

            '''
            OFFSET
            Given the flooring/unrolling scheme, across the considered dimension if irrelevant partition loops are present
            "islands" (the flooring blocks) of repeated values can be localized. While the distance inside the flooring 
            blocks is computed as the inter-block distance, each block is also characterized by an offset wrt source.

            The offset for each block is computed as base_step * j + baseline. 
                Baseline is the distance to be covered that contains all previous flooring blocks
                Base step is the product of the sizes of the innermost relevant partitions in the flooring block
            '''
            offset = []
            base_step = 1
            if operand_partitions[dim]:
                first_relevant_partition = len(operand_partitions[dim])
                for i in range(len(operand_partitions[dim])):
                    if operand_partitions[dim][i][0] not in operand_irrelevant_types[op]:
                        first_relevant_partition = i
                        for j in range(0, i):
                            base_step = base_step * operand_partitions[dim][j][1]
                        baseline = 0
                        for j in range(0, operand_partitions[dim][first_relevant_partition][1]):
                            offset.append(base_step * j + baseline)
                        break

                for i in range(first_relevant_partition + 1, len(operand_partitions[dim])):
                    if operand_partitions[dim][i][0] not in operand_irrelevant_types[op]:
                        for k in range(1, operand_partitions[dim][i][1]):
                            baseline = 1
                            if i == 0:
                                baseline = 0
                            for j in range(0, i):
                                baseline = baseline * operand_partitions[dim][j][1] * k
                            for j in range(0, operand_partitions[dim][first_relevant_partition][1]):
                                offset.append(base_step * j + baseline)
            if not operand_partitions[dim] or not offset:
                offset = [0]

            rtl = 1
            fifo_distance = 0
            if op == 'I':
                if is_fifo == True:
                    unroll_size = 1
                    unroll_loop_type = 0
                    for j in range(len(operand_partitions[dim])):
                        if operand_partitions[dim][j][0] in [1, 2, 3, 4]:
                            unroll_loop_type = operand_partitions[dim][j][0]
                            for m in range(0, j + 1):
                                unroll_size *= operand_partitions[dim][m][1]
                        break
                    first_relevant_temporal_loop_size = 1
                    tmp_tl = [i for i in input_temporal_loops if i[0] != 6]

                    if tmp_tl:
                        if unroll_loop_type == 1:
                            if tmp_tl[0][0] == 3:
                                rtl = tmp_tl[0][1]
                        if unroll_loop_type == 3:
                            try:
                                if tmp_tl[0][0] == 1:
                                    rtl = tmp_tl[0][1]
                            except:
                                a = 1
                        if unroll_loop_type == 2:
                            if tmp_tl[0][0] == 4:
                                rtl = tmp_tl[0][1]
                        if unroll_loop_type == 4:
                            if tmp_tl[0][0] == 2:
                                rtl = tmp_tl[0][1]
                    # TODO this formula has to be corrected. 1 should be the number of irrelevant jumps and 0 the sum of the lenghts of the irrelevant jumps
                    # Since we assume that there are no replications (FOR NOW) it will be corrected later

                    fifo_distance = (rtl - 1) * (unroll_size - 1) * 1 + 0

            div_factor = 1
            # if op != 'I':
            #     for j in range(len(operand_partitions[dim])):
            #         if operand_partitions[dim][j][0] not in operand_irrelevant_types[op]:
            #             div_factor *= operand_partitions[dim][j][1]
            # else:
            #     for j in range(len(operand_partitions[dim])):
            #         if operand_partitions[dim][j][0] not in operand_irrelevant_types[op]:
            #             div_factor *= operand_partitions[dim][j][1]
            #     div_factor = div_factor + rtl - 1

            dim_distance.append((sum(distance) * len(offset) + sum(offset) + fifo_distance) / div_factor)

        '''
        In the case of two dimensions, the distance is computed as in a 2D mesh network:
            The distance across rows is taken only once
            The distance across columns is multiplied by the number of rows
        '''
        if len(operand_partitions) == 2:
            num_rows = 1
            for i in range(len(operand_partitions[1])):
                num_rows = num_rows * operand_partitions[1][i][1]
            row_distance = dim_distance[0] * num_rows
            col_distance = dim_distance[1]
            total_distance_opit = row_distance + col_distance
        if len(operand_partitions) == 1:
            total_distance_opit = dim_distance[0]

        opit_distance[opit] = total_distance_opit

        return opit_distance


def get_operand_level_wire_cost(op, level, schedule_info, mac_array_info, loop, mem_fifo):
    return 0

    # """
    # Wire cost is calculated as inter-PE cost + memory interconnection cost
    # """
    # # Inter-PE cost
    # """
    # Get above-array-level memory (just one level above the array) access count for W/I/O (total access for each),
    # and times them with corresponding inter-PE movement step (based on spatial unrolling type and size)
    # and unit_wire_energy (energy for 1-bit data movement between neighbour PE)
    # """
    # """
    # Inter-PE distance covered
    # """
    #
    # operand_types = ['W', 'I', 'O']
    #
    # partitions = {
    #     'W': [],
    #     'I': [],
    #     'O': []}
    #
    # '''
    # Given that the spatial unrolling and the flooring loops are stored in two different variables (ref
    # cost_model_input.schedule_info), in order to compute the array cost is easier to change the representation in a
    # unique variable that stores the information for both spatial unrolling and flooring.
    # Operand partitions are represented as:
    #     operand_paritions[operand][level][dimension][flooring block type][flooring block size]
    # Where operand in ['I','W','O']
    # '''
    # for operand in operand_types:
    #     for lev in range(0, len(schedule_info['spatial'][operand])):
    #         partitions[operand].append([])
    #         for floor_dim in range(0, len(schedule_info['flooring'][operand][lev])):
    #             partitions[operand][lev].append([])
    #             for flooring_type in range(0, len(schedule_info['flooring'][operand][lev][floor_dim])):
    #                 w, a = zip(*schedule_info['spatial'][operand][lev])
    #                 try:
    #                     partitions[operand][lev][floor_dim].append(
    #                         list(schedule_info['spatial'][operand][lev][
    #                                  w.index(schedule_info['flooring'][operand][lev][floor_dim][flooring_type])]))
    #                 except:
    #                     return 0
    # for operand in operand_types:
    #     partitions[operand] += [[]] * (level)
    #     for lev in range(0, len(partitions[operand])):
    #         if not partitions[operand][lev]:
    #             partitions[operand][lev].append([])
    # try:
    #     operand_partitions = {'I': partitions['I'][level], 'O': partitions['O'][level], 'W': partitions['W'][level]}
    # except IndexError:
    #     print({'I': partitions['I'], 'O': partitions['O'], 'W': partitions['W']})
    #     print(level, op)
    #     # continue
    #     # sys.exit()
    #     a = 1
    # '''
    # Given the adopted representation for the unrolling and flooring schemes, the variable is passed to the function that
    # computes the distance that each single variable has to cover in order to reach its position in the array
    # '''
    # try:
    #     operand_distance = get_operand_level_inter_pe_distance(op, operand_partitions,
    #                                                            schedule_info['temporal'][op][level], loop.I_fifo[level])
    # except IndexError:
    #     operand_distance = get_operand_level_inter_pe_distance(op, operand_partitions,
    #                                                            schedule_info['temporal'][op][level], False)
    # operand_distance = np.array(operand_distance)
    # operand_irrelevant = {'W': [3, 4, 7], 'I': [6], 'O': [1, 2, 5]}
    # '''
    # INPUT COST : computed as (distance covered) x (read accesses from level above) x (unit bit wire cost) x (bits
    # of precision)
    # '''
    # div_factor = 1
    # for partition in operand_partitions[op]:
    #     for unrolling in partition:
    #         if unrolling[0] not in operand_irrelevant[op]:
    #             div_factor *= unrolling[1]
    #
    # if op == 'I':
    #     if mem_fifo[op][level]:
    #         array_cost = operand_distance[0] * loop.mem_access_elem['I'][level][0] * mac_array_info[
    #             'unit_wire_energy'][level] * \
    #                      mac_array_info['precision'] / div_factor
    #     else:
    #         array_cost = operand_distance[0] * loop.mem_access_elem['I'][level][0] * mac_array_info[
    #             'unit_wire_energy'][level] * \
    #                      mac_array_info['precision'] / div_factor
    #
    # '''
    # WEIGHT COST : computed as (distance covered) x (read accesses from level above) x (unit bit wire cost) x (bits
    # of precision)
    # '''
    # if op == 'W':
    #     array_cost = operand_distance[0] * loop.mem_access_elem['W'][level][0] * mac_array_info['unit_wire_energy'][
    #         level] * \
    #                  mac_array_info['precision'] / div_factor
    #
    # '''
    # OUTPUT COST :
    #     if PARTIAL OUTPUT: (distance covered) x (read+write accesses from level above) x (unit bit wire cost) x (bits of
    #         precision+headroom bits)
    #     if FINAL OUTPUT:   (distance covered) x (write accesses to level above) x (unit bit wire cost) x (bits of
    #         precision)
    # '''
    # if op == 'O':
    #     if loop.mem_access_elem['O_final'][level][0][1] == 0:
    #         array_cost = operand_distance[0] * sum(loop.mem_access_elem['O_partial'][level][0]) * \
    #                      mac_array_info['unit_wire_energy'][level] * \
    #                      sum([mac_array_info['precision'] * 2, mac_array_info['headroom']]) / div_factor
    #     else:
    #         array_cost = operand_distance[0] * loop.mem_access_elem['O_final'][level][0][1] * \
    #                      mac_array_info['unit_wire_energy'][level] * \
    #                      mac_array_info['precision'] / div_factor
    #
    # return array_cost
    #
    # # TODO Memory interconnection cost


def get_operand_level_wire_distance(op, level, schedule_info, mac_array_info, loop, mem_fifo):
    return [0, 0]
    # """
    #     Wire cost is calculated as inter-PE cost + memory interconnection cost
    #     """
    # # Inter-PE cost
    # """
    # Get above-array-level memory (just one level above the array) access count for W/I/O (total access for each),
    # and times them with corresponding inter-PE movement step (based on spatial unrolling type and size)
    # and unit_wire_energy (energy for 1-bit data movement between neighbour PE)
    # """
    # """
    # Inter-PE distance covered
    # """
    #
    # operand_types = ['W', 'I', 'O']
    #
    # partitions = {
    #     'W': [],
    #     'I': [],
    #     'O': []}
    #
    # '''
    # Given that the spatial unrolling and the flooring loops are stored in two different variables (ref
    # cost_model_input.schedule_info), in order to compute the array cost is easier to change the representation in a
    # unique variable that stores the information for both spatial unrolling and flooring.
    # Operand partitions are represented as:
    #     operand_paritions[operand][level][dimension][flooring block type][flooring block size]
    # Where operand in ['I','W','O']
    # '''
    # # for operand in operand_types:
    # #     for lev in range(0, len(schedule_info['spatial'][operand])):
    # #         partitions[operand].append([])
    # #         for floor_dim in range(0, len(schedule_info['flooring'][operand][lev])):
    # #             partitions[operand][lev].append([])
    # #             for flooring_type in range(0, len(schedule_info['flooring'][operand][lev][floor_dim])):
    # #                 w, a = zip(*schedule_info['spatial'][operand][lev])
    # #                 try:
    # #                     partitions[operand][lev][floor_dim].append(
    # #                     list(schedule_info['spatial'][operand][lev][
    # #                              w.index(schedule_info['flooring'][operand][lev][floor_dim][flooring_type])]))
    # #                 except:
    # #                     a=1
    # for operand in operand_types:
    #     for lev in range(0, len(schedule_info['spatial'][operand])):
    #         partitions[operand].append([])
    #         for floor_dim in range(0, len(schedule_info['flooring'][operand][lev])):
    #             partitions[operand][lev].append([])
    #             for flooring_type in range(0, len(schedule_info['flooring'][operand][lev][floor_dim])):
    #                 w, a = zip(*schedule_info['spatial'][operand][lev])
    #                 try:
    #                     partitions[operand][lev][floor_dim].append(
    #                         list(schedule_info['spatial'][operand][lev][
    #                                  w.index(schedule_info['spatial'][operand][lev][flooring_type][0])]))
    #                 except:
    #                     return 0
    #
    # for operand in operand_types:
    #     partitions[operand] += [[]] * (level)
    #     for lev in range(0, len(partitions[operand])):
    #         if not partitions[operand][lev]:
    #             partitions[operand][lev].append([])
    # try:
    #     operand_partitions = {'I': partitions['I'][level], 'O': partitions['O'][level], 'W': partitions['W'][level]}
    # except IndexError:
    #     print({'I': partitions['I'], 'O': partitions['O'], 'W': partitions['W']})
    #     print(level, op)
    #     # continue
    #     # sys.exit()
    #     a = 1
    # operand_partitions = {'I': partitions['I'][level], 'O': partitions['O'][level], 'W': partitions['W'][level]}
    #
    # '''
    # Given the adopted representation for the unrolling and flooring schemes, the variable is passed to the function that
    # computes the distance that each single variable has to cover in order to reach its position in the array
    # '''
    # try:
    #     operand_distance = get_operand_level_inter_pe_distance(op, operand_partitions,
    #                                                            schedule_info['temporal'][op][level],
    #                                                            loop.I_fifo[level])
    # except IndexError:
    #     operand_distance = get_operand_level_inter_pe_distance(op, operand_partitions,
    #                                                            schedule_info['temporal'][op][level], False)
    # operand_distance = np.array(operand_distance)
    #
    # operand_irrelevant = {'W': [3, 4, 7], 'I': [6], 'O': [1, 2, 5]}
    # div_factor = 1
    # for partition in operand_partitions[op]:
    #     for unrolling in partition:
    #         if unrolling[0] not in operand_irrelevant[op]:
    #             div_factor *= unrolling[1]
    #
    # if op == 'I':
    #     if mem_fifo[op][level]:
    #         array_distance = operand_distance[0] * loop.mem_access_elem['I'][level][0] * mac_array_info[
    #             'precision'] / div_factor
    #     else:
    #         array_distance = operand_distance[0] * loop.mem_access_elem['I'][level][0] * mac_array_info[
    #             'precision'] / div_factor
    #
    # if op == 'W':
    #     array_distance = operand_distance[0] * loop.mem_access_elem['W'][level][0] * mac_array_info[
    #         'precision'] / div_factor
    #
    # if op == 'O':
    #
    #     if loop.mem_access_elem['O_final'][level][0][1] == 0:
    #
    #         array_distance = operand_distance[0] * sum(loop.mem_access_elem['O_partial'][level][0]) * sum(
    #             [mac_array_info['precision'] * 2, mac_array_info['headroom']]) / div_factor
    #     else:
    #         array_distance = operand_distance[0] * loop.mem_access_elem['O_final'][level][0][1] * mac_array_info[
    #             'precision'] / div_factor
    #
    # return [array_distance, list(operand_distance)]


def iterative_data_format_clean(original_dict):
    new_dict = {'W':[], 'I':[], 'O':[]}
    for operand in ['W', 'I', 'O']:
        for li in original_dict[operand]:
            new_dict[operand].append(li[0])
    return new_dict


def get_operand_level_dynamic_mem_cost(operand, level, loop, mem_word_cost, mem_scheme, precision, utilization,
                                       sum_shared_bw):
    """
    The function computes the dynamic energy consumed for accessing a memory at a certain level for a given operand
    :param operand : Should be one of ['I', 'O', 'W']
    :param level : Integer number, level with respect to temporal blocking distribution
    :param loop : loop object, contains number of memory accesses. For ref, view loop class
    :param mem_word_cost : mem word energy cost

    The dynamic energy consumption is computed as read cost + write cost at a given memory level for a defined operand

    For 'I', 'W':
        Read (write) cost are computed as the product of number of read (write) memory accesses per level times the
        cost per word  access
    For 'O':
        Given that outputs are divided in two categories (O_partial and O_final) with different access costs and
        Given that at each level there's different numbers of writes and reads to level below and above
        The read (write) cost is computed as the sum of read (write) accesses to level below + read (write) accesses to
        level above for O_partial and for O_final times the relative access costs
    """

    """
    FOR COMPUTING SINGLE COST : (PRECISION / BW) * ACCESS COST
    """

    if type(mem_scheme.mem_bw['W'][0][0]) in [list, tuple]:
        mem_scheme.mem_bw = iterative_data_format_clean(mem_scheme.mem_bw)

    if type(mem_word_cost['W'][0][0]) in [list, tuple]:
        mem_word_cost = iterative_data_format_clean(mem_word_cost)

    if not sum_shared_bw:
        # READ COST
        if utilization.req_mem_bw_bit[operand][level][0] <= mem_scheme.mem_bw[operand][level][0]:
            if operand == 'O':
                read_cost = (((loop.mem_access_elem['O_final'][level][0][0] + loop.mem_access_elem['O_final'][level][1][0]) * precision['O_final'] / \
                              mem_scheme.mem_bw[operand][level][0]) * utilization.pun_factor[operand][level] *
                              mem_word_cost['O'][level][0]) + (((loop.mem_access_elem['O_partial'][level][0][0] +
                                                                loop.mem_access_elem['O_partial'][level][1][0]) *
                                                               precision['O'] / mem_scheme.mem_bw[operand][level][0])
                                                              * utilization.pun_factor[operand][level] *
                                                              mem_word_cost['O'][level][0])
            else:
                read_cost = (loop.mem_access_elem[operand][level][0] * precision[operand] /
                             mem_scheme.mem_bw[operand][level][0]) * utilization.pun_factor[operand][level] * \
                             mem_word_cost[operand][level][0]

        else:
            if operand == 'O':
                read_cost = (((loop.mem_access_elem['O_final'][level][0][0] + loop.mem_access_elem['O_final'][level][1][0]) *
                              (precision['O_final'] / mem_scheme.mem_bw[operand][level][0])) * mem_word_cost['O'][level][0]) + \
                            (((loop.mem_access_elem['O_partial'][level][0][0] +
                               loop.mem_access_elem['O_partial'][level][1][0]) *
                              (precision['O'] / mem_scheme.mem_bw[operand][level][0])) * mem_word_cost['O'][level][0])
            else:
                read_cost = (loop.mem_access_elem[operand][level][0] * precision[operand] /
                             mem_scheme.mem_bw[operand][level][0]) * mem_word_cost[operand][level][0]

        # WRITE COST
        if utilization.req_mem_bw_bit[operand][level][1] <= mem_scheme.mem_bw[operand][level][1]:
            if operand == 'O':
                write_cost = (((loop.mem_access_elem['O_final'][level][0][1] +
                                loop.mem_access_elem['O_final'][level][1][1]) * precision['O_final'] /
                               mem_scheme.mem_bw[operand][level][1]) * mem_word_cost['O'][level][1]) + (
                                     ((loop.mem_access_elem['O_partial'][level][0][1] +
                                       loop.mem_access_elem['O_partial'][level][1][1]) * precision['O'] /
                                      mem_scheme.mem_bw[operand][level][1])
                                     * mem_word_cost['O'][level][1])
            else:
                write_cost = (loop.mem_access_elem[operand][level][1] * precision[operand] /
                              mem_scheme.mem_bw[operand][level][1]) * mem_word_cost[operand][level][1]
        else:
            if operand == 'O':
                write_cost = ((loop.mem_access_elem['O_final'][level][0][1] + loop.mem_access_elem['O_final'][level][1][1]) *
                              (precision['O_final'] / mem_scheme.mem_bw[operand][level][1]) * mem_word_cost['O'][level][1]) + (
                                     (loop.mem_access_elem['O_partial'][level][0][1] +
                                      loop.mem_access_elem['O_partial'][level][1][1]) *
                                     (precision['O'] / mem_scheme.mem_bw[operand][level][1]) * mem_word_cost['O'][level][1])
            else:
                write_cost = loop.mem_access_elem[operand][level][1] * (
                            precision[operand] / mem_scheme.mem_bw[operand][level][1]) * mem_word_cost[operand][level][1]

    else:
        if utilization.req_sh_mem_bw_bit[operand][level][0] <= mem_scheme.mem_bw[operand][level][0]:
            if operand == 'O':
                read_cost = (((loop.mem_access_elem['O_final'][level][0][0] + loop.mem_access_elem['O_final'][level][1][0]) * precision['O_final'] / \
                              mem_scheme.mem_bw[operand][level][0]) * utilization.pun_factor[operand][level] *
                             mem_word_cost['O'][level][0]) + (
                                    ((loop.mem_access_elem['O_partial'][level][0][0] +
                                      loop.mem_access_elem['O_partial'][level][1][0]) * precision['O'] /
                                     mem_scheme.mem_bw[operand][level][0]) * utilization.pun_factor[operand][level]
                                    * mem_word_cost['O'][level][0])
            else:
                read_cost = (loop.mem_access_elem[operand][level][0] * precision[operand] /
                             mem_scheme.mem_bw[operand][level][0]) * utilization.pun_factor[operand][level] * \
                            mem_word_cost[operand][level][0]
        else:
            if operand == 'O':
                read_cost = (((loop.mem_access_elem['O_final'][level][0][0] + loop.mem_access_elem['O_final'][level][1][0]) * \
                              (precision['O_final'] / mem_scheme.mem_bw[operand][level][0])) * mem_word_cost['O'][level][0]) + (((loop.mem_access_elem['O_partial'][level][0][0] +
                                              loop.mem_access_elem['O_partial'][level][1][0]) * (
                                                     precision['O'] / mem_scheme.mem_bw[operand][level][0]))
                                            * mem_word_cost['O'][level][0])
            else:
                read_cost = (loop.mem_access_elem[operand][level][0] * (
                        precision[operand] / mem_scheme.mem_bw[operand][level][0])) * mem_word_cost[operand][level][0]

        # WRITE COST
        if utilization.req_mem_bw_bit[operand][level][1] <= mem_scheme.mem_bw[operand][level][1]:
            if operand == 'O':
                write_cost = (((loop.mem_access_elem['O_final'][level][0][1] +
                                loop.mem_access_elem['O_final'][level][1][1]) * precision['O_final'] / \
                               mem_scheme.mem_bw[operand][level][1]) * mem_word_cost['O'][level][1]) + (
                                     ((loop.mem_access_elem['O_partial'][level][0][1] +
                                       loop.mem_access_elem['O_partial'][level][1][1]) * precision['O'] /
                                      mem_scheme.mem_bw[operand][level][1])
                                     * mem_word_cost['O'][level][1])
            else:
                write_cost = (loop.mem_access_elem[operand][level][1] * precision[operand] /
                              mem_scheme.mem_bw[operand][level][1]) * \
                             mem_word_cost[operand][level][1]
        else:
            if operand == 'O':
                write_cost = ((loop.mem_access_elem['O_final'][level][0][1] + loop.mem_access_elem['O_final'][level][1][1]) *
                              (precision['O_final'] / mem_scheme.mem_bw[operand][level][1]) * mem_word_cost['O'][1][
                                  level]) + (
                                     (loop.mem_access_elem['O_partial'][level][0][1] +
                                      loop.mem_access_elem['O_partial'][level][1][1]) *
                                     (precision['O'] / mem_scheme.mem_bw[operand][level][1]) * mem_word_cost['O'][1][
                                         level])
            else:
                write_cost = loop.mem_access_elem[operand][level][1] * (
                        precision[operand] / mem_scheme.mem_bw[operand][level][1]) * mem_word_cost[operand][level][1]

    return read_cost + write_cost


# TODO need to know memory operating frequency and leakage power. Ignore static memory cost for now.
def get_static_mem_cost():
    return 0


def su_correction(mem_scheme):
    su_len = {'W': len(mem_scheme.spatial_unrolling[0]['W']),
              'I': len(mem_scheme.spatial_unrolling[0]['I']),
              'O': len(mem_scheme.spatial_unrolling[0]['O'])}
    mem_len = {'W': len(mem_scheme.mem_type['W']),
               'I': len(mem_scheme.mem_type['I']),
               'O': len(mem_scheme.mem_type['O'])}

    for operand in ['W','I','O']:
        if su_len[operand] > mem_len[operand]+1:
            mem_scheme.spatial_unrolling[0][operand] = mem_scheme.spatial_unrolling[0][operand][:mem_len[operand]+1]
            mem_scheme.flooring[0][operand] = mem_scheme.flooring[0][operand][:mem_len[operand]+1]
        elif su_len[operand] < mem_len[operand]+1:
            append_su = [[]]*(mem_len[operand] + 1 -su_len[operand])
            mem_scheme.spatial_unrolling[0][operand].extend(append_su)
            mem_scheme.flooring[0][operand].extend(append_su)
    return mem_scheme


def get_mem_complete_unrolling_count(spatial_unrolling, flooring, array_size):
    """
    This function compute the complete memory unrolling count (active ones + inactive ones) for later area estimation.
    """
    XY_dimension_unrolling = [[], []]
    XY_dimension_unit_count = [
        {'W': [1] * len(flooring['W']), 'I': [1] * len(flooring['I']), 'O': [1] * len(flooring['O'])},
        {'W': [1] * len(flooring['W']), 'I': [1] * len(flooring['I']), 'O': [1] * len(flooring['O'])}]
    XY_dimension_mem_count_active = [
        {'W': [1] * (len(flooring['W'])-1), 'I': [1] * (len(flooring['I'])-1), 'O': [1] * (len(flooring['O'])-1)},
        {'W': [1] * (len(flooring['W'])-1), 'I': [1] * (len(flooring['I'])-1), 'O': [1] * (len(flooring['O'])-1)}]
    XY_dimension_mem_count_total = [
        {'W': [1] * (len(flooring['W'])-1), 'I': [1] * (len(flooring['I'])-1), 'O': [1] * (len(flooring['O'])-1)},
        {'W': [1] * (len(flooring['W'])-1), 'I': [1] * (len(flooring['I'])-1), 'O': [1] * (len(flooring['O'])-1)}]
    mem_count_active = {'W': [1] * (len(flooring['W']) - 1), 'I': [1] * (len(flooring['I']) - 1),
                        'O': [1] * (len(flooring['O']) - 1)}
    mem_count_total = {'W': [1] * (len(flooring['W'])-1), 'I': [1] * (len(flooring['I'])-1),
                       'O': [1] * (len(flooring['O'])-1)}
    XY_dimension_area_utilize = [{'W': 0, 'I': 0, 'O': 0},
                                 {'W': 0, 'I': 0, 'O': 0}]
    op_ir_loops = {'W': [3, 4, 7], 'I': [6], 'O': [1, 2, 5]}
    for floor_level in flooring['W']:
        if floor_level:
            for XY, floor_XY in enumerate(floor_level):
                if floor_XY:
                    XY_dimension_unrolling[XY].extend(floor_XY)
    for op in ['W', 'I', 'O']:
        for level, floor_level in enumerate(flooring[op]):
            if floor_level:
                i = 0
                for XY, floor_XY in enumerate(floor_level):
                    if floor_XY:
                        for floor_single in floor_XY:
                            if spatial_unrolling[op][level][i][0] != floor_single:
                                raise ValueError("spatial_unrolling's and flooring's order do not match.")
                            XY_dimension_unit_count[XY][op][level] *= spatial_unrolling[op][level][i][1]
                            i += 1
    for XY in range(len(XY_dimension_unit_count)):
        for op in ['W', 'I', 'O']:
            XY_dimension_area_utilize[XY][op] = prod(XY_dimension_unit_count[XY][op]) / array_size[XY]
            for level in range(1, len(XY_dimension_unit_count[XY][op])):
                XY_dimension_mem_count_active[XY][op][level-1] = prod(XY_dimension_unit_count[XY][op][level:])
    for op in ['W', 'I', 'O']:
        for level in range(1, len(spatial_unrolling[op])):
            if spatial_unrolling[op][level]:
                for XY in [0, 1]:
                    if all(loop_type in op_ir_loops[op] for loop_type in XY_dimension_unrolling[XY]):
                        XY_dimension_mem_count_total[XY][op][level-1] = XY_dimension_mem_count_active[XY][op][level-1]
                    else:
                        XY_dimension_mem_count_total[XY][op][level - 1] = \
                            int(round(XY_dimension_mem_count_active[XY][op][level-1]/XY_dimension_area_utilize[XY][op]))
    for op in ['W', 'I', 'O']:
        for level in range(len(XY_dimension_mem_count_active[0][op])):
            mem_count_active[op][level] = XY_dimension_mem_count_active[0][op][level] * \
                                          XY_dimension_mem_count_active[1][op][level]
            mem_count_total[op][level] = XY_dimension_mem_count_total[0][op][level] * \
                                          XY_dimension_mem_count_total[1][op][level]
    return mem_count_active, mem_count_total

import numpy as np
import copy

class Array(object):
    def __init__(self, layer, temporal_loop, spatial_loop):
        self.total_array_distance = 0
        self.total_array_cost = 0
        self.total_element_distance = 0
        self.array_distance_op_level = {}
        self.array_cost_op_level = {}
        self.array_element_distance = {}

    def get_operand_level_inter_pe_distance(self, op, operand_partitions, input_temporal_loops, is_fifo):
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
                        for j in range(len(operand_partitions[dim]) - 1, i, -1):
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
                if operand_partitions[dim][0]:
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
                if not operand_partitions[dim][0] or not offset:
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

                        if unroll_loop_type == 1:
                            if tmp_tl[0][0] == 3:
                                rtl = tmp_tl[0][1]
                        if unroll_loop_type == 3:
                            if tmp_tl[0][0] == 1:
                                rtl = tmp_tl[0][1]
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
                if op != 'I':
                    for j in range(len(operand_partitions[dim])):
                        if operand_partitions[dim][j][0] not in operand_irrelevant_types[op]:
                            div_factor *= operand_partitions[dim][j][1]
                else:
                    for j in range(len(operand_partitions[dim])):
                        if operand_partitions[dim][j][0] not in operand_irrelevant_types[op]:
                            div_factor *= operand_partitions[dim][j][1]
                    div_factor = div_factor + rtl - 1

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

    def get_operand_level_wire_cost(self, op, level, schedule_info, mac_array_info, loop, mem_fifo):
        """
        Wire cost is calculated as inter-PE cost + memory interconnection cost
        """
        # Inter-PE cost
        """
        Get above-array-level memory (just one level above the array) access count for W/I/O (total access for each),
        and times them with corresponding inter-PE movement step (based on spatial unrolling type and size)
        and unit_wire_energy (energy for 1-bit data movement between neighbour PE)
        """
        """
        Inter-PE distance covered
        """

        operand_types = ['W', 'I', 'O']

        partitions = {
            'W': [],
            'I': [],
            'O': []}

        '''
        Given that the spatial unrolling and the flooring loops are stored in two different variables (ref 
        cost_model_input.schedule_info), in order to compute the array cost is easier to change the representation in a 
        unique variable that stores the information for both spatial unrolling and flooring.
        Operand partitions are represented as:
            operand_paritions[operand][level][dimension][flooring block type][flooring block size]
        Where operand in ['I','W','O']
        '''
        for operand in operand_types:
            for lev in range(0, len(schedule_info['spatial'][operand])):
                partitions[operand].append([])
                for floor_dim in range(0, len(schedule_info['flooring'][operand][lev])):
                    partitions[operand][lev].append([])
                    for flooring_type in range(0, len(schedule_info['flooring'][operand][lev][floor_dim])):
                        w, a = zip(*schedule_info['spatial'][operand][lev])
                        partitions[operand][lev][floor_dim].append(
                            list(schedule_info['spatial'][operand][lev][
                                     w.index(schedule_info['flooring'][operand][lev][floor_dim][flooring_type])]))

        operand_partitions = {'I': partitions['I'][level], 'O': partitions['O'][level], 'W': partitions['W'][level]}

        '''
        Given the adopted representation for the unrolling and flooring schemes, the variable is passed to the function that 
        computes the distance that each single variable has to cover in order to reach its position in the array
        '''
        operand_distance = self.get_operand_level_inter_pe_distance(op, operand_partitions,
                                                               schedule_info['temporal'][op][level],
                                                               mem_fifo[op][level])
        operand_distance = np.array(operand_distance)

        '''
        INPUT COST : computed as (distance covered) x (read accesses from level above) x (unit bit wire cost) x (bits
        of precision)
        '''

        if op == 'I':
            if mem_fifo[op][level]:
                array_cost = operand_distance[0] * loop.mem_access_elem['I'][level][0] * mac_array_info[
                    'unit_wire_energy'][level] * \
                             mac_array_info['precision']
            else:
                array_cost = operand_distance[0] * loop.mem_access_elem['I'][level][0] * mac_array_info[
                    'unit_wire_energy'][level] * \
                             mac_array_info['precision']

        '''
        WEIGHT COST : computed as (distance covered) x (read accesses from level above) x (unit bit wire cost) x (bits
        of precision)
        '''
        if op == 'W':
            array_cost = operand_distance[0] * loop.mem_access_elem['W'][level][0] * mac_array_info['unit_wire_energy'][
                level] * \
                         mac_array_info['precision']

        '''
        OUTPUT COST : 
            if PARTIAL OUTPUT: (distance covered) x (read+write accesses from level above) x (unit bit wire cost) x (bits of 
                precision+headroom bits) 
            if FINAL OUTPUT:   (distance covered) x (write accesses to level above) x (unit bit wire cost) x (bits of 
                precision)
        '''
        if op == 'O':
            if loop.mem_access_elem['O_final'][level][0][1] == 0:
                array_cost = operand_distance[0] * sum(map(sum, zip(*loop.mem_access_elem['O_partial'][level]))) * \
                             mac_array_info['unit_wire_energy'][level] * \
                             sum([mac_array_info['precision'], mac_array_info['headroom']])
            else:
                array_cost = operand_distance[0] * loop.mem_access_elem['O_final'][level][0][1] * \
                             mac_array_info['unit_wire_energy'][level] * \
                             mac_array_info['precision']

        return array_cost
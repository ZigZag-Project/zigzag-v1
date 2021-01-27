from copy import deepcopy
from math import prod

class Order(object):
    """
    A class to handle the ordering, consisting of multiple LPF's.
    The ordering will be allocated to W, I and O.
    As this happens sequentially by passing through each MemoryNode at every memory level,
    this class keeps track of what LPFs have been assigned/have not been assigned for each operand.
    """
    def __init__(self, order, spatial_loop, layer, input_settings, n_mem_levels):

        # Relevant loop type numbers for each operand
        relevant_loop_type_numbers = {'W': [1,2,5,6], 'I': [5,7], 'O': [3,4,6,7]}
        pr_loop_type_numbers_I = [1,2,3,4] # 1 = FX, 2 = FY, 3 = OX, 4 = OY
        self.pr_loop_type_numbers_I = pr_loop_type_numbers_I
        self.relevant_loop_type_numbers = relevant_loop_type_numbers
        
        self.order_raw = order
        self.length = len(order)
        self.precision = input_settings.precision
        self.n_mem_levels = n_mem_levels

        self.spatial_loop = spatial_loop
        total_MAC_op = layer.total_MAC_op
        self.total_MAC_op = total_MAC_op

        ### Compute the starting (i.e. without taking into account spatial unrolling at memory levels)
        ### required size/total cycles/number of memory accesses as we assign LPFs from the order
        ## WEIGHT
        size_W = [spatial_loop.su_relevant_size_dict['W'][0]] # total relevant su @ MAC level
        access_W = [total_MAC_op / spatial_loop.su_irrelevant_size_dict['W'][0]]
        total_cycles_W = [1]
        irrelevant_loop_W = [1]
        for (loop_type_number, dimension) in order:
            if loop_type_number in relevant_loop_type_numbers['W']:
                size_W.append(size_W[-1] * dimension)
                access_W.append(access_W[-1])
                irrelevant_loop_W.append(irrelevant_loop_W[-1])
            else:
                size_W.append(size_W[-1])
                access_W.append(access_W[-1] / dimension)
                irrelevant_loop_W.append(irrelevant_loop_W[-1] * dimension)
            total_cycles_W.append(total_cycles_W[-1] * dimension)
        self.size_W = size_W
        self.access_W = access_W
        self.total_cycles_W = total_cycles_W
        self.irrelevant_loop_W = irrelevant_loop_W

        ## OUTPUT
        size_O = [spatial_loop.su_relevant_size_dict['O'][0]]
        access_O = [total_MAC_op / spatial_loop.su_irrelevant_size_dict['O'][0]]
        total_cycles_O = [1]
        irrelevant_loop_O = [1]
        for (loop_type_number, dimension) in order:
            if loop_type_number in relevant_loop_type_numbers['O']:
                size_O.append(size_O[-1] * dimension)
                access_O.append(access_O[-1])
                irrelevant_loop_O.append(irrelevant_loop_O[-1])
            else:
                size_O.append(size_O[-1])
                access_O.append(access_O[-1] / dimension)
                irrelevant_loop_O.append(irrelevant_loop_O[-1] * dimension)
            total_cycles_O.append(total_cycles_O[-1] * dimension)
        self.size_O = size_O
        self.access_O = access_O
        self.total_cycles_O = total_cycles_O
        self.irrelevant_loop_O = irrelevant_loop_O
        # Get index of last ir loop (needed for O precision)
        ir_loop_type_numbers_O = [1,2,5]
        order_ir_O = [x[0] in ir_loop_type_numbers_O for x in order]
        try:
            self.last_ir_index_O = len(order_ir_O) - 1 - order_ir_O[::-1].index(True)
        except:
            self.last_ir_index_O = len(order_ir_O) + 1 # no O ir loop in order
        self.lpfs_seen_O = 0
        self.all_ir_seen_O = False

        ## INPUT
        # For input, a slightly different approach is necessary.
        # If there is spatial unrolling of a pr/r loop at any level, this requires a
        # recalculation of ix, iy, and thus will change the required size/access.
        # At this point, we don't know after which LPF this su will occur,
        # so the only thing we calculate here is for every LPF the total pr and r below (including that LPF).
        su_pr_size_dict_input = spatial_loop.su_pr_size_dict_input
        pr_size_dict_I = {  1: [su_pr_size_dict_input[1][0]], # 1 = FX; Init with su @ MAC level
                            2: [su_pr_size_dict_input[2][0]], # 2 = FY; Init with su @ MAC level
                            3: [su_pr_size_dict_input[3][0]], # 3 = OX; Init with su @ MAC level
                            4: [su_pr_size_dict_input[4][0]]} # 4 = OY; Init with su @ MAC level
        relevant_size_I = [spatial_loop.su_relevant_size_dict['I'][0]] # Init with su @ MAC level
        total_cycles_I = [1]
        MAC_op_I = [prod([loop[1] for loop in spatial_loop.spatial_loop_list])]
        for (loop_type_number, dimension) in order:
            pr_factors = [0, 1, 1, 1, 1] # 0 inserted to match index with pr_loop_type_numbers
            if loop_type_number in pr_loop_type_numbers_I:
                pr_factors[loop_type_number] *= dimension
            pr_size_dict_I[1].append(pr_size_dict_I[1][-1] * pr_factors[1])
            pr_size_dict_I[2].append(pr_size_dict_I[2][-1] * pr_factors[2])
            pr_size_dict_I[3].append(pr_size_dict_I[3][-1] * pr_factors[3])
            pr_size_dict_I[4].append(pr_size_dict_I[4][-1] * pr_factors[4])
            if loop_type_number in relevant_loop_type_numbers['I']:
                relevant_size_I.append(relevant_size_I[-1] * dimension)
            else:
                relevant_size_I.append(relevant_size_I[-1])
            total_cycles_I.append(total_cycles_I[-1] * dimension)
            MAC_op_I.append(MAC_op_I[-1] * dimension)
        self.pr_size_dict_I = pr_size_dict_I
        self.relevant_size_I = relevant_size_I
        self.total_cycles_I = total_cycles_I
        self.MAC_op_I = MAC_op_I
        # self.pr_seen_below = {1: False, 2: False, 3: False, 4: False}
        self.pr_seen_below = {  1: pr_size_dict_I[1][0] != 1, # If there is FX su @ MAC level, this will be True
                                2: pr_size_dict_I[2][0] != 1, # If there is FY su @ MAC level, this will be True
                                3: pr_size_dict_I[3][0] != 1, # If there is OX su @ MAC level, this will be True
                                4: pr_size_dict_I[4][0] != 1} # If there is OY su @ MAC level, this will be True
        # if self.order_raw == ((1, 3), (2, 3), (5, 3), (3, 2), (3, 4), (4, 2), (4, 56)):
        #     print("FOUND")
        #     print(su_pr_size_dict_input)
        #     print(pr_size_dict_I)
        #     print(self.pr_seen_below)
        self.fifo_partner_loop_type_number = {1: 3, 2: 4, 3: 1, 4: 2}
        self.sx = layer.SX
        self.sy = layer.SY
        self.sfx = layer.SFX
        self.sfy = layer.SFY
        self.total_input_data_reuse = layer.total_data_reuse['I']

        ### Initialize the empty allocated orders for all operands
        self.allocated_order_W = []
        self.allocated_order_I = []
        self.allocated_order_O = []

        ### Initialize the remaining lpfs for all operands (first elements represents no LPF added)
        self.remaining_lpf_W = [None] + list(order).copy()
        self.remaining_lpf_I = [None] + list(order).copy()
        self.remaining_lpf_O = [None] + list(order).copy()

        ### Initialize the loop_cycles dict, which will hold the product of all loops within each level
        self.loop_cycles = {'W': [], 'I': [], 'O': []}

        ### Initialize the irrelevant_loop dict, which will hold the product of all ir loops within each level
        ### for input, this will hold the compounded input data reuse
        self.irrelevant_loop = {'W': [1], 'I': [spatial_loop.unit_duplicate['I'][0]], 'O': [1]}


    def allocate_memory(self, node, level):
        '''
        Function to (partially) allocate this order for the given MemoryNode.
        This could be a single-operand, 2-level shared, 3-level shared node.
        Based on which operands the node is shared on, different functions are invoked.
        '''

        # Check operand(s) of node
        operands = node.operand

        # Invoke different functions based on operands of this node
        if operands == ('W',):
            self.allocate_memory_W(node, level)
        elif operands == ('I',):
            self.allocate_memory_I(node, level)
        elif operands == ('O',):
            self.allocate_memory_O(node, level)
        elif all(op in ('W','I') for op in operands):
            self.allocate_memory_WI(node, level)
        elif all(op in ('I','O') for op in operands):
            self.allocate_memory_IO(node, level)
        elif all(op in ('W','O') for op in operands):
            self.allocate_memory_WO(node, level)
        elif all(op in ('W','I','O') for op in operands):
            self.allocate_memory_WIO(node, level)
        else:
            raise ValueError("Operands = {} for Memory Node {}".format(operands, node.memory_level["name"]))


    def allocate_memory_W(self, node, level):
        '''
        Allocate LPFs to this W memory node.
        '''
        
        # Max size (in words) of this node
        max_size = node.memory_level["size_bit"] / self.precision['W']

        # Find the index of the last lpf that fits within this node
        lpf_index = len([x for x in self.size_W if x <= max_size])

        assert lpf_index != 0, "W Node {} can't hold all loops from level below".format(node.memory_level['name'])

        # Take the correct LPFs (up until lpf_index) and allocate to memory
        # Update the remaining LPFs/size/access to exclude all allocated ones
        # except for the last element of size/access:
        # which will represent 'size/access if adding no new LPFs' for next memory level
        self.allocated_order_W.append(self.remaining_lpf_W[1:lpf_index]) # start at 1 to exclude None
        self.remaining_lpf_W = [None] + self.remaining_lpf_W[lpf_index:]
        self.size_W = self.size_W[lpf_index - 1:]
        self.access_W = self.access_W[lpf_index - 1:]

        self.loop_cycles['W'].append(self.total_cycles_W[lpf_index - 1])
        self.total_cycles_W = self.total_cycles_W[lpf_index - 1:]

        self.irrelevant_loop['W'].append(self.irrelevant_loop_W[lpf_index - 1])
        self.irrelevant_loop_W = self.irrelevant_loop_W[lpf_index - 1:]

        # Update remaining size with spatial unrolling of this memory
        su_relevant_factor = self.spatial_loop.su_relevant_size_dict['W'][level + 1]
        self.size_W = [size * su_relevant_factor for size in self.size_W]

        # Take into account spatial unrolling of this node to update access from level above
        su_irrelevant_factor = self.spatial_loop.su_irrelevant_size_dict['W'][level + 1]
        self.access_W = [access / su_irrelevant_factor for access in self.access_W]


        # TODO: keep track of total_cycles, irrelevant loop for lightweight temporal_loop

    def allocate_memory_I(self, node, level):
        '''
        Allocate LPFs to this I memory node.

        The difference with W, O is the calculation of the total input data size:
        ix * iy * c * b
        For ix/iy we need the total pr LPFs
        For c * b we need the total relevant LPF product
        '''

        # Max size (in words) of this node
        max_size_I = node.memory_level["size_bit"] / self.precision['I']

        # To compute the number of accesses from above as we add LPFs,
        # we need the spatial unrolling of this level.
        # For the required size of this node, this is not the case,
        # resulting in two different ix/iy (if pr/relevant su).
        # Right now, we have the pr_size_dict_I and relevant_size_I,
        # which only takes into account already allocated LPFs + SU of lower levels.
        # We create a copy that takes into account this levels spatial unrolling.
        pr_size_copy = deepcopy(self.pr_size_dict_I)
        relevant_size_copy = deepcopy(self.relevant_size_I)
        for pr_loop_type_number in pr_size_copy.keys():
            pr_su_factor = self.spatial_loop.su_pr_size_dict_input[pr_loop_type_number][level + 1]
            pr_size_copy[pr_loop_type_number] = \
                [ x * pr_su_factor for x in pr_size_copy[pr_loop_type_number] ]
        relevant_su_factor = self.spatial_loop.su_relevant_size_dict['I'][level + 1]
        relevant_size_copy = [ x * relevant_su_factor for x in relevant_size_copy]

        # Calculate required size of this memory for each LPF (using original pr/relevant size)
        # as long as the required size is smaller than the maximal size for this node.
        size_I, _, reuse_I = self.get_size_access_reuse_I(max_size_I, pr_size_copy, relevant_size_copy)

        # Get the index of the last LPF that fits within the node. Done here instead of inside for loop
        # because of case if all the remaining LPFs will fit in the node.
        lpf_index = len([x for x in size_I if x <= max_size_I])

        assert lpf_index != 0, "I Node {} can't hold all loops from level below".format(node.memory_level['name'])

        # Take the correct LPFs (up until lpf_index) and allocate to memory
        # Update the remaining LPFs/pr_size_dict_I/relevant_size_I to exclude all allocated ones
        # except for the last element of pr_size_copy/relevant_size_copy:
        # which will represent 'pr_size_dict_I/relevant_size_I if adding no new LPFs' for next memory level
        to_allocate_I = self.remaining_lpf_I[1:lpf_index] # start at 1 to exclude 'None'
        for (lt_number, _) in to_allocate_I:
            if lt_number in self.pr_loop_type_numbers_I:
                self.pr_seen_below[lt_number] = True
        self.allocated_order_I.append(to_allocate_I) 
        self.remaining_lpf_I = [None] + self.remaining_lpf_I[lpf_index:]
        self.pr_size_dict_I = { 1: pr_size_copy[1][lpf_index - 1:], 2: pr_size_copy[2][lpf_index - 1:],
                                3: pr_size_copy[3][lpf_index - 1:], 4: pr_size_copy[4][lpf_index - 1:] }
        self.relevant_size_I = relevant_size_copy[lpf_index - 1:]

        self.loop_cycles['I'].append(self.total_cycles_I[lpf_index - 1])
        self.total_cycles_I = self.total_cycles_I[lpf_index - 1:]

        self.MAC_op_I = self.MAC_op_I[lpf_index - 1:]
        self.irrelevant_loop['I'].append(reuse_I[lpf_index - 1])

    def allocate_memory_O(self, node, level):
        '''
        Allocate LPFs to this O memory node.

        The difference with W is the precision. The order class will keep track if
        the last ir loop was seen for Output
        '''

        # Max size (in words) of this node
        if self.all_ir_seen_O:
            precision = self.precision['O_final']
        else:
            precision = self.precision['O']
        max_size = node.memory_level["size_bit"] / precision

        # Find the index of the last lpf that fits within this node
        lpf_index = len([x for x in self.size_O if x <= max_size])

        assert lpf_index != 0, "O Node {} can't hold all loops from level below".format(node.memory_level['name'])

        # Take the correct LPFs (up until lpf_index) and allocate to memory
        # Update the remaining LPFs/size/access to exclude allocated ones
        to_allocate = self.remaining_lpf_O[1:lpf_index]
        self.lpfs_seen_O += len(to_allocate)
        self.allocated_order_O.append(to_allocate)
        self.remaining_lpf_O = [None] + self.remaining_lpf_O[lpf_index:]
        self.size_O = self.size_O[lpf_index - 1:]
        self.access_O = self.access_O[lpf_index - 1:]

        self.loop_cycles['O'].append(self.total_cycles_O[lpf_index - 1])
        self.total_cycles_O = self.total_cycles_O[lpf_index - 1:]

        self.irrelevant_loop['O'].append(self.irrelevant_loop_O[lpf_index - 1])
        self.irrelevant_loop_O = self.irrelevant_loop_O[lpf_index - 1:]

        # Update remaining size with spatial unrolling of this memory
        su_relevant_factor = self.spatial_loop.su_relevant_size_dict['O'][level + 1]
        self.size_O = [size * su_relevant_factor for size in self.size_O]

        # Take into account spatial unrolling of this node to update access to level above
        # NOTE: If shared-operand node, this should be taken into account at start of function
        su_irrelevant_factor = self.spatial_loop.su_irrelevant_size_dict['O'][level + 1]
        self.access_O = [access / su_irrelevant_factor for access in self.access_O]


        # TODO: Check if last ir loop included in this node
        if self.lpfs_seen_O > self.last_ir_index_O:
            self.all_ir_seen_O = True

    def allocate_memory_WI(self, node, level):
        '''
        Allocate LPFs to this W-I shared memory node.

            - Get the max LPFs from W that would fit in this node + the accesses to above memory for each LPF.
            - Get the max LPFs from I that would fit in this node + the accesses to above memory for each LPF.
            - Try all the combinations to fit in this node (# combinations = # LPF_W * # LPF_I)
            - Allocate the combination that yields the lowest memory access cost to above memory that fits size.
        '''

        # Max size (in words) of this node for W and I
        precision_W = self.precision['W']
        precision_I = self.precision['I']
        node_size_bits = node.memory_level["size_bit"]
        en_per_access_W = node.read_from_above_cost['W']
        en_per_access_I = node.read_from_above_cost['I']
        max_size_W = node_size_bits / precision_W
        max_size_I = node_size_bits / precision_I

        # Take into account spatial unrolling of this node for W access from level above
        su_irrelevant_factor = self.spatial_loop.su_irrelevant_size_dict['W'][level + 1]
        self.access_W = [access / su_irrelevant_factor for access in self.access_W]

        # Take into account spatial unrolling of this node for I access from level above
        pr_size_copy = deepcopy(self.pr_size_dict_I)
        relevant_size_copy = deepcopy(self.relevant_size_I)
        for pr_loop_type_number in pr_size_copy.keys():
            pr_su_factor = self.spatial_loop.su_pr_size_dict_input[pr_loop_type_number][level + 1]
            pr_size_copy[pr_loop_type_number] = \
                [ x * pr_su_factor for x in pr_size_copy[pr_loop_type_number] ]
        relevant_su_factor = self.spatial_loop.su_relevant_size_dict['I'][level + 1]
        relevant_size_copy = [ x * relevant_su_factor for x in relevant_size_copy]

        # Get the size and access for the remaining LPFs of W
        size_W = [x for x in self.size_W if x <= max_size_W]
        access_W = self.access_W[:len(size_W)]            

        # Get the size and access for the remaining LPFs of I
        size_I, access_I, reuse_I = self.get_size_access_reuse_I(max_size_I, pr_size_copy, relevant_size_copy)  

        # Convert size_W and size_I to size in bits as operands might use different precision
        size_bit_W = [x * precision_W for x in size_W]
        size_bit_I = [x * precision_I for x in size_I]  

        # Convert access_W and access_I to energy cost
        access_en_W = [x * en_per_access_W for x in access_W]
        access_en_I = [x * en_per_access_I for x in access_I]
        
        # Calculate the optimal combination of LPFs for both operands.
        idx_W, idx_I = self.get_optimal_lpf_combination_dual(size_bit_W, access_en_W, size_bit_I, access_en_I, node_size_bits)

        # Update W attributes
        self.allocated_order_W.append(self.remaining_lpf_W[1:idx_W + 1]) # start at 1 to exclude 'None'
        self.remaining_lpf_W = [None] + self.remaining_lpf_W[idx_W + 1:]
        self.size_W = self.size_W[idx_W:]
        self.access_W = self.access_W[idx_W:]
        self.loop_cycles['W'].append(self.total_cycles_W[idx_W])
        self.total_cycles_W = self.total_cycles_W[idx_W:]
        self.irrelevant_loop['W'].append(self.irrelevant_loop_W[idx_W])
        self.irrelevant_loop_W = self.irrelevant_loop_W[idx_W:]
        # Update remaining size with spatial unrolling of this memory
        su_relevant_factor = self.spatial_loop.su_relevant_size_dict['W'][level + 1]
        self.size_W = [x * su_relevant_factor for x in self.size_W]

        # Update I attributes
        to_allocate_I = self.remaining_lpf_I[1:idx_I + 1] # start at 1 to exclude 'None'
        for (lt_number, _) in to_allocate_I:
            if lt_number in self.pr_loop_type_numbers_I:
                self.pr_seen_below[lt_number] = True
        self.allocated_order_I.append(to_allocate_I) 
        self.remaining_lpf_I = [None] + self.remaining_lpf_I[idx_I + 1:]
        self.pr_size_dict_I = { 1: pr_size_copy[1][idx_I:], 2: pr_size_copy[2][idx_I:],
                                3: pr_size_copy[3][idx_I:], 4: pr_size_copy[4][idx_I:]}
        self.relevant_size_I = relevant_size_copy[idx_I:]
        self.loop_cycles['I'].append(self.total_cycles_I[idx_I])
        self.total_cycles_I = self.total_cycles_I[idx_I:]
        self.MAC_op_I = self.MAC_op_I[idx_I:]
        self.irrelevant_loop['I'].append(reuse_I[idx_I])


    def allocate_memory_IO(self, node, level):
        '''
        Allocate LPFs to this I-O shared memory node.

            - Get the max LPFs from I that would fit in this node + the accesses to above memory for each LPF.
            - Get the max LPFs from O that would fit in this node + the accesses to above memory for each LPF.
            - Try all the combinations to fit in this node (# combinations = # LPF_I * # LPF_O)
            - Allocate the combination that yields the lowest memory access cost to above memory that fits size.
        '''

        # Max size (in words) of this node for I and O
        precision_I = self.precision['I']
        if self.all_ir_seen_O:
            precision_O = self.precision['O_final']
        else:
            precision_O = self.precision['O']
        node_size_bits = node.memory_level["size_bit"]
        en_per_access_I = node.read_from_above_cost['I']
        # For output energy per access, we take the combined energy of writing to above and reading back in
        # This leads to a slight over-estimation of energy cost, as # reads = # writes - # output elements
        # but not taking this into account will yield the same minimum, while requiring less work.
        en_per_access_O = node.read_from_above_cost['O'] + node.write_to_above_cost['O']
        max_size_I = node_size_bits / precision_I
        max_size_O = node_size_bits / precision_O

        # Take into account spatial unrolling of this node for I access from level above
        pr_size_copy = deepcopy(self.pr_size_dict_I)
        relevant_size_copy = deepcopy(self.relevant_size_I)
        for pr_loop_type_number in pr_size_copy.keys():
            pr_su_factor = self.spatial_loop.su_pr_size_dict_input[pr_loop_type_number][level + 1]
            pr_size_copy[pr_loop_type_number] = \
                [ x * pr_su_factor for x in pr_size_copy[pr_loop_type_number] ]
        relevant_su_factor = self.spatial_loop.su_relevant_size_dict['I'][level + 1]
        relevant_size_copy = [ x * relevant_su_factor for x in relevant_size_copy]

        # Take into account spatial unrolling of this node for O access to/from level above
        su_irrelevant_factor = self.spatial_loop.su_irrelevant_size_dict['O'][level + 1]
        self.access_O = [access / su_irrelevant_factor for access in self.access_O]

        # Get the size and access for the remaining LPFs of I
        size_I, access_I, reuse_I = self.get_size_access_reuse_I(max_size_I, pr_size_copy, relevant_size_copy)

        # Get the size and access for the remaining LPFs of O
        size_O = [x for x in self.size_O if x <= max_size_O]
        access_O = self.access_O[:len(size_O)]

        # Convert size_I and size_O to size in bits as operands might use different precision
        size_bit_I = [x * precision_I for x in size_I]
        size_bit_O = [x * precision_O for x in size_O]

        # Convert access_I and access_O to energy cost
        access_en_I = [x * en_per_access_I for x in access_I]
        access_en_O = [x * en_per_access_O for x in access_O]

        # Calculate the optimal combination of LPFs for both operands
        idx_I, idx_O = self.get_optimal_lpf_combination_dual(size_bit_I, access_en_I, size_bit_O, access_en_O, node_size_bits)

        # Update I attributes
        to_allocate_I = self.remaining_lpf_I[1:idx_I + 1] # start at 1 to exclude 'None'
        for (lt_number, _) in to_allocate_I:
            if lt_number in self.pr_loop_type_numbers_I:
                self.pr_seen_below[lt_number] = True
        self.allocated_order_I.append(to_allocate_I) # start at 1 to excluded 'None'
        self.remaining_lpf_I = [None] + self.remaining_lpf_I[idx_I + 1:]
        self.pr_size_dict_I = { 1: pr_size_copy[1][idx_I:], 2: pr_size_copy[2][idx_I:],
                                3: pr_size_copy[3][idx_I:], 4: pr_size_copy[4][idx_I:]}
        self.relevant_size_I = relevant_size_copy[idx_I:]
        self.loop_cycles['I'].append(self.total_cycles_I[idx_I])
        self.total_cycles_I = self.total_cycles_I[idx_I:]
        self.MAC_op_I = self.MAC_op_I[idx_I:]
        self.irrelevant_loop['I'].append(reuse_I[idx_I])

        # Update O attributes
        to_allocate_O = self.remaining_lpf_O[1:idx_O + 1]
        self.lpfs_seen_O += len(to_allocate_O)
        self.allocated_order_O.append(to_allocate_O)
        self.remaining_lpf_O = [None] + self.remaining_lpf_O[idx_O + 1:]
        self.size_O = self.size_O[idx_O:]
        self.access_O = self.access_O[idx_O:]
        self.loop_cycles['O'].append(self.total_cycles_O[idx_O])
        self.total_cycles_O = self.total_cycles_O[idx_O:]
        self.irrelevant_loop['O'].append(self.irrelevant_loop_O[idx_O])
        self.irrelevant_loop_O = self.irrelevant_loop_O[idx_O:]
        # Update remaining size with spatial unrolling of this memory
        su_relevant_factor = self.spatial_loop.su_relevant_size_dict['O'][level + 1]
        self.size_O = [x * su_relevant_factor for x in self.size_O]
        # Check if last ir loop included in this node
        if self.lpfs_seen_O > self.last_ir_index_O:
            self.all_ir_seen_O = True

        

    def allocate_memory_WO(self, node, level):
        '''
        Allocate LPFs to this W-O shared memory node.

            - Get the max LPFs from W that would fit in this node + the accesses to above memory for each LPF.
            - Get the max LPFs from O that would fit in this node + the accesses to above memory for each LPF.
            - Try all the combinations to fit in this node (# combinations = # LPF_W * # LPF_O)
            - Allocate the combination that yields the lowest memory access cost to above memory that fits size.
        '''

        # Max size (in words) of this node for W and O
        precision_W = self.precision['W']
        if self.all_ir_seen_O:
            precision_O = self.precision['O_final']
        else:
            precision_O = self.precision['O']
        node_size_bits = node.memory_level["size_bit"]
        en_per_access_W = node.read_from_above_cost['W']
        en_per_access_O = node.read_from_above_cost['O'] + node.write_to_above_cost['O']
        max_size_W = node_size_bits / precision_W
        max_size_O = node_size_bits / precision_O

        # Take into account spatial unrolling of this node for W access from level above
        su_irrelevant_factor_W = self.spatial_loop.su_irrelevant_size_dict['W'][level + 1]
        self.access_W = [access / su_irrelevant_factor_W for access in self.access_W]

        # Take into account spatial unrolling of this node for O access to/from level above
        su_irrelevant_factor_O = self.spatial_loop.su_irrelevant_size_dict['O'][level + 1]
        self.access_O = [access / su_irrelevant_factor_O for access in self.access_O]

        # Get the size and access for the remaining LPFs of W
        size_W = [x for x in self.size_W if x <= max_size_W]
        access_W = self.access_W[:len(size_W)]

        # Get the size and access for the remaining LPFs of O
        size_O = [x for x in self.size_O if x <= max_size_O]
        access_O = self.access_O[:len(size_O)]

        # Convert size_W and size_O to size in bits as operands might use different precision
        size_bit_W = [x * precision_W for x in size_W]
        size_bit_O = [x * precision_O for x in size_O]

        # Convert access_W and access_O to energy cost
        access_en_W = [x * en_per_access_W for x in access_W]
        access_en_O = [x * en_per_access_O for x in access_O]

        # Calculate the optimal combination of LPFs for both operands
        idx_W, idx_O = self.get_optimal_lpf_combination_dual(size_bit_W, access_en_W, size_bit_O, access_en_O, node_size_bits)

        # Update W attributes
        self.allocated_order_W.append(self.remaining_lpf_W[1:idx_W + 1]) # start at 1 to exclude 'None'
        self.remaining_lpf_W = [None] + self.remaining_lpf_W[idx_W + 1:]
        self.size_W = self.size_W[idx_W:]
        self.access_W = self.access_W[idx_W:]
        self.loop_cycles['W'].append(self.total_cycles_W[idx_W])
        self.total_cycles_W = self.total_cycles_W[idx_W:]
        self.irrelevant_loop['W'].append(self.irrelevant_loop_W[idx_W])
        self.irrelevant_loop_W = self.irrelevant_loop_W[idx_W:]
        # Update remaining size with spatial unrolling of this memory
        su_relevant_factor = self.spatial_loop.su_relevant_size_dict['W'][level + 1]
        self.size_W = [x * su_relevant_factor for x in self.size_W]

        # Update O attributes
        to_allocate_O = self.remaining_lpf_O[1:idx_O + 1]
        self.lpfs_seen_O += len(to_allocate_O)
        self.allocated_order_O.append(to_allocate_O)
        self.remaining_lpf_O = [None] + self.remaining_lpf_O[idx_O + 1:]
        self.size_O = self.size_O[idx_O:]
        self.access_O = self.access_O[idx_O:]
        self.loop_cycles['O'].append(self.total_cycles_O[idx_O])
        self.total_cycles_O = self.total_cycles_O[idx_O:]
        self.irrelevant_loop['O'].append(self.irrelevant_loop_O[idx_O])
        self.irrelevant_loop_O = self.irrelevant_loop_O[idx_O:]
        # Update remaining size with spatial unrolling of this memory
        su_relevant_factor = self.spatial_loop.su_relevant_size_dict['O'][level + 1]
        self.size_O = [x * su_relevant_factor for x in self.size_O]
        # Check if last ir loop included in this node
        if self.lpfs_seen_O > self.last_ir_index_O:
            self.all_ir_seen_O = True

    def allocate_memory_WIO(self, node, level):
        '''
        Allocate LPFs to this W-I-O shared memory node.

            - Get the max LPFs from W that would fit in this node + the accesses to above memory for each LPF.
            - Get the max LPFs from I that would fit in this node + the accesses to above memory for each LPF.
            - Get the max LPFs from O that would fit in this node + the accesses to above memory for each LPF.
            - Try all the combinations to fit in this node (# combinations = # LPF_W * # LPF_I * # LPF_O)
            - Allocate the combination that yields the lowest memory access cost to above memory that fits size.
        '''

        # Max size (in words) of this node for W, I and O
        precision_W = self.precision['W']
        precision_I = self.precision['I']
        if self.all_ir_seen_O:
            precision_O = self.precision['O_final']
        else:
            precision_O = self.precision['O']
        node_size_bits = node.memory_level["size_bit"]
        en_per_access_W = node.read_from_above_cost['W']
        en_per_access_I = node.read_from_above_cost['I']
        en_per_access_O = node.read_from_above_cost['O'] + node.write_to_above_cost['O']
        max_size_W = node_size_bits / precision_W
        max_size_I = node_size_bits / precision_I
        max_size_O = node_size_bits / precision_O

        # Take into account spatial unrolling of this node for W access from level above
        su_irrelevant_factor_W = self.spatial_loop.su_irrelevant_size_dict['W'][level + 1]
        self.access_W = [access / su_irrelevant_factor_W for access in self.access_W]

        # Take into account spatial unrolling of this node for I access from level above
        pr_size_copy = deepcopy(self.pr_size_dict_I)
        relevant_size_copy = deepcopy(self.relevant_size_I)
        for pr_loop_type_number in pr_size_copy.keys():
            pr_su_factor = self.spatial_loop.su_pr_size_dict_input[pr_loop_type_number][level + 1]
            pr_size_copy[pr_loop_type_number] = \
                [ x * pr_su_factor for x in pr_size_copy[pr_loop_type_number] ]
        relevant_su_factor = self.spatial_loop.su_relevant_size_dict['I'][level + 1]
        relevant_size_copy = [ x * relevant_su_factor for x in relevant_size_copy]

        # Take into account spatial unrolling of this node for O access to/from level above
        su_irrelevant_factor_O = self.spatial_loop.su_irrelevant_size_dict['O'][level + 1]
        self.access_O = [access / su_irrelevant_factor_O for access in self.access_O]

        # Get the size and access for the remaining LPFs of W
        size_W = [x for x in self.size_W if x <= max_size_W]
        access_W = self.access_W[:len(size_W)]

        # Get the size and access for the remaining LPFs of I
        size_I, access_I, reuse_I = self.get_size_access_reuse_I(max_size_I, pr_size_copy, relevant_size_copy)

        # Get the size and access for the remaining LPFs of O
        size_O = [x for x in self.size_O if x <= max_size_O]
        access_O = self.access_O[:len(size_O)]

        # Convert size_W, size_I and size_O to size in bits as operands might use different precision
        size_bit_W = [x * precision_W for x in size_W]
        size_bit_I = [x * precision_I for x in size_I]
        size_bit_O = [x * precision_O for x in size_O]

        # Convert access_W, access_I and access_O to energy cost
        access_en_W = [x * en_per_access_W for x in access_W]
        access_en_I = [x * en_per_access_I for x in access_I]
        access_en_O = [x * en_per_access_O for x in access_O]

        # Calculate the optimal combination of LPFs for three oprands
        idx_W, idx_I, idx_O = self.get_optimal_lpf_combination_triple(size_bit_W, access_en_W,
                                                                    size_bit_I, access_en_I,
                                                                    size_bit_O, access_en_O, node_size_bits)
        
        # Update W attributes
        self.allocated_order_W.append(self.remaining_lpf_W[1:idx_W + 1]) # start at 1 to exclude 'None'
        self.remaining_lpf_W = [None] + self.remaining_lpf_W[idx_W + 1:]
        self.size_W = self.size_W[idx_W:]
        self.access_W = self.access_W[idx_W:]
        self.loop_cycles['W'].append(self.total_cycles_W[idx_W])
        self.total_cycles_W = self.total_cycles_W[idx_W:]
        self.irrelevant_loop['W'].append(self.irrelevant_loop_W[idx_W])
        self.irrelevant_loop_W = self.irrelevant_loop_W[idx_W:]
        # Update remaining size with spatial unrolling of this memory
        su_relevant_factor = self.spatial_loop.su_relevant_size_dict['W'][level + 1]
        self.size_W = [x * su_relevant_factor for x in self.size_W]

        # Update I attributes
        to_allocate_I = self.remaining_lpf_I[1:idx_I + 1] # start at 1 to exclude 'None'
        for (lt_number, _) in to_allocate_I:
            if lt_number in self.pr_loop_type_numbers_I:
                self.pr_seen_below[lt_number] = True
        self.allocated_order_I.append(to_allocate_I) # start at 1 to excluded 'None'
        self.remaining_lpf_I = [None] + self.remaining_lpf_I[idx_I + 1:]
        self.pr_size_dict_I = { 1: pr_size_copy[1][idx_I:], 2: pr_size_copy[2][idx_I:],
                                3: pr_size_copy[3][idx_I:], 4: pr_size_copy[4][idx_I:]}
        self.relevant_size_I = relevant_size_copy[idx_I:]
        self.loop_cycles['I'].append(self.total_cycles_I[idx_I])
        self.total_cycles_I = self.total_cycles_I[idx_I:]
        self.MAC_op_I = self.MAC_op_I[idx_I:]
        self.irrelevant_loop['I'].append(reuse_I[idx_I])

        # Update O attributes
        to_allocate_O = self.remaining_lpf_O[1:idx_O + 1]
        self.lpfs_seen_O += len(to_allocate_O)
        self.allocated_order_O.append(to_allocate_O)
        self.remaining_lpf_O = [None] + self.remaining_lpf_O[idx_O + 1:]
        self.size_O = self.size_O[idx_O:]
        self.access_O = self.access_O[idx_O:]
        self.loop_cycles['O'].append(self.total_cycles_O[idx_O])
        self.total_cycles_O = self.total_cycles_O[idx_O:]
        self.irrelevant_loop['O'].append(self.irrelevant_loop_O[idx_O])
        self.irrelevant_loop_O = self.irrelevant_loop_O[idx_O:]
        # Update remaining size with spatial unrolling of this memory
        su_relevant_factor = self.spatial_loop.su_relevant_size_dict['O'][level + 1]
        self.size_O = [x * su_relevant_factor for x in self.size_O]
        # Check if last ir loop included in this node
        if self.lpfs_seen_O > self.last_ir_index_O:
            self.all_ir_seen_O = True

    def allocate_remaining(self):
        '''
        Allocate the remaining LPFs for all three operands.
        Set the complete allocated_order attribute containing the three operands.
        '''

        self.allocated_order_W.append(self.remaining_lpf_W[1:]) # start at 1 to exclude 'None'
        self.allocated_order_I.append(self.remaining_lpf_I[1:]) # start at 1 to exclude 'None'
        self.allocated_order_O.append(self.remaining_lpf_O[1:]) # start at 1 to exclude 'None'

        self.loop_cycles['W'].append(self.total_cycles_W[-1])
        self.loop_cycles['I'].append(self.total_cycles_I[-1])
        self.loop_cycles['O'].append(self.total_cycles_O[-1])

        self.irrelevant_loop['W'].append(self.irrelevant_loop_W[-1])
        self.irrelevant_loop['I'].append(self.total_input_data_reuse)
        self.irrelevant_loop['O'].append(self.irrelevant_loop_O[-1])
        
        # Convert this compounded irrelevant_loop to level-wise irrelevant_loop
        irrelevant_loop = {'W': [], 'I': [], 'O': []}
        for i in range(1, len(self.irrelevant_loop['W'])):
            irrelevant_loop['W'].append(self.irrelevant_loop['W'][i] / self.irrelevant_loop['W'][i-1]) 
        for i in range(1, len(self.irrelevant_loop['I'])):
            irrelevant_loop['I'].append(self.irrelevant_loop['I'][i] / self.irrelevant_loop['I'][i-1])
        for i in range(1, len(self.irrelevant_loop['O'])):
            irrelevant_loop['O'].append(self.irrelevant_loop['O'][i] / self.irrelevant_loop['O'][i-1])
        self.irrelevant_loop = irrelevant_loop
        # if self.order_raw == ((1, 3), (2, 3), (5, 3), (3, 2), (3, 4), (4, 2), (4, 56)):
        #     print("irrelevent loop I=", self.irrelevant_loop['I'])
        

        # NOTE: The self.size_X, self.access_X and other attributes won't be updated as no longer required.
        # Thus, beware of using them past this point.

        self.allocated_order = {'W': self.allocated_order_W, 'I': self.allocated_order_I, 'O': self.allocated_order_O}
    
        return self.allocated_order

    def get_size_access_reuse_I(self, max_size_I, pr_size_copy, relevant_size_copy):
        '''
        Get the size, access and reuse for the remaining LPFs of I

        for size calculation we need self.pr_size_dict_I/self.relevant_size_I
        for access calculation we need pr_size_copy/relevant_size_copy
        for access we also need to take the fifo effect into account
        '''

        n_lpfs = len(self.remaining_lpf_I)
        size_I = []
        access_I = []
        reuse_I = []
        for i in range(n_lpfs):
            input_data_size = self.calc_input_data_size(self.pr_size_dict_I[1][i], self.pr_size_dict_I[2][i], 
                                                        self.pr_size_dict_I[3][i], self.pr_size_dict_I[4][i], 
                                                        self.relevant_size_I[i])
            if input_data_size > max_size_I:
                break
            else:
                size_I.append(input_data_size)
                n_MAC = self.MAC_op_I[i] 
                if i == 0: # No LPF added, accesses from above should be the same as for previous level
                    input_data_access = self.total_MAC_op / input_data_size
                    # Recalculate input_data_size for data reuse as this levels su can influence data size
                    input_data_size_reuse = self.calc_input_data_size(pr_size_copy[1][0], pr_size_copy[2][0],
                                                                    pr_size_copy[3][0], pr_size_copy[4][0], 
                                                                    relevant_size_copy[0])
                    input_data_reuse = n_MAC / input_data_size_reuse
                else: # FIFO effect possible, also chain FIFO possible. Example: OX | FX OX FX OX
                    lt_number = self.remaining_lpf_I[i][0]
                    if lt_number in self.pr_loop_type_numbers_I:
                        self.pr_seen_below[lt_number] = True

                    try:
                        fifo_lt_number = self.remaining_lpf_I[i+1][0]
                        fifo_partner_lt_number = self.fifo_partner_loop_type_number[fifo_lt_number]
                        if self.pr_seen_below[fifo_partner_lt_number]:
                            fifo_effect_occurs = True
                        else:
                            fifo_effect_occurs = False
                    except: # triggered for i == n_lpfs - 1
                        fifo_effect_occurs = False # FIFO effect can't happen if all LPFs are added
                    fx = pr_size_copy[1][i]
                    fy = pr_size_copy[2][i]
                    ox = pr_size_copy[3][i]
                    oy = pr_size_copy[4][i]
                    pr_sizes = [0, fx, fy, ox, oy] # 0 inserted so index matches lt_number
                    if fifo_effect_occurs:
                        next_lpf_i = i + 1
                        # Check for chain effect
                        while next_lpf_i < n_lpfs:
                            (next_lt_number, next_dimension) = self.remaining_lpf_I[next_lpf_i]
                            if next_lt_number == fifo_lt_number or next_lt_number == fifo_partner_lt_number:
                                pr_sizes[next_lt_number] *= next_dimension
                                n_MAC *= next_dimension
                                next_lpf_i += 1
                            else:
                                break
                    
                    input_data_size_reuse = self.calc_input_data_size(pr_sizes[1], pr_sizes[2],
                                                                    pr_sizes[3], pr_sizes[4],
                                                                    relevant_size_copy[i], fifo=fifo_effect_occurs)                                              
                    input_data_access = self.total_MAC_op / input_data_size_reuse
                    input_data_reuse = n_MAC / input_data_size_reuse

                    # if self.order_raw == ((3, 2), (4, 7), (1, 11), (3, 2), (4, 2), (4, 4), (2, 11), (5, 3), (6, 6)):
                    #     print(i)
                    #     print(n_MAC)
                    #     print(pr_sizes, relevant_size_copy[i])
                    #     print(input_data_reuse)
                    #     print()

                access_I.append(input_data_access)
                reuse_I.append(input_data_reuse)

        return size_I, access_I, reuse_I


    def calc_input_data_size(self, fx, fy, ox, oy, cb, fifo=False):
        '''
        Function to calculate the total input data size given the loop dimensions

        Arguments
        =========
        - bc: The product of the B and C loop type dimensions
        - fifo: boolean to cope with different handling of stride when fifo effect occurs.
                TemporalLoop always uses stride for both IX and IY if fifo effect occured.
                This is probably wrong, and should be changed, but done here to stay consistent with TemporalLoop
        '''
        if fifo:
            ix = self.sx * (ox - 1) + self.sfx * (fx - 1) + 1
            iy = self.sy * (oy - 1) + self.sfy * (fy - 1) + 1
        else:
            if ox == 1 or fx == 1:
                ix = ox + fx - 1
                interleaved_storage_ix = True
            else:
                ix = self.sx * (ox - 1) + self.sfx * (fx - 1) + 1
                interleaved_storage_ix = False
            if oy == 1 or fy == 1:
                iy = oy + fy - 1
                interleaved_storage_iy = True
            else:
                iy = self.sy * (oy - 1) + self.sfy * (fy - 1) + 1
                interleaved_storage_iy = False
        
        input_data_size = ix * iy * cb

        return input_data_size


    @staticmethod
    def get_optimal_lpf_combination_dual(size_1, en_1, size_2, en_2, node_size):
        '''
        Method to get the indexes of the optimal LPF combination for a dual operand shared memory.
        '''

        # Iterate through each combination. 
        # Save the indexes of the minimal combination that fits size constraint.
        n_1 = len(size_1)
        n_2 = len(size_2)
        en_min = float('inf')
        i_min = None
        j_min = None
        for i in range(n_1):
            for j in range(n_2):
                size_comb = size_1[i] + size_2[j]
                if size_comb > node_size:
                    break
                else:
                    en_comb = en_1[i] + en_2[j]
                    if en_comb <= en_min:
                        en_min = en_comb
                        i_min = i
                        j_min = j
        return i_min, j_min

    @staticmethod
    def get_optimal_lpf_combination_triple(size_1, en_1, size_2, en_2, size_3, en_3, node_size):
        '''
        Method to get the indexes of the optimal LPF combination for a triple operand shared memory.
        '''

        # Iterate through each combination. 
        # Save the indexes of minmal combination that fits size constraint.
        n_1 = len(size_1)
        n_2 = len(size_2)
        n_3 = len(size_3)
        en_min = float('inf')
        i_min = None
        j_min = None
        k_min = None
        for i in range(n_1):
            for j in range(n_2):
                for k in range(n_3):
                    size_comb = size_1[i] + size_2[j] + size_3[k]
                    if size_comb > node_size:
                        break
                    else:
                        en_comb = en_1[i] + en_2[j] + en_3[k]
                        if en_comb <= en_min:
                            en_min = en_comb
                            i_min = i
                            j_min = j
                            k_min = k
        return i_min, j_min, k_min
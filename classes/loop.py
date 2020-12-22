"""
Overall loop information integration based on temporal & spatial loop.
"""

import numpy as np
import copy


class Loop(object):

    def __init__(self, layer, temporal_loop, spatial_loop, precision, size_check):
        """
        Loop class provides:

        req_mem_size: # of element stored at each memory level for W/I/O.

        effective_mem_size: effective/minimal # of element that need to be stored at each memory level for W/I/O
        based on memory storage life time analysis.

        req_aver_mem_bw: value of (# of element need to be read from / written to certain level of memory
        per # of MAC clock cycle).

        High level memory could operate at a lower frequency than MAC.
        Note that over memory hierarchy, W/I flow in one direction (from top/big mem to bottom/small mem),
        while O can flow in two directions due to partial sum accumulation.

        req_aver_mem_bw: the average read/write memory bandwidth at each level of memory,
        i.e. number of element that need to be read/written at each level of memory PER number of MAC clock cycle.

        req_aver_mem_bw[operand] = [(#, #), (#, #)], [(#, #), (#, #)] ...
        [(# of element talks to lower level of memory, # of MAC clock cycle), (talks to higher level of memory, ...)]

        data_reuse: data reuse count (both spatial and temporal reuse) for each operand at each level (MAC & memory).
        The product of data_reuse of each level (MAC & memory) for each operand should be equal to the max_data_reuse
        for that operand.

        mem_access_elem: memory access (read & write) at each level. (Unit: # of element)
        mem_access_bit: memory access (read & write) at each level. (Unit: bit)

        mem_access_*[operand] = [(read, write @ mem 0),(read, write @ mem 1), ...]

        req_inst_mem_bw: required instant memory bw to prevent MAC array stall.
        req_inst_mem_bw = {operand: (# of element, # of cycles)}

        When can a level of memory start being written (overwrite its original data) / its above level memory
        start being read?
        -> When a certain block of the memory data are fully used and no longer needed.
        """

        mem_level = temporal_loop.loop_levels

        req_mem_size = {'W': [1] * mem_level['W'],
                        'I': [1] * mem_level['I'],
                        'O': [1] * mem_level['O'],
                        'O_final': [],
                        'O_partial': []}

        req_mem_count = {'W': [1] * mem_level['W'],
                         'I': [1] * mem_level['I'],
                         'O': [1] * mem_level['O'],
                         'O_final': [],
                         'O_partial': []}

        ''' 
        req_mem_size (unit: # of element) is calculated by multiplying all the relevant 
        temporal loop index values from the lowest mem level to the current level and 
        all the relevant spatial loop index values from the lowest mem level to the 
        (current - 1) level.
        
        req_mem_count is the unit_count in spatial loop.

        Using ().item() to change result datatype from numpy int64 to python default int.
        '''

        for level in range(mem_level['W']):
            ''' 
            Be careful of  Python intrinsic Negative Indexing.
            
            Also note that list[0:0] = [], list[0:1] = list[0].
            
            All the spatial_loop indexes (both begin and end) +1 to align mem level with temporal_loop 
            because of the additional innermost MAC level in spatial loops.
            E.g. memory level 0 is corresponding to the 0th level in temporal loops 
            but the 1st level in spatial loops.
            '''
            req_mem_size['W'][level] = int((np.prod(
                temporal_loop.K['W'][0:level + 1] + temporal_loop.C['W'][0:level + 1] +
                temporal_loop.FY['W'][0:level + 1] + temporal_loop.FX['W'][0:level + 1] +
                spatial_loop.Ku['W'][0:level + 1] + spatial_loop.Cu['W'][0:level + 1] +
                spatial_loop.FYu['W'][0:level + 1] + spatial_loop.FXu['W'][0:level + 1])
                                           ).item())

            req_mem_count['W'][level] = spatial_loop.unit_count['W'][level + 1]

        for level in range(mem_level['I']):
            '''
            For input size calculation, 
            temporal parts (IY/IX) need to take stride (SY/SX/SFY/SFX) into consideration, while
            spatial parts (IYu/IXu) don't need to take stride (SY/SX/SFY/SFX) into consideration.
            '''
            if temporal_loop.interleaved_storage_IY[level]:
                IY = int(((np.prod(temporal_loop.OY['I'][0:level + 1] + spatial_loop.OYu['I'][0:level + 1])).item() - 1) +
                         ((np.prod(temporal_loop.FY['I'][0:level + 1] + spatial_loop.FYu['I'][0:level + 1])).item() - 1) + 1)
            else:
                IY = int(layer.SY * ((np.prod(temporal_loop.OY['I'][0:level + 1] + spatial_loop.OYu['I'][0:level + 1])).item() - 1) +
                         layer.SFY * ((np.prod(temporal_loop.FY['I'][0:level + 1] + spatial_loop.FYu['I'][0:level + 1])).item() - 1) + 1)

            if temporal_loop.interleaved_storage_IX[level]:
                IX = int(((np.prod(temporal_loop.OX['I'][0:level + 1] + spatial_loop.OXu['I'][0:level + 1])).item() - 1) +
                         ((np.prod(temporal_loop.FX['I'][0:level + 1] + spatial_loop.FXu['I'][0:level + 1])).item() - 1) + 1)
            else:
                IX = int(layer.SX * ((np.prod(temporal_loop.OX['I'][0:level + 1] + spatial_loop.OXu['I'][0:level + 1])).item() - 1) +
                         layer.SFX * ((np.prod(temporal_loop.FX['I'][0:level + 1] + spatial_loop.FXu['I'][0:level + 1])).item() - 1) + 1)

            req_mem_size['I'][level] = int((np.prod(
                temporal_loop.B['I'][0:level + 1] + temporal_loop.C['I'][0:level + 1] + [IY] + [IX] +
                spatial_loop.Bu['I'][0:level + 1] + spatial_loop.Cu['I'][0:level + 1])).item())

            req_mem_count['I'][level] = spatial_loop.unit_count['I'][level + 1]

        for level in range(mem_level['O']):
            req_mem_size['O'][level] = int((np.prod(
                temporal_loop.B['O'][0:level + 1] + temporal_loop.K['O'][0:level + 1] +
                temporal_loop.OX['O'][0:level + 1] + temporal_loop.OY['O'][0:level + 1] +
                spatial_loop.Bu['O'][0:level + 1] + spatial_loop.Ku['O'][0:level + 1] +
                spatial_loop.OYu['O'][0:level + 1] + spatial_loop.OXu['O'][0:level + 1])
                                           ).item())

            req_mem_count['O'][level] = spatial_loop.unit_count['O'][level + 1]

        ''' 
        effective_mem_size (unit: # of element) is calculated by firstly finding the top irrelevant loop 
        at the current memory level, then getting the product of the all the relevant loops below it.
        
        It is equivalent to always merging the top relevant loop to the memory level above.
        
        It is just opposite to stationarity, always merging the bottom irrelevant loop to the memory level below.
        '''

        # TODO Here we assume when calculating the current level effective memory size,
        #  the memory size of levels below it are fixed, which equal to the req_mem_size.

        effective_mem_size = copy.deepcopy(req_mem_size)

        l_type = 0
        l_range = 1
        for level, loops in enumerate(temporal_loop.temporal_loop['W']):
            if not loops:
                continue
            else:
                for loop in reversed(loops):
                    if loop[l_type] in [1, 2, 5, 6]:
                        effective_mem_size['W'][level] = req_mem_size['W'][level] / loop[l_range]

                    else:
                        break

        for level, loops in enumerate(temporal_loop.temporal_loop['I']):
            if not loops:
                continue
            else:
                for loop in reversed(loops):
                    if loop[l_type] in [5, 7]:
                        effective_mem_size['I'][level] = req_mem_size['I'][level] / loop[l_range]
                    else:
                        break

        for level, loops in enumerate(temporal_loop.temporal_loop['O']):
            if not loops:
                continue
            else:
                for loop in reversed(loops):
                    if loop[l_type] in [3, 4, 6, 7]:
                        effective_mem_size['O'][level] = req_mem_size['O'][level] / loop[l_range]
                    else:
                        break

        ''' 
        top_ir: For each memory level, from top to bottom, the product of top few irrelevant loops.
        top_ir is used for later required instant memory bandwidth calculation.
        '''

        # TODO Treating Fx and Fy as ir loops for 'I' is overestimated, i.e. the worst case.

        top_ir = {'W': [1] * mem_level['W'],
                  'I': [1] * mem_level['I'],
                  'O': [1] * mem_level['O']}

        l_type = 0
        l_range = 1
        for level, loops in enumerate(temporal_loop.temporal_loop['W']):
            if not loops:
                continue
            else:
                for loop in reversed(loops):
                    if loop[l_type] in [3, 4, 7]:
                        top_ir['W'][level] *= loop[l_range]
                    else:
                        break

        for level, loops in enumerate(temporal_loop.temporal_loop['I']):
            if not loops:
                continue
            else:
                for loop in reversed(loops):
                    if loop[l_type] in [1, 2, 6]:
                        top_ir['I'][level] *= loop[l_range]
                    else:
                        break

        for level, loops in enumerate(temporal_loop.temporal_loop['O']):
            if not loops:
                continue
            else:
                for loop in reversed(loops):
                    if loop[l_type] in [1, 2, 5]:
                        top_ir['O'][level] *= loop[l_range]
                    else:
                        break

        ''' 
        req_aver_mem_bw: required average memory bandwidth, assuming double buffering at all memory levels.
        req_inst_mem_bw: required instant memory bandwidth, assuming dual-port memory at all levels.
        
        For W and I: below level <- [read bw, write bw] <- above level
        For O:   below level (<-)-> [write bw, read bw] (<-)-> above level (depend on psum)
        '''
        # TODO input stall could be sometimes overestimated
        #  by only taking the baseline fetching pattern into consideration.

        req_aver_mem_bw = {'W': [],
                           'I': [],
                           'O': [],
                           'O_final': [],
                           'O_partial': []}
        req_inst_mem_bw = {'W': [],
                           'I': [],
                           'O': [],
                           'O_final': [],
                           'O_partial': []}

        for operand in ['W', 'I', 'O']:
            for level in range(req_mem_size[operand].__len__()):
                if level == 0:
                    req_aver_mem_bw[operand].append([(spatial_loop.real_bw_boost[operand][0], 1),
                                                     (req_mem_size[operand][0], temporal_loop.loop_cycles[operand][0])])

                    req_inst_mem_bw[operand].append([(spatial_loop.real_bw_boost[operand][0], 1),
                                                     (req_mem_size[operand][0], temporal_loop.loop_cycles[operand][0] /
                                                      top_ir[operand][0])])
                else:
                    req_aver_mem_bw[operand].append([(spatial_loop.real_bw_boost[operand][level] *
                                                      req_mem_size[operand][level - 1],
                                                      temporal_loop.loop_cycles[operand][level - 1]),
                                                     (req_mem_size[operand][level],
                                                      temporal_loop.loop_cycles[operand][level])])

                    req_inst_mem_bw[operand].append([(spatial_loop.real_bw_boost[operand][level] *
                                                      req_mem_size[operand][level - 1],
                                                      temporal_loop.loop_cycles[operand][level - 1] /
                                                      top_ir[operand][level - 1]),
                                                     (req_mem_size[operand][level],
                                                      temporal_loop.loop_cycles[operand][level] /
                                                      top_ir[operand][level])])

        '''
        dt_bloc: temporal loop data & time block,
        i.e. when each temporal loop's index +1, how many element & how many clock cycle are covered.
        (# of elements, # of clock cycles)
        
        Right now, the only use for it is calculating output off-loading cycles in utilization.py
        '''
        dt_bloc = {'W': [],
                   'I': [],
                   'O': []}

        dt_bloc_below = {'W': [],
                         'I': [],
                         'O': []}

        spatial_para = copy.deepcopy(spatial_loop.real_bw_boost['O'])

        l_type = 0
        l_range = 1
        data_bloc = 1
        data_bloc_below = spatial_para[0]
        cycle_bloc = 1
        cycle_bloc_below = 1
        for level, loops in enumerate(temporal_loop.temporal_loop['O']):
            dt_bloc['O'].append([])
            dt_bloc_below['O'].append([])
            if not loops:
                continue
            else:
                for loop in loops:
                    cycle_bloc *= loop[l_range]
                    if loop[l_type] in [3, 4, 6, 7]:
                        # relevant loops contribute to data block size.
                        data_bloc *= loop[l_range] * spatial_para[level]
                    dt_bloc['O'][level].append((data_bloc, cycle_bloc))
                    dt_bloc_below['O'][level].append((data_bloc_below * spatial_para[level], cycle_bloc_below))
                    cycle_bloc_below = cycle_bloc
                    data_bloc_below = data_bloc
                    # reset spatial_para (real_mem_boost) to 1 to make sure only multiply with it once per level.
                    spatial_para[level] = 1

        # data_block = 1
        # cycle_block = 1
        # spatial_para = spatial_loop.real_bw_boost['I']
        # FX_count = 1
        # FY_count = 1
        # OX_count = 1
        # OY_count = 1
        # C_count = 1
        # B_count = 1
        # for level, loops in enumerate(temporal_loop.temporal_loop['I']):
        #     temporal_loop_dt_bloc['I'].append([])
        #     if not loops:
        #         continue
        #     else:
        #         for loop in loops:
        #             cycle_block *= loop[l_range]
        #             if loop[l_type] == 7:
        #                 B_count *= loop[l_range]
        #             elif loop[l_type] == 6:
        #                 continue
        #             elif loop[l_type] == 5:
        #                 C_count *= loop[l_range]
        #             elif loop[l_type] == 4:
        #                 OY_count *= loop[l_range]
        #             elif loop[l_type] == 3:
        #                 OX_count *= loop[l_range]
        #             elif loop[l_type] == 2:
        #                 FY_count *= loop[l_range]
        #             elif loop[l_type] == 1:
        #                 FX_count *= loop[l_range]
        #             data_block = B_count * C_count * (FX_count + OX_count - 1) * (FY_count + OY_count - 1)
        #             temporal_loop_dt_bloc['I'][level].append((data_block, cycle_block))

        '''
        Each operand has its maximum reuse possibility for a workload, which need to be split out 
        upon all levels (MAC level + all memory levels). 
        
        data_reuse is used to capture this split-out reuse count for each operand at each level.
        It is initialized with MAC-level reuse count since it is already been calculated 
        as 0th-level (MAC-level) unit_duplicate in spatial_loop class.
        '''

        data_reuse = {'W': [spatial_loop.unit_duplicate['W'][0]],
                      'I_base': [spatial_loop.unit_duplicate['I'][0]],
                      'I': [spatial_loop.unit_duplicate['I'][0]],
                      'I_zigzag': [spatial_loop.unit_duplicate['I'][0]],
                      'O': [spatial_loop.unit_duplicate['O'][0]]}

        for operand in ['W', 'I_base', 'I', 'O']:
            data_reuse[operand].extend(temporal_loop.irrelevant_loop[operand])

        '''
        mem_access_total_element indicates total # of read/write access at each level of memory.
        Note that output feature map's partial sum has a higher bitwidth, thus is separated.
        
        Bitwidth(O_partial) = Bitwidth(W) + Bitwidth(I) + accumulation headroom.
        
        O_partial ends at the level (and above) which covers all the loops of C, Fx, and Fy 
        (irrelevant loops of output/Sum-Together loops).
        '''

        mem_access_per_element = {'W': [],
                                  'I': [],
                                  'O': []}

        mem_access_total_element = {'W': [],
                                    'I_base': [],
                                    'I': [],
                                    'I_zig_zag': [],
                                    'O': [],
                                    'O_final': [],
                                    'O_partial': []}

        '''
        Memory access data format for weight and input:
        [(Read, Write @ mem 0),(Read, Write @ mem 1), ...]
        '''

        mem_read = {'W': [],
                    'I': [],
                    'I_base': []}

        for operand in ['W', 'I']:
            for level in range(mem_level[operand]):
                mem_read[operand].append(np.prod(
                    temporal_loop.irrelevant_loop[operand][level:mem_level[operand]] +
                    spatial_loop.unit_duplicate[operand][level + 1:mem_level[operand] + 1]
                ).item())

            for level in range(mem_level[operand]):
                if level == mem_level[operand] - 1:
                    ''' 
                    For now, we don't consider the writing access from outside peripherals to the top-level memory. 
                    '''
                    mem_write = 0
                else:
                    mem_write = mem_read[operand][level + 1]

                mem_access_per_element[operand].append([mem_read[operand][level], mem_write])
                mem_access_total_element[operand].append(
                    [int(mem_read[operand][level] * layer.total_data_size[operand]),
                     int(mem_write * layer.total_data_size[operand])])

        # Treat 'I' separately in case im2col need correction
        # for operand in ['I']:
        #     for level in range(mem_level[operand]):
        #         mem_read[operand].append(np.prod(
        #             temporal_loop.irrelevant_loop[operand][level:mem_level[operand]] +
        #             spatial_loop.unit_duplicate[operand][level + 1:mem_level[operand] + 1]
        #         ).item())
        #
        #     for level in range(mem_level[operand]):
        #         if level == mem_level[operand] - 1:
        #             '''
        #             For now, we don't consider the writing access from outside peripherals to the top-level memory.
        #             '''
        #             mem_write = 0
        #         else:
        #             mem_write = mem_read[operand][level + 1]
        #
        #         mem_access_per_element[operand].append([mem_read[operand][level], mem_write])
        #         mem_access_total_element[operand].append(
        #             [int(mem_read[operand][level] * layer_for_compute.total_data_size[operand]),
        #              int(mem_write * layer_for_compute.total_data_size[operand])])
        #
        #     if im2col_need_correct:
        #         mem_access_total_element['I'] = im2col_mem_access_correction(
        #             layer_for_correct, layer_for_compute, mem_access_total_element['I'], temporal_loop, spatial_loop,
        #             im2col_top_mem_level)

        ''' I_base '''
        for level in range(mem_level['I']):
            mem_read['I_base'].append(np.prod(
                temporal_loop.irrelevant_loop['I_base'][level:mem_level['I']] +
                spatial_loop.unit_duplicate['I'][level + 1:mem_level['I'] + 1]
            ).item())

        for level in range(mem_level['I']):
            if level == mem_level['I'] - 1:
                ''' 
                For now, we don't consider the writing access from outside peripherals to the top-level memory. 
                '''
                mem_write = 0
            else:
                mem_write = mem_read['I_base'][level + 1]

            mem_access_per_element[operand].append([mem_read['I_base'][level], mem_write])
            mem_access_total_element['I_base'].append(
                [int(mem_read['I_base'][level] * layer.total_data_size['I']),
                 int(mem_write * layer.total_data_size['I'])])

        '''
        Memory access data format for zig-zag input:
        [(Read, Write @ mem 0),(Read, Write @ mem 1), ...]
        '''
        # temp = copy.deepcopy(temporal_loop.temporal_loop['I'])
        # spat = copy.deepcopy(spatial_loop.spatial_loop['I'])
        # input_irrelevant_unrolling_MAC_level = 1
        # for unroll in spat[0]:
        #     if unroll[0] == 6:
        #         input_irrelevant_unrolling_MAC_level *= unroll[1]
        #
        # mem_read_zigzag = int(layer.total_MAC_op / input_irrelevant_unrolling_MAC_level)
        # for level in range(1, mem_level['I']):
        #
        #     input_zig_zag = self.zig_zag_input_access(level, temp, spat)
        #     # mem_read['I_zig_zag'].append(self.zig_zag_input_access(level, temp, spat))
        #     C_spat = 1
        #     C_temp = 1
        #     NStationary = 1
        #     irr = [6]
        #     rel = [1, 2, 3, 4]
        #     for lev in range(0, level + 2):
        #         for i in range(0, len(spat[lev])):
        #             if spat[lev][i][0] == 5:
        #                 C_spat *= spat[lev][i][1]
        #     for lev in range(0, len(temp)):
        #         for i in range(0, len(temp[lev])):
        #             if temp[lev][i][0] == 5:
        #                 C_temp *= temp[lev][i][1]
        #
        #     ns_loops_index = len(temp[level])
        #     for i in range(0, len(temp[level])):
        #         if temp[level][i][0] == 6 and i != 0:
        #             ns_loops_index = i
        #             break
        #     for j in range(ns_loops_index, len(temp[level])):
        #         if temp[level][j][0] not in [1, 2, 3, 4, 5]:
        #             NStationary *= temp[level][j][1]
        #
        #     for lev in range(level + 1, len(temp)):
        #         ns_loops_index = len(temp[lev])
        #         for i in range(0, len(temp[lev])):
        #             if temp[lev][i][0] in [1, 2, 3, 4, 5]:
        #                 ns_loops_index = i
        #                 break
        #         for j in range(ns_loops_index, len(temp[lev])):
        #             if temp[lev][j][0] != 5:
        #                 NStationary *= temp[lev][j][1]
        #
        #     mem_write_zigzag = input_zig_zag * C_spat * C_temp * NStationary
        #     mem_access_total_element['I_zig_zag'].append([mem_read_zigzag, mem_write_zigzag])
        #     mem_read_zigzag = mem_write_zigzag
        #
        # mem_access_total_element['I_zig_zag'].append([mem_read_zigzag, 0])
        mem_access_total_element['I_zig_zag'].append(0)

        '''
        input data_reuse in zigzag mode (2D shifting reuse) is calculated reversely, from mem_access to data_reuse.
        Note that input data_reuse in baseline (no shifting reuse) and FIFO mode (1D shifting reuse) 
        are used to calculated mem_access.
        '''
        # zz_reuse_level_and_above = []
        # for level in range(mem_level['I']):
        #     zz_reuse_level_and_above.append(
        #         mem_access_total_element['I_zig_zag'][level][0] / layer.total_data_size['I'])
        #
        # zz_reuse_level_and_above.append(1)
        # for level in range(mem_level['I']):
        #     data_reuse['I_zigzag'].append(zz_reuse_level_and_above[level] / zz_reuse_level_and_above[level + 1])

        '''
        Memory access data format for output:
        [((ReadL, WriteL @ mem 0),(ReadH, WriteH @ mem 0)), ((ReadL, WriteL @ mem 1),(ReadH, WriteH @ mem 1)), ...]
        
        'ReadL' at current level is corresponding to 'WriteH' at one level below it.
        'WriteL' at current level is corresponding to 'ReadH' at one level below it.
        '''
        mem_read_L = []
        mem_write_L = []
        mem_read_H = []
        mem_write_H = []

        total_num_of_output = layer.total_data_size['O']

        for level in range(mem_level['O']):
            mem_write_L.append(np.prod(
                temporal_loop.irrelevant_loop['O'][level:mem_level['O']] +
                spatial_loop.unit_duplicate['O'][level + 1:mem_level['O'] + 1]
            ).item())

            mem_read_L.append(mem_write_L[level] - 1)

        for level in range(mem_level['O']):
            if level == mem_level['O'] - 1:
                ''' 
                For now, we don't consider the writing access from outside peripherals to the top-level memory. 
                '''
                mem_read_H.append(0)
                mem_write_H.append(0)
            else:
                mem_read_H.append(mem_write_L[level + 1])
                mem_write_H.append(mem_read_L[level + 1])

            mem_access_per_element['O'].append(
                ((mem_read_L[level], mem_write_L[level]), (mem_read_H[level], mem_write_H[level])))

            mem_access_total_element['O'].append(
                [(mem_read_L[level] * total_num_of_output,
                  mem_write_L[level] * total_num_of_output),
                 (mem_read_H[level] * total_num_of_output,
                  mem_write_H[level] * total_num_of_output)])

        '''
        Distinguish partial output and final output.
        '''

        output_precision = []
        output_distinguish = []

        for level in range(mem_level['O']):
            if (mem_read_L[level], mem_write_L[level]) == (0, 1):
                '''
                Final outputs are only written once from lower level of memory to current level of memory.
                '''
                mem_access_total_element['O_final'].append(
                    [(mem_read_L[level] * total_num_of_output, mem_write_L[level] * total_num_of_output),
                     (mem_read_H[level] * total_num_of_output, mem_write_H[level] * total_num_of_output)])

                mem_access_total_element['O_partial'].append([(0, 0), (0, 0)])

                req_aver_mem_bw['O_final'].append(req_aver_mem_bw['O'][level])
                req_aver_mem_bw['O_partial'].append([])

                req_inst_mem_bw['O_final'].append(req_inst_mem_bw['O'][level])
                req_inst_mem_bw['O_partial'].append([])

                req_mem_size['O_final'].append(req_mem_size['O'][level])
                req_mem_size['O_partial'].append(0)

                req_mem_count['O_final'].append(req_mem_count['O'][level])
                req_mem_count['O_partial'].append(0)

                effective_mem_size['O_final'].append(effective_mem_size['O'][level])
                effective_mem_size['O_partial'].append(0)

                output_precision.append((precision['O_final'], precision['O_final']))
                output_distinguish.append(('fsum', 'fsum'))


            elif (mem_read_H[level], mem_write_H[level]) == (1, 0):
                '''
                Partial outputs are written multiple times from lower level of memory to current level of memory.
                Final outputs are only read once from current level of memory to higher level of memory.
                '''
                mem_access_total_element['O_final'].append(
                    [(0, 0), (mem_read_H[level] * total_num_of_output, mem_write_H[level] * total_num_of_output)])

                mem_access_total_element['O_partial'].append(
                    [(mem_read_L[level] * total_num_of_output, mem_write_L[level] * total_num_of_output), (0, 0)])

                req_aver_mem_bw['O_final'].append([(), req_aver_mem_bw['O'][level][1]])
                req_aver_mem_bw['O_partial'].append([req_aver_mem_bw['O'][level][0], ()])

                req_inst_mem_bw['O_final'].append([(), req_inst_mem_bw['O'][level][1]])
                req_inst_mem_bw['O_partial'].append([req_inst_mem_bw['O'][level][0], ()])

                req_mem_size['O_final'].append(0)
                req_mem_size['O_partial'].append(req_mem_size['O'][level])

                req_mem_count['O_final'].append(0)
                req_mem_count['O_partial'].append(req_mem_count['O'][level])

                effective_mem_size['O_final'].append(0)
                effective_mem_size['O_partial'].append(effective_mem_size['O'][level])

                output_precision.append((precision['O'], precision['O_final']))
                output_distinguish.append(('psum', 'fsum'))


            else:
                '''
                Partial outputs are written multiple times from lower level of memory to current level of memory.
                Partial outputs are read multiple times from current level of memory to higher level of memory.
                '''
                mem_access_total_element['O_final'].append([(0, 0), (0, 0)])

                mem_access_total_element['O_partial'].append(
                    [(mem_read_L[level] * total_num_of_output, mem_write_L[level] * total_num_of_output),
                     (mem_read_H[level] * total_num_of_output, mem_write_H[level] * total_num_of_output)])

                req_aver_mem_bw['O_final'].append([])
                req_aver_mem_bw['O_partial'].append(req_aver_mem_bw['O'][level])

                req_inst_mem_bw['O_final'].append([])
                req_inst_mem_bw['O_partial'].append(req_inst_mem_bw['O'][level])

                req_mem_size['O_final'].append(0)
                req_mem_size['O_partial'].append(req_mem_size['O'][level])

                req_mem_count['O_final'].append(0)
                req_mem_count['O_partial'].append(req_mem_count['O'][level])

                effective_mem_size['O_final'].append(0)
                effective_mem_size['O_partial'].append(effective_mem_size['O'][level])

                output_precision.append((precision['O'], precision['O']))
                output_distinguish.append(('psum', 'psum'))

        '''
        For the 'addressable' & lowest level (one level above the MAC) memory, for 'W' and 'I', 
        if it has MAC-level stationarity, meaning that data from same address need to be repetitively read out, 
        we assume only the fist-time read consumes energy. 
        
        For 'O', such rule doesn't hold since output need to be firstly written and then read.
        '''
        level0 = 0
        read = 0
        for operand in ['W', 'I']:
            mem_access_total_element[operand][level0][read] = \
                mem_access_total_element[operand][level0][read] / temporal_loop.MAC_level_stationary[operand]

        '''input memory FIFO checking'''
        unrolled_loops = list(zip(*spatial_loop.spatial_loop_list))[0]
        I_fifo = [True] if all(x in unrolled_loops for x in [1, 3]) or all(x in unrolled_loops for x in [2, 4]) \
            else [False]
        I_fifo.extend([False] * mem_level['I'])
    
        I_fifo_reuse = 1
        I_base_reuse = 1
        for level in range(1, data_reuse['I'].__len__()):

            I_fifo_reuse *= data_reuse['I'][level]
            I_base_reuse *= data_reuse['I_base'][level]
            if I_fifo_reuse == I_base_reuse:
                continue
            # if I_fifo_reuse >= I_base_reuse and \
            #         req_mem_size['I'][level-1]*req_mem_count['I'][level-1] == spatial_loop.unit_unique['I'][level-1]:
            #     I_fifo[level-1] = True
            if I_fifo_reuse >= I_base_reuse:
                I_fifo[level] = True
                break

        if size_check:
            for operand in ['W', 'I', 'O']:
                if req_mem_size[operand][-1] != layer.total_data_size[operand]:
                    raise ValueError("For operand %s, your mapping size (%d) does not match with the NN layer size (%d). Please check your mapping setting."
                                     % (operand, req_mem_size[operand][-1], layer.total_data_size[operand]))

        self.req_mem_size = req_mem_size
        self.req_mem_count = req_mem_count
        self.effective_mem_size = effective_mem_size
        self.req_aver_mem_bw = req_aver_mem_bw
        self.req_inst_mem_bw = req_inst_mem_bw
        self.top_ir = top_ir
        self.data_reuse = data_reuse
        self.mem_access_elem = mem_access_total_element

        self.array_wire_distance = {'W': [], 'I': [], 'O': []}

        self.output_precision = output_precision
        self.output_distinguish = output_distinguish

        self.dt_bloc = dt_bloc
        self.dt_bloc_below = dt_bloc_below

        self.I_fifo = I_fifo

    # def zig_zag_input_access(self, level, temp_loop, spat_loop):
    #     '''
    #     NN layer indexes
    #     (7)B:   Batch size
    #     (6)K:   Filter kernels / Output channels
    #     (5)C:   Filter channels / Input channels
    #     (4)OY:  Y dimension of output feature map
    #     (3)OX:  X dimension of output feature map
    #     (2)FY:  Y dimension of filters
    #     (1)FX:  X dimension of filters
    #     '''
    #
    #     '''
    #     Very weird way to extract the necessary information for the input schedule
    #     Makes sure that input_schedule[level - 1] contains both temporal and spatial loops
    #     '''
    #
    #     temp = copy.deepcopy(temp_loop)
    #     spat = copy.deepcopy(spat_loop)
    #     input_schedule = copy.deepcopy(temp)
    #
    #     for lev in range(1, level + 1):
    #         if temp[lev - 1]:
    #             t_loop_type, t_loop_sizes = zip(*temp[lev - 1])
    #         else:
    #             t_loop_type = []
    #         for i in range(0, len(spat[lev])):
    #             if spat[lev][i][0] in t_loop_type:
    #                 lst = list(temp[lev - 1][t_loop_type.index(spat[lev][i][0])])
    #                 lst[1] *= spat[lev][i][1]
    #                 tempp = [j for j in temp[lev - 1] if j[0] != spat[lev][i][0]]
    #                 tempp.append(tuple(lst))
    #                 input_schedule[lev - 1] = tempp
    #             else:
    #                 input_schedule[lev - 1].append(spat[lev][i])
    #
    #     delta = [0, 1]
    #     delta_nfs0 = [0, 1]
    #     delta_ns0 = [0, 1]
    #
    #     gamma = [0, 1]
    #     gamma_nfs0 = [0, 1]
    #     gamma_ns0 = [0, 1]
    #
    #     beta = [0, 1]
    #     beta_s0 = [0, 1]
    #     beta_fs1 = [0, 1]
    #     beta_nfs0 = [0, 1]
    #     beta_ns0 = [0, 1]
    #
    #     alpha = [0, 1]
    #     alpha_s0 = [0, 1]
    #     alpha_fs1 = [0, 1]
    #     alpha_nfs0 = [0, 1]
    #     alpha_ns0 = [0, 1]
    #
    #     fx0 = [1, 1]
    #     fy0 = [2, 1]
    #     x0 = [3, 1]
    #     y0 = [4, 1]
    #
    #     split_multiplier = 1
    #     split_index = len(input_schedule[level])
    #     if any(i_l[0] == 5 or i_l[0] == 7 for i_l in input_schedule[level]):
    #         split_index = input_schedule[level].index(next(x for x in input_schedule[level] if x[0] == 5 or x[0] == 7))
    #     if not input_schedule[level]:
    #         if level == 0:
    #             return 1
    #         else:
    #             for j in range(1, level + 1):
    #                 for i in range(0, len(input_schedule[j - 1])):
    #                     if input_schedule[j - 1][i][0] == 1:
    #                         fx0[1] *= input_schedule[j - 1][i][1]
    #                     if input_schedule[j - 1][i][0] == 2:
    #                         fy0[1] *= input_schedule[j - 1][i][1]
    #                     if input_schedule[j - 1][i][0] == 3:
    #                         x0[1] *= input_schedule[j - 1][i][1]
    #                     if input_schedule[j - 1][i][0] == 4:
    #                         y0[1] *= input_schedule[j - 1][i][1]
    #         return (fx0[1] + x0[1] - 1) * (fy0[1] + y0[1] - 1)
    #
    #     split_multiplier = 1
    #     split_index = len(input_schedule[level])
    #     split_index2 = split_index
    #     lps = [x[0] for x in input_schedule[level]]
    #     if any(i_l[0] == 5 or i_l[0] == 7 for i_l in input_schedule[level]):
    #         split_index = input_schedule[level].index(next(x for x in input_schedule[level] if x[0] == 5 or x[0] == 7))
    #     if any(lps.count(x) > 1 for x in [1, 2, 3, 4]):
    #         count_split = {1: 0, 2: 0, 3: 0, 4: 0}
    #         for lp in range(0, len(input_schedule[level])):
    #             if input_schedule[level][lp][0] in [1, 2, 3, 4]:
    #                 count_split[input_schedule[level][lp][0]] += 1
    #                 if count_split[input_schedule[level][lp][0]] > 1:
    #                     split_index2 = lp
    #                     break
    #     split_index = min([split_index, split_index2])
    #     tmp_input_schedule = copy.deepcopy(input_schedule)
    #     for i in range(0, len(input_schedule[level])):
    #         if i > split_index:
    #             tmp_input_schedule[level].remove(input_schedule[level][i])
    #     tmp_input_schedule = [i for i in tmp_input_schedule[level] if i[0] in [1, 2, 3, 4]]
    #     tmp_input_schedule = [tmp_input_schedule]
    #     split_index_aux = len(tmp_input_schedule[0])
    #
    #     X_loop_compressable = True
    #     Y_loop_compressable = True
    #     s0 = 1
    #     fs1 = 1
    #     for i in range(0, len(input_schedule[level - 1])):
    #         if input_schedule[level - 1][i][0] == 3:
    #             s0 *= input_schedule[level - 1][i][1]
    #     for i in range(0, split_index - 1):
    #         if input_schedule[level][i][0] == 1:
    #             fs1 *= input_schedule[level][i][1]
    #     if s0 - fs1 + 1 < 0:
    #         X_loop_compressable = False
    #     s0 = 1
    #     fs1 = 1
    #     for i in range(0, len(input_schedule[level - 1])):
    #         if input_schedule[level - 1][i][0] == 4:
    #             s0 *= input_schedule[level - 1][i][1]
    #     for i in range(0, split_index - 1):
    #         if input_schedule[level][i][0] == 2:
    #             fs1 *= input_schedule[level][i][1]
    #     if s0 - fs1 + 1 < 0:
    #         Y_loop_compressable = False
    #
    #
    #     for i in range(0, split_index - 1):
    #         if input_schedule[level][i][0] == 1 and input_schedule[level][i + 1][0] == 3 and X_loop_compressable:
    #             tmp_input_schedule[0].remove(tuple([3, input_schedule[level][i + 1][
    #                 1]]))
    #             tmp_input_schedule[0].remove(tuple([1, input_schedule[level][i][
    #                 1]]))
    #             tmp_input_schedule[0].append(
    #                 tuple([1, input_schedule[level][i][1] + input_schedule[level][i + 1][1] - 1]))
    #             split_index_aux -= 1
    #         if input_schedule[level][i][0] == 3 and input_schedule[level][i + 1][0] == 1  and X_loop_compressable:
    #             tmp_input_schedule[0].remove(tuple([3, input_schedule[level][i][
    #                 1]]))
    #             tmp_input_schedule[0].remove(tuple([1, input_schedule[level][i + 1][
    #                 1]]))
    #             tmp_input_schedule[0].append(
    #                 tuple([1, input_schedule[level][i][1] + input_schedule[level][i + 1][1] - 1]))
    #             split_index_aux -= 1
    #         if input_schedule[level][i][0] == 2 and input_schedule[level][i + 1][0] == 4 and Y_loop_compressable:
    #             tmp_input_schedule[0].remove(tuple([4, input_schedule[level][i + 1][
    #                 1]]))
    #             tmp_input_schedule[0].remove(tuple([2, input_schedule[level][i][
    #                 1]]))
    #             tmp_input_schedule[0].append(
    #                 tuple([2, input_schedule[level][i][1] + input_schedule[level][i + 1][1] - 1]))
    #             split_index_aux -= 1
    #         if input_schedule[level][i][0] == 4 and input_schedule[level][i + 1][0] == 2 and Y_loop_compressable:
    #             tmp_input_schedule[0].remove(tuple([2, input_schedule[level][i + 1][
    #                 1]]))
    #             tmp_input_schedule[0].remove(tuple([4, input_schedule[level][i][
    #                 1]]))
    #             tmp_input_schedule[0].append(
    #                 tuple([2, input_schedule[level][i][1] + input_schedule[level][i + 1][1] - 1]))
    #             split_index_aux -= 1
    #
    #
    #
    #
    #     if not input_schedule[level]:
    #         if level == 0:
    #             return 1
    #         else:
    #             for j in range(1, level + 1):
    #                 for i in range(0, len(input_schedule[j - 1])):
    #                     if input_schedule[j - 1][i][0] == 1:
    #                         fx0[1] *= input_schedule[j - 1][i][1]
    #                     if input_schedule[j - 1][i][0] == 2:
    #                         fy0[1] *= input_schedule[j - 1][i][1]
    #                     if input_schedule[j - 1][i][0] == 3:
    #                         x0[1] *= input_schedule[j - 1][i][1]
    #                     if input_schedule[j - 1][i][0] == 4:
    #                         y0[1] *= input_schedule[j - 1][i][1]
    #         return (fx0[1] + x0[1] - 1) * (fy0[1] + y0[1] - 1)
    #
    #     if tmp_input_schedule[0]:
    #         loop_type, loop_sizes = zip(*tmp_input_schedule[0])
    #     else:
    #         loop_type = []
    #         loop_sizes = []
    #
    #     if all(elem in loop_type for elem in [1, 2]):
    #         delta_assigned = 0
    #         for i in range(0, split_index_aux):
    #             if (delta_assigned == 0) and (tmp_input_schedule[0][i][0] == 1 or tmp_input_schedule[0][i][0] == 2):
    #                 delta[0] = tmp_input_schedule[0][i][0]
    #                 delta[1] = tmp_input_schedule[0][i][1]
    #
    #                 delta_assigned = 1
    #                 continue
    #             if (delta_assigned == 1) and (tmp_input_schedule[0][i][0] == 1 or tmp_input_schedule[0][i][0] == 2):
    #                 gamma[0] = tmp_input_schedule[0][i][0]
    #                 gamma[1] = tmp_input_schedule[0][i][1]
    #                 break
    #
    #     if any(elem in loop_type for elem in [1, 2]) and not (all(elem in loop_type for elem in [1, 2])):
    #         for i in range(0, split_index_aux):
    #             if tmp_input_schedule[0][i][0] == 1 or tmp_input_schedule[0][i][0] == 2:
    #                 gamma[0] = tmp_input_schedule[0][i][0]
    #                 gamma[1] = tmp_input_schedule[0][i][1]
    #                 delta.append(1)
    #                 break
    #
    #     if all(elem in loop_type for elem in [3, 4]):
    #         beta_assigned = 0
    #         for i in range(0, split_index_aux):
    #             if (beta_assigned == 0) and (tmp_input_schedule[0][i][0] == 3 or tmp_input_schedule[0][i][0] == 4):
    #                 beta[0] = tmp_input_schedule[0][i][0]
    #                 beta[1] = tmp_input_schedule[0][i][1]
    #
    #                 beta_assigned = 1
    #                 continue
    #             if (beta_assigned == 1) and (tmp_input_schedule[0][i][0] == 3 or tmp_input_schedule[0][i][0] == 4):
    #                 alpha[0] = tmp_input_schedule[0][i][0]
    #                 alpha[1] = tmp_input_schedule[0][i][1]
    #                 break
    #
    #     if any(elem in loop_type for elem in [3, 4]) and not (all(elem in loop_type for elem in [3, 4])):
    #         for i in range(0, split_index_aux):
    #             if tmp_input_schedule[0][i][0] == 3 or tmp_input_schedule[0][i][0] == 4:
    #                 alpha[0] = tmp_input_schedule[0][i][0]
    #                 alpha[1] = tmp_input_schedule[0][i][1]
    #
    #                 beta.append(1)
    #                 break
    #
    #     if level == 0:
    #         if (alpha[0] == 3 and gamma[0] == 1) or (alpha[0] == 4 and gamma[0] == 2):
    #             return (alpha[1] + gamma[1] - 1) * (beta[1] + delta[1] - 1)
    #         else:
    #             return (alpha[1] + delta[1] - 1) * (beta[1] + gamma[1] - 1)
    #     else:
    #         for j in range(1, level + 1):
    #             for i in range(0, len(input_schedule[j - 1])):
    #                 if input_schedule[j - 1][i][0] == 1:
    #                     fx0[1] *= input_schedule[j - 1][i][1]
    #                 if input_schedule[j - 1][i][0] == 2:
    #                     fy0[1] *= input_schedule[j - 1][i][1]
    #                 if input_schedule[j - 1][i][0] == 3:
    #                     x0[1] *= input_schedule[j - 1][i][1]
    #                 if input_schedule[j - 1][i][0] == 4:
    #                     y0[1] *= input_schedule[j - 1][i][1]
    #
    #         if alpha[0] != 0:
    #             if alpha[0] == 3:
    #                 alpha_fs1[0] = 1
    #                 for i in range(0, len(tmp_input_schedule[0])):
    #                     if tmp_input_schedule[0][i][0] == 1:
    #                         alpha_fs1[1] = tmp_input_schedule[0][i][1]
    #                 alpha_nfs0[0] = 2
    #                 for j in range(1, level + 1):
    #                     for i in range(0, len(input_schedule[j - 1])):
    #                         if input_schedule[j - 1][i][0] == 2:
    #                             alpha_nfs0[1] *= input_schedule[j - 1][i][1]
    #                 alpha_ns0[0] = 4
    #                 for j in range(1, level + 1):
    #                     for i in range(0, len(input_schedule[j - 1])):
    #                         if input_schedule[j - 1][i][0] == 4:
    #                             alpha_ns0[1] *= input_schedule[j - 1][i][1]
    #                 alpha_s0[0] = 3
    #                 for j in range(1, level + 1):
    #                     for i in range(0, len(input_schedule[j - 1])):
    #                         if input_schedule[j - 1][i][0] == 3:
    #                             alpha_s0[1] *= input_schedule[j - 1][i][1]
    #             if alpha[0] == 4:
    #                 alpha_fs1[0] = 2
    #                 for i in range(0, len(tmp_input_schedule[0])):
    #                     if tmp_input_schedule[0][i][0] == 2:
    #                         alpha_fs1[1] = tmp_input_schedule[0][i][1]
    #                 alpha_nfs0[0] = 1
    #                 for j in range(1, level + 1):
    #                     for i in range(0, len(input_schedule[j - 1])):
    #                         if input_schedule[j - 1][i][0] == 1:
    #                             alpha_nfs0[1] *= input_schedule[j - 1][i][1]
    #                 alpha_ns0[0] = 3
    #                 for j in range(1, level + 1):
    #                     for i in range(0, len(input_schedule[j - 1])):
    #                         if input_schedule[j - 1][i][0] == 3:
    #                             alpha_ns0[1] *= input_schedule[j - 1][i][1]
    #                 alpha_s0[0] = 4
    #                 for j in range(1, level + 1):
    #                     for i in range(0, len(input_schedule[j - 1])):
    #                         if input_schedule[j - 1][i][0] == 4:
    #                             alpha_s0[1] *= input_schedule[j - 1][i][1]
    #         if beta[0] != 0:
    #             if beta[0] == 3:
    #                 beta_fs1[0] = 1
    #                 for i in range(0, len(tmp_input_schedule[0])):
    #                     if tmp_input_schedule[0][i][0] == 1:
    #                         beta_fs1[1] = tmp_input_schedule[0][i][1]
    #                 beta_nfs0[0] = 2
    #                 for j in range(1, level + 1):
    #                     for i in range(0, len(input_schedule[j - 1])):
    #                         if input_schedule[j - 1][i][0] == 2:
    #                             beta_nfs0[1] = input_schedule[j - 1][i][1]
    #                 beta_ns0[0] = 4
    #                 for j in range(1, level + 1):
    #                     for i in range(0, len(input_schedule[j - 1])):
    #                         if input_schedule[j - 1][i][0] == 4:
    #                             beta_ns0[1] = input_schedule[j - 1][i][1]
    #                 beta_s0[0] = 3
    #                 for j in range(1, level + 1):
    #                     for i in range(0, len(input_schedule[j - 1])):
    #                         if input_schedule[j - 1][i][0] == 3:
    #                             beta_s0[1] = input_schedule[j - 1][i][1]
    #             if beta[0] == 4:
    #                 beta_fs1[0] = 2
    #                 for i in range(0, len(tmp_input_schedule[0])):
    #                     if tmp_input_schedule[0][i][0] == 2:
    #                         beta_fs1[1] = tmp_input_schedule[0][i][1]
    #                 beta_nfs0[0] = 1
    #                 for j in range(1, level + 1):
    #                     for i in range(0, len(input_schedule[j - 1])):
    #                         if input_schedule[j - 1][i][0] == 1:
    #                             beta_nfs0[1] = input_schedule[j - 1][i][1]
    #                 beta_ns0[0] = 3
    #                 for j in range(1, level + 1):
    #                     for i in range(0, len(input_schedule[j - 1])):
    #                         if input_schedule[j - 1][i][0] == 3:
    #                             beta_ns0[1] = input_schedule[j - 1][i][1]
    #                 beta_s0[0] = 4
    #                 for j in range(1, level + 1):
    #                     for i in range(0, len(input_schedule[j - 1])):
    #                         if input_schedule[j - 1][i][0] == 4:
    #                             beta_s0[1] = input_schedule[j - 1][i][1]
    #         if gamma[0] != 0:
    #             if gamma[0] == 1:
    #                 gamma_nfs0[0] = 2
    #                 for j in range(1, level + 1):
    #                     for i in range(0, len(input_schedule[j - 1])):
    #                         if input_schedule[j - 1][i][0] == 2:
    #                             gamma_nfs0[1] = input_schedule[j - 1][i][1]
    #                 gamma_ns0[0] = 4
    #                 for j in range(1, level + 1):
    #                     for i in range(0, len(input_schedule[j - 1])):
    #                         if input_schedule[j - 1][i][0] == 4:
    #                             gamma_ns0[1] = input_schedule[j - 1][i][1]
    #             if gamma[0] == 2:
    #                 gamma_nfs0[0] = 1
    #                 for j in range(1, level + 1):
    #                     for i in range(0, len(input_schedule[j - 1])):
    #                         if input_schedule[j - 1][i][0] == 1:
    #                             gamma_nfs0[1] = input_schedule[j - 1][i][1]
    #                 gamma_ns0[0] = 3
    #                 for j in range(1, level + 1):
    #                     for i in range(0, len(input_schedule[j - 1])):
    #                         if input_schedule[j - 1][i][0] == 3:
    #                             gamma_ns0[1] = input_schedule[j - 1][i][1]
    #         if delta[0] != 0:
    #             if delta[0] == 1:
    #                 delta_nfs0[0] = 2
    #                 for j in range(1, level + 1):
    #                     for i in range(0, len(input_schedule[j - 1])):
    #                         if input_schedule[j - 1][i][0] == 2:
    #                             delta_nfs0[1] = input_schedule[j - 1][i][1]
    #                 delta_ns0[0] = 4
    #                 for j in range(1, level + 1):
    #                     for i in range(0, len(input_schedule[j - 1])):
    #                         if input_schedule[j - 1][i][0] == 4:
    #                             delta_ns0[1] = input_schedule[j - 1][i][1]
    #             if delta[0] == 2:
    #                 delta_nfs0[0] = 1
    #                 for j in range(1, level + 1):
    #                     for i in range(0, len(input_schedule[j - 1])):
    #                         if input_schedule[j - 1][i][0] == 1:
    #                             delta_nfs0[1] = input_schedule[j - 1][i][1]
    #                 delta_ns0[0] = 3
    #                 for j in range(1, level + 1):
    #                     for i in range(0, len(input_schedule[j - 1])):
    #                         if input_schedule[j - 1][i][0] == 3:
    #                             delta_ns0[1] = input_schedule[j - 1][i][1]
    #
    #         a = 0
    #         b = 0
    #         c = 0
    #         d = 0
    #         tot = 1
    #
    #         if (alpha_s0[1] - alpha_fs1[1] + 1) < 0 and delta[1] == 1:
    #             tot *= alpha[1] * (
    #                     (fx0[1] + x0[1] - 1) * (fy0[1] + y0[1] - 1) + beta[1] * (gamma[1] - 1) * (fx0[1] + x0[1] - 1))
    #         else:
    #             a = (fx0[1] + x0[1] - 1) * (fy0[1] + y0[1] - 1)
    #             b = (delta[1] - 1) * (delta_nfs0[1] + delta_ns0[1] - 1)
    #             c = (gamma[1] - 1) * ((gamma_nfs0[1] + gamma_ns0[1] - 1) + b)
    #             if (beta_s0[1] - beta_fs1[1] + 1) >= 0:
    #                 d = (beta[1] - 1) * ((beta_s0[1] - beta_fs1[1] + 1) * (beta_nfs0[1] + beta_ns0[1] - 1) + b + c)
    #             else:
    #                 d = (beta[1] - 1) * (a + b + c)
    #             if (alpha_s0[1] - alpha_fs1[1] + 1) >= 0 and (
    #                     (beta[1] % 2) == 1 or (gamma[1] == 1 and delta[1] == 1) or (
    #                     beta[1] != 1 and ((gamma[0] == 1 and beta[0] == 3) or (gamma[0] == 2 and beta[0] == 4)))):
    #                 tot *= (alpha[1] - 1) * (
    #                         (alpha_s0[1] - alpha_fs1[1] + 1) * (alpha_nfs0[1] + alpha_ns0[1] - 1) + b + c + d)
    #             else:
    #                 tot *= (alpha[1] - 1) * (a + b + c + d)
    #
    #         for j in range(split_index, len(input_schedule[level])):
    #             if input_schedule[level][j][0] in [1, 2, 3, 4]:
    #                 split_multiplier *= input_schedule[level][j][1]
    #
    #         return (tot + a + b + c + d) * split_multiplier

    @classmethod
    def extract_loop_info(cls, layer, temporal_loop, spatial_loop, precision, size_check):
        return cls(layer, temporal_loop, spatial_loop, precision, size_check)

"""
Temporal loop information extraction and integration.
"""

import numpy as np
import copy


class TemporalLoop(object):

    def __init__(self, layer, temporal_loop, spatial_loop):
        self.layer = layer
        self.temporal_loop_raw = temporal_loop

        ''' 
        temporal_loop_st: reshaped temporal loop which takes stationary into consideration, 
        i.e. merge the lowest ir loops at each mem level to the level below.
        Doing this won't have any impact on required mem size but will reduce 
        the memory access account due to taking stationarity into account.
        
        'li' is short for 'list'.
        'lo' is short for 'loop'.
        '''
        # if one want to disable this stationary loop merging transformation,
        # just disable this whole temporal_loop_st generating block,
        # and uncomment the following:
        #
        # temporal_loop_st = copy.deepcopy(temporal_loop)

        temporal_loop_st = {}

        ir_loop = {'W': [3, 4, 7],
                   'I': [6],
                   'O': [1, 2, 5]}

        temporal_loop_copy = copy.deepcopy(temporal_loop)
        temporal_loop_previous = copy.deepcopy(temporal_loop)
        not_finish = True
        while not_finish:
            for operand in ['W', 'I', 'O']:
                temporal_loop_st[operand] = []
                for level, li in enumerate(temporal_loop_previous[operand]):
                    temporal_loop_st[operand].append([])
                    if level == 0 or not li:
                        temporal_loop_st[operand][level] = copy.deepcopy(li)
                    else:
                        li_save = copy.deepcopy(li)
                        for lo in li_save:
                            if lo[0] in ir_loop[operand]:
                                temporal_loop_st[operand][level-1].append(lo)
                                temporal_loop_copy[operand][level].remove(lo)
                            else:
                                temporal_loop_st[operand][level] = copy.deepcopy(temporal_loop_copy[operand][level])
                                break
            if temporal_loop_st != temporal_loop_previous:
                temporal_loop_previous = copy.deepcopy(temporal_loop_st)
                temporal_loop_copy = copy.deepcopy(temporal_loop_st)
                continue
            else:
                not_finish = False

        B = {}
        K = {}
        C = {}
        OY = {}
        OX = {}
        FY = {}
        FX = {}

        loop_individual_level_cycles = {}
        loop_cycles = {}
        loop_levels = {}
        irrelevant_loop = {}
        available_update_cycles = {}

        for operand in temporal_loop_st.keys():

            ''' 
            Initialize the list size to the number of memory levels. 

            B/K/.../FX: temporal loop size for each index at different mem level in W/I/O mem system.
            '''

            B[operand] = [1] * temporal_loop_st[operand].__len__()
            K[operand] = [1] * temporal_loop_st[operand].__len__()
            C[operand] = [1] * temporal_loop_st[operand].__len__()
            OY[operand] = [1] * temporal_loop_st[operand].__len__()
            OX[operand] = [1] * temporal_loop_st[operand].__len__()
            FY[operand] = [1] * temporal_loop_st[operand].__len__()
            FX[operand] = [1] * temporal_loop_st[operand].__len__()

            '''
            each_level_cycles: the number of total iteration at each mem level in W/I/O mem system.

            loop_levels: the number of mem levels in W/I/O mem system.
            '''
            loop_individual_level_cycles[operand] = [1] * temporal_loop_st[operand].__len__()
            loop_cycles[operand] = [1] * temporal_loop_st[operand].__len__()
            loop_levels[operand] = temporal_loop_st[operand].__len__()
            irrelevant_loop[operand] = [1] * temporal_loop_st[operand].__len__()
            available_update_cycles = [1] * temporal_loop_st[operand].__len__()

            for level, current_level_loops in enumerate(temporal_loop_st[operand]):
                for loop in current_level_loops:
                    if loop[0] == 7:
                        B[operand][level] *= loop[1]
                    elif loop[0] == 6:
                        K[operand][level] *= loop[1]
                    elif loop[0] == 5:
                        C[operand][level] *= loop[1]
                    elif loop[0] == 4:
                        OY[operand][level] *= loop[1]
                    elif loop[0] == 3:
                        OX[operand][level] *= loop[1]
                    elif loop[0] == 2:
                        FY[operand][level] *= loop[1]
                    elif loop[0] == 1:
                        FX[operand][level] *= loop[1]
                    else:
                        raise IndexError('The loop index can only be values from "1" to "7".')

                    loop_individual_level_cycles[operand][level] *= loop[1]

                loop_cycles[operand][level] = (np.prod(loop_individual_level_cycles[operand][0:level + 1])).item()

        '''
        irrelevant_loop is calculated by multiplying all the irrelevant loop indexes 
        for each operand at each level together.
        '''

        for level in range(loop_levels['W']):
            irrelevant_loop['W'][level] = B['W'][level] * OY['W'][level] * OX['W'][level]

        for level in range(loop_levels['O']):
            irrelevant_loop['O'][level] = C['O'][level] * FY['O'][level] * FX['O'][level]

        ''' Input '''

        ifmap_size = {'B': [1] * loop_levels['I'],
                      'C': [1] * loop_levels['I'],
                      'IY': [1] * loop_levels['I'],
                      'IX': [1] * loop_levels['I']}

        num_of_MAC_Op = [1] * loop_levels['I']

        interleaved_storage_IY = [False] * loop_levels['I']
        interleaved_storage_IX = [False] * loop_levels['I']

        for level in range(loop_levels['I']):
            ifmap_size['B'][level] = (np.prod(B['I'][0:level + 1] + spatial_loop.Bu['I'][0:level + 2])).item()
            ifmap_size['C'][level] = (np.prod(C['I'][0:level + 1] + spatial_loop.Cu['I'][0:level + 2])).item()

            if np.prod(OY['I'][0:level + 1] + spatial_loop.OYu['I'][0:level + 2]) == 1 or \
                    np.prod(FY['I'][0:level + 1] + spatial_loop.FYu['I'][0:level + 2]) == 1:
                IY = int((np.prod(OY['I'][0:level + 1] + spatial_loop.OYu['I'][0:level + 2])).item() +
                         (np.prod(FY['I'][0:level + 1] + spatial_loop.FYu['I'][0:level + 2])).item() - 1)
                interleaved_storage_IY[level] = True
            else:
                IY = int(layer.SY * ((np.prod(OY['I'][0:level + 1] + spatial_loop.OYu['I'][0:level + 2])).item() - 1) +
                         layer.SFY * ((np.prod(FY['I'][0:level + 1] + spatial_loop.FYu['I'][0:level + 2])).item() - 1) + 1)

            if np.prod(OX['I'][0:level + 1] + spatial_loop.OXu['I'][0:level + 2]) == 1 or \
                    np.prod(FX['I'][0:level + 1] + spatial_loop.FXu['I'][0:level + 2]) == 1:
                IX = int((np.prod(OX['I'][0:level + 1] + spatial_loop.OXu['I'][0:level + 2])).item() +
                         (np.prod(FX['I'][0:level + 1] + spatial_loop.FXu['I'][0:level + 2])).item() - 1)
                interleaved_storage_IX[level] = True
            else:
                IX = int(layer.SX * ((np.prod(OX['I'][0:level + 1] + spatial_loop.OXu['I'][0:level + 2])).item() - 1) +
                         layer.SFX * ((np.prod(FX['I'][0:level + 1] + spatial_loop.FXu['I'][0:level + 2])).item() - 1) + 1)
            ifmap_size['IY'][level] = IY
            ifmap_size['IX'][level] = IX

            num_of_MAC_Op[level] = loop_cycles['I'][level] * spatial_loop.unit_count['I'][0]

        '''
        Input access baseline: no data shifting reuse from IX/IY dimension, only data reuse from irrelevant loop of K.
        '''

        average_input_reuse_base = [spatial_loop.unit_duplicate['I'][0]]
        irrelevant_loop['I_base'] = [1] * temporal_loop_st['I'].__len__()
        for level in range(loop_levels['I']):
            num_of_input_elem = ifmap_size['B'][level] * ifmap_size['C'][level] * \
                                ifmap_size['IY'][level] * ifmap_size['IX'][level]
            average_input_reuse_base.append(num_of_MAC_Op[level] / num_of_input_elem)

            irrelevant_loop['I_base'][level] = average_input_reuse_base[level + 1] / average_input_reuse_base[level]

        '''
        Input 'FIFO effect':
        
        For Input, unlike Weight and Output, the irrelevant_loop values are not always integer 
        when taking IX/IY dimension data shifting reuse into consideration.
        
        Use average_input_reuse, i.e. total loop cycle / total # of element, 
        to get equivalent irrelevant_loop for input at each level.
        
        Collecting innermost_relevant_loop is to handle the input 'FIFO effect'.
        
        (level below | level above) 
        Input 1D data shifting reuse:
        Input 'FIFO effect' can happen when (IX != 1 | Fx...)
        Input 'Advanced FIFO effect' can happen when (IX != 1 | Fx,OX...)
        
        Input 2D data shifting reuse:
        Input 'Zigzag' can happen when (IX&IY != 1 | Fx,OX & Fy,OY...)
        Input 'Advanced Zigzag' can happen when (IX = ALL, IY = 1  | Fy,OY...)
        
        Here we consider 'FIFO effect' (1D).
        'Zigzag'(2D) has been covered in Loop class.
        '''
        # TODO 'Advanced Zigzag' to be done.

        l_type = 0
        l_range = 1
        innermost_relevant_loop = [[] for i in range(loop_levels['I'])]
        for level, loops in enumerate(temporal_loop_st['I']):
            previous_loop_type = 0
            if not loops:
                continue
            else:
                for loop in loops:
                    if (previous_loop_type in [0, 6]) and (loop[l_type] in [6]):
                        continue
                    elif (previous_loop_type in [0, 1, 3]) and (loop[l_type] in [1, 3]) or \
                            (previous_loop_type in [0, 2, 4]) and (loop[l_type] in [2, 4]):
                        innermost_relevant_loop[level].append(loop)
                        previous_loop_type = loop[l_type]
                    else:
                        break
        innermost_relevant_loop.append([])

        average_input_reuse = [spatial_loop.unit_duplicate['I'][0]]
        ifmap_size_merged = {'IY': [1] * loop_levels['I'],
                             'IX': [1] * loop_levels['I']}
        num_of_MAC_Op_merged = copy.deepcopy(num_of_MAC_Op)

        for level in range(loop_levels['I']):
            FX_merged_down = []
            FY_merged_down = []
            OX_merged_down = []
            OY_merged_down = []
            # if (not innermost_relevant_loop[level + 1]) or \
            #         (innermost_relevant_loop[level + 1][0][l_type] in [2, 4] and ifmap_size['IY'][level] == 1) or \
            #         (innermost_relevant_loop[level + 1][0][l_type] in [1, 3] and ifmap_size['IX'][level] == 1):
            if (not innermost_relevant_loop[level + 1]) or \
               (innermost_relevant_loop[level + 1][0][l_type] == 2 and np.prod(OY['I'][0:level + 1]) == 1) or \
               (innermost_relevant_loop[level + 1][0][l_type] == 4 and np.prod(FY['I'][0:level + 1]) == 1) or \
               (innermost_relevant_loop[level + 1][0][l_type] == 1 and np.prod(OX['I'][0:level + 1]) == 1) or \
               (innermost_relevant_loop[level + 1][0][l_type] == 3 and np.prod(FX['I'][0:level + 1]) == 1):
                num_of_input_elem = ifmap_size['B'][level] * ifmap_size['C'][level] * \
                                    ifmap_size['IY'][level] * ifmap_size['IX'][level]
                average_input_reuse.append(num_of_MAC_Op[level] / num_of_input_elem)

            else:
                for loop in innermost_relevant_loop[level + 1]:
                    num_of_MAC_Op_merged[level] *= loop[l_range]
                    if loop[l_type] == 1:
                        FX_merged_down += [loop[l_range]]
                    elif loop[l_type] == 2:
                        FY_merged_down += [loop[l_range]]
                    elif loop[l_type] == 3:
                        OX_merged_down += [loop[l_range]]
                    elif loop[l_type] == 4:
                        OY_merged_down += [loop[l_range]]

                IY = int(layer.SY * ((np.prod(OY['I'][0:level + 1] + spatial_loop.OYu['I'][0:level + 2] +
                                              OY_merged_down)).item() - 1) +
                         layer.SFY * ((np.prod(FY['I'][0:level + 1] + spatial_loop.FYu['I'][0:level + 2] +
                                               FY_merged_down)).item() - 1) + 1)
                IX = int(layer.SX * ((np.prod(OX['I'][0:level + 1] + spatial_loop.OXu['I'][0:level + 2] +
                                              OX_merged_down)).item() - 1) +
                         layer.SFX * ((np.prod(FX['I'][0:level + 1] + spatial_loop.FXu['I'][0:level + 2] +
                                               FX_merged_down)).item() - 1) + 1)

                ifmap_size_merged['IY'][level] = IY
                ifmap_size_merged['IX'][level] = IX

                num_of_input_elem = ifmap_size['B'][level] * ifmap_size['C'][level] * \
                                    ifmap_size_merged['IY'][level] * ifmap_size_merged['IX'][level]
                average_input_reuse.append(num_of_MAC_Op_merged[level] / num_of_input_elem)

            irrelevant_loop['I'][level] = average_input_reuse[level + 1] / average_input_reuse[level]

        '''
        MAC_level_stationary is used to capture how many continuing computing cycle 
        can each operand remain the same element at the MAC level.
        It can be also seen as single_element_stationary.
        
        MAC_level_stationary is calculated by multiplying all the irrelevant loops' indexes together 
        which locate just above the MAC level.
        '''
        MAC_level_stationary = {'W': 1,
                                'I': 1,
                                'O': 1}

        level0 = 0
        for loop in temporal_loop_st['W'][level0]:
            if not loop:
                break
            elif loop[l_type] == 3 or loop[l_type] == 4 or loop[l_type] == 7:
                MAC_level_stationary['W'] *= loop[l_range]
            else:
                break

        for loop in temporal_loop_st['I'][level0]:
            if not loop:
                break
            elif loop[l_type] == 6:
                MAC_level_stationary['I'] *= loop[l_range]
            else:
                break

        for loop in temporal_loop_st['O'][level0]:
            if not loop:
                break
            elif loop[l_type] == 1 or loop[l_type] == 2 or loop[l_type] == 5:
                MAC_level_stationary['O'] *= loop[l_range]
            else:
                break

        '''
        available_update_cycles: available clock cycles to update elements at current level.
        
        When can a level of memory start being written (overwrite its original data) / its above level memory
        start being read?
        -> When a certain block of the memory data are fully used and no longer needed.
        '''

        self.temporal_loop = temporal_loop_st

        self.B = B
        self.K = K
        self.C = C
        self.OY = OY
        self.OX = OX
        self.FY = FY
        self.FX = FX

        self.loop_individual_level_cycles = loop_individual_level_cycles
        self.loop_cycles = loop_cycles
        self.total_cycles = loop_cycles['W'][loop_levels['W'] - 1]
        self.loop_levels = loop_levels
        self.irrelevant_loop = irrelevant_loop

        self.MAC_level_stationary = MAC_level_stationary
        self.available_update_cycles = available_update_cycles

        self.interleaved_storage_IY = interleaved_storage_IY
        self.interleaved_storage_IX = interleaved_storage_IX

    @classmethod
    def extract_loop_info(cls, layer, temporal_loop, spatial_loop):
        return cls(layer, temporal_loop, spatial_loop)

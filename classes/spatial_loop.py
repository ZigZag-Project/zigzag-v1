"""
Spatial loop information extraction and integration.

Always note that the first level in spatial loops is MAC level (pure logic), not a memory level.
Thus the spatial loops always have one more level than temporal loops.
"""

import numpy as np


class SpatialLoop(object):

    def __init__(self, spatial_loop, layer_loop_info):
        self.spatial_loop = spatial_loop

        Bu = {}
        Ku = {}
        Cu = {}
        OYu = {}
        OXu = {}
        FYu = {}
        FXu = {}

        unroll_size = {}
        unit_count = {}
        unit_unique = {}
        unit_duplicate = {}
        loop_levels = {}

        unit_serve_scope = {}
        data_serve_scope = {}
        real_bw_boost = {}
        real_bw_boost_high = {}
        real_bw_boost_low = {}

        for operand in ['W', 'I', 'O']:

            ''' 
            Initialize the list size to the number of memory levels.

            Bu/Ku/.../FXu: the spatial loop unrolling size for each index 
            at different mem level in W/I/O mem system.
            '''

            Bu[operand] = [1] * spatial_loop[operand].__len__()
            Ku[operand] = [1] * spatial_loop[operand].__len__()
            Cu[operand] = [1] * spatial_loop[operand].__len__()
            OYu[operand] = [1] * spatial_loop[operand].__len__()
            OXu[operand] = [1] * spatial_loop[operand].__len__()
            FYu[operand] = [1] * spatial_loop[operand].__len__()
            FXu[operand] = [1] * spatial_loop[operand].__len__()

            ''' 
            unroll_size: the individual spatial unrolling at each level.

            unit_count: number of MAC/mem unit at certain level in W/I/O mem system, 
            i.e. total spatial unrolling at each level.

            unit_unique: number of unique MAC/mem unit at certain level in W/I/O mem system,
            i.e. MACs operate on different values / mem units that hold different values.

            unit_duplicate: number of duplicated MAC/mem unit at certain level in W/I/O mem system,
            i.e. MACs operate on same values / mem units that hold same values.

            loop_levels: number of mem levels in W/I/O mem system + !!! the innermost MAC logic level !!! .
            '''

            unroll_size[operand] = [1] * spatial_loop[operand].__len__()
            unit_count[operand] = [1] * spatial_loop[operand].__len__()
            unit_unique[operand] = [1] * spatial_loop[operand].__len__()
            unit_duplicate[operand] = [1] * spatial_loop[operand].__len__()
            loop_levels[operand] = spatial_loop[operand].__len__()

            '''
            unit_serve_scope: one mem at current level serves how many unit at one level below.
            
            data_serve_scope: one data element at current level serves how many unit at one level below.
            (!! this parameter can be used as spatially data-sharing hint !!)
            
            real_bw_boost: one mem at current level serves how many unit at one level below with different data.
            '''

            unit_serve_scope[operand] = [1] * (spatial_loop[operand].__len__() - 1)
            data_serve_scope[operand] = [1] * (spatial_loop[operand].__len__() - 1)
            real_bw_boost[operand] = [1] * (spatial_loop[operand].__len__() - 1)
            real_bw_boost_high[operand] = [1] * (spatial_loop[operand].__len__() - 1)
            real_bw_boost_low[operand] = [1] * (spatial_loop[operand].__len__() - 1)

            for level, current_level_loops in enumerate(spatial_loop[operand]):
                for loop in current_level_loops:
                    if loop[0] == 7:
                        Bu[operand][level] *= loop[1]
                    elif loop[0] == 6:
                        Ku[operand][level] *= loop[1]
                    elif loop[0] == 5:
                        Cu[operand][level] *= loop[1]
                    elif loop[0] == 4:
                        OYu[operand][level] *= loop[1]
                    elif loop[0] == 3:
                        OXu[operand][level] *= loop[1]
                    elif loop[0] == 2:
                        FYu[operand][level] *= loop[1]
                    elif loop[0] == 1:
                        FXu[operand][level] *= loop[1]
                    else:
                        raise IndexError('The loop index can only be values from "1" to "7".')

                    unroll_size[operand][level] *= loop[1]

        ''' 
        unit_count is calculated by multiplying all the spatial loop unrolling index values (unroll_size) for W/I/O 
        from current level to top level.

        Using ().item() to change datatype from numpy int64 to python default int.
        '''

        for operand in ['W', 'I', 'O']:
            for level in range(loop_levels[operand]):
                unit_count[operand][level] = (np.prod(unroll_size[operand][level:loop_levels[operand]])).item()

        ''' 
        unit_unique is calculated by multiplying all the relevant spatial loop unrolling index values for W/I/O 
        from current level to top level.

        buf_replicate = unit_count/unit_unique
        
        Using // (Floor Division) here to get an integer result out from division.
        '''

        for level in range(loop_levels['W']):
            unit_unique['W'][level] = (np.prod(Ku['W'][level:loop_levels['W']] +
                                               Cu['W'][level:loop_levels['W']] +
                                               FXu['W'][level:loop_levels['W']] +
                                               FYu['W'][level:loop_levels['W']])).item()

            unit_duplicate['W'][level] = unit_count['W'][level] / unit_unique['W'][level]

        for level in range(loop_levels['I']):
            # only when both FY and OY are spatially unrolled, IYu should be calculated by the 'advanced' equation:
            # IY = SY * (OY - 1) + SFY * (FY - 1) + 1

            # when only one of them is spatially unrolled, IYu should be calculated by the 'basic' equation:
            # IY = OY + FY - 1

            # For example: when stride on IY dimension = 2:
            # OYu 4 & FYu 1 -> IY = OY + FY - 1 = 4 + 1 - 1 = 4
            # we need IX 1, 3, 5, 7. (4 elements in total)

            # OYu 4 & FYu 3 -> IY = SY * (OY - 1) + SFY * (FY - 1) + 1 = 2*(4-1)+1*(3-1)+1 = 9
            # we need IX 1, 2, 3, 4, 5, 6, 7, 8, 9. (9 elements in total)

            if OYu['I'][level] == 1 or FYu['I'][level] == 1:
                IYu = (np.prod(OYu['I'][level:loop_levels['I']]) +
                       np.prod(FYu['I'][level:loop_levels['I']]) - 1).item()
            else:
                IYu = (layer_loop_info['SY'] * (np.prod(OYu['I'][level:loop_levels['I']]) - 1) +
                       layer_loop_info['SFY'] * (np.prod(FYu['I'][level:loop_levels['I']]) - 1) + 1).item()

            if OXu['I'][level] == 1 or FXu['I'][level] == 1:
                IXu = (np.prod(OXu['I'][level:loop_levels['I']]) +
                       np.prod(FXu['I'][level:loop_levels['I']]) - 1).item()
            else:
                IXu = (layer_loop_info['SX'] * (np.prod(OXu['I'][level:loop_levels['I']]) - 1) +
                       layer_loop_info['SFX'] * (np.prod(FXu['I'][level:loop_levels['I']]) - 1) + 1).item()
            unit_unique['I'][level] = (np.prod(Bu['I'][level:loop_levels['I']] +
                                               Cu['I'][level:loop_levels['I']] +
                                               [IYu] + [IXu])).item()

            unit_duplicate['I'][level] = unit_count['I'][level] / unit_unique['I'][level]

        for level in range(loop_levels['O']):
            unit_unique['O'][level] = (np.prod(Bu['O'][level:loop_levels['O']] +
                                               Ku['O'][level:loop_levels['O']] +
                                               OXu['O'][level:loop_levels['O']] +
                                               OYu['O'][level:loop_levels['O']])).item()

            unit_duplicate['O'][level] = unit_count['O'][level] / unit_unique['O'][level]

        ''' 
        unit_serve_scope is calculated by dividing unit_count at current level by unit_count at one level above.
        
        data_serve_scope is calculated by dividing unit_duplicate at current level by unit_count at one level above.
        
        real_bw_boost can calculated by either dividing unit_unique at current level by unit_count at one level above,
        or by dividing unit_serve_scope by data_serve_scope element-wise.

        Note that the number of level here equals to total memory level. 
        MAC level is excluded naturally here.
        
        e.g. real_bw_boost      = [ 1,  2,  3,  4], 
             real_bw_boost_high = [ 1,  2,  6, 24],
             real_bw_boost_low  = [24, 24, 12,  4]
        '''

        for operand in ['W', 'I', 'O']:
            for level in range(int(loop_levels[operand] - 1)):
                unit_serve_scope[operand][level] = unit_count[operand][level] / unit_count[operand][level + 1]
                data_serve_scope[operand][level] = unit_duplicate[operand][level] / unit_duplicate[operand][level + 1]
                real_bw_boost[operand][level] = unit_unique[operand][level] / unit_unique[operand][level + 1]

        for operand in spatial_loop.keys():
            for level in range(int(loop_levels[operand] - 1)):
                real_bw_boost_high[operand][level] = (np.prod(real_bw_boost[operand][0:level + 1]))
                real_bw_boost_low[operand][level] = (np.prod(real_bw_boost[operand][level:loop_levels[operand]]))

        '''
        Simply extract spatial unrolling loops in spatial_loop_list.
        '''
        spatial_loop_list = []
        for loop_list in spatial_loop['W']:
            if not loop_list:
                continue
            else:
                for this_loop in loop_list:
                    spatial_loop_list.append(this_loop)

        '''
        Added for LOMA
        '''
        # Relevant loop type numbers for each operand
        relevant_loop_type_numbers = {'W': [1,2,5,6], 'I': [5,7], 'O': [3,4,6,7]}
        irrelevant_loop_type_numbers = {'W': [3,4,7], 'I': [], 'O': [1,2,5]}
        
        ## Extract the relevant/irrelevant loop unrolling for each operand
        su_relevant_size_dict = {'W': [], 'I': [], 'O': []}
        su_irrelevant_size_dict = {'W': [], 'I': [], 'O': []}
        # WEIGHT and OUTPUT and INPUT relevant
        for operand in ['W', 'O', 'I']:
            for level in range(0, len(spatial_loop[operand])): # start at 0 =  include MAC level
                su_relevant_size = 1
                su_irrelevant_size = 1
                for [loop_type_number, su_factor] in spatial_loop[operand][level]:
                    if loop_type_number in relevant_loop_type_numbers[operand]:
                        su_relevant_size *= su_factor
                    elif loop_type_number in irrelevant_loop_type_numbers[operand]:
                        su_irrelevant_size *= su_factor
                su_relevant_size_dict[operand].append(su_relevant_size)
                su_irrelevant_size_dict[operand].append(su_irrelevant_size)
        # INPUT partially relevant
        su_pr_size_dict_input = {1: [], 2: [], 3: [], 4: []} # 1 = FX, 2 = FY, 3 = OX, 4 = OY
        pr_loops = [1,2,3,4] # 1 = FX, 2 = FY, 3 = OX, 4 = OY
        for level in range(0, len(spatial_loop['I'])):
            su_pr_size = {1: 1, 2: 1, 3: 1, 4: 1}
            for [loop_type_number, su_factor] in spatial_loop[operand][level]:
                if loop_type_number in pr_loops:
                    su_pr_size[loop_type_number] *= su_factor
            for key in pr_loops:
                su_pr_size_dict_input[key].append(su_pr_size[key])

        self.Bu = Bu
        self.Ku = Ku
        self.Cu = Cu
        self.OYu = OYu
        self.OXu = OXu
        self.FYu = FYu
        self.FXu = FXu

        self.unroll_size = unroll_size
        self.unit_count = unit_count
        self.unit_unique = unit_unique
        self.unit_duplicate = unit_duplicate
        self.loop_levels = loop_levels

        self.unit_serve_scope = unit_serve_scope
        self.data_serve_scope = data_serve_scope
        self.real_bw_boost = real_bw_boost
        self.real_bw_boost_high = real_bw_boost_high
        self.real_bw_boost_low = real_bw_boost_low

        self.spatial_loop_list = spatial_loop_list

        self.su_relevant_size_dict = su_relevant_size_dict
        self.su_irrelevant_size_dict = su_irrelevant_size_dict
        self.su_pr_size_dict_input = su_pr_size_dict_input


    @classmethod
    def extract_loop_info(cls, spatial_loop, layer_loop_info):
        return cls(spatial_loop, layer_loop_info)

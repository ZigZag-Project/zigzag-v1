"""
Utilization information extraction and integration.

Including memory utilization at each level, MAC array utilization (spatial and temporal together).
"""

import numpy as np
from math import ceil as ceil
import copy


def flatten(L):
    for item in L:
        try:
            yield from flatten(item)
        except TypeError:
            yield item

def iterative_data_format_clean(original_dict):
    new_dict = {'W':[], 'I':[], 'O':[]}
    for operand in ['W', 'I', 'O']:
        for li in original_dict[operand]:
            new_dict[operand].append(li[0])
    return new_dict

class Utilization(object):

    def __init__(self, layer, temporal_loop, spatial_loop, loop, mac_array_info, mem_size_bit, mem_share,
                 mem_type, mac_array_stall, precision, mem_bw_bit, clk_domain={}):

        # spatial_loop is a list, in which spatial_loop[0] is the rounded one and spatial_loop[1] is the fractional one.
        if type(spatial_loop) != type([1]):
            spatial_loop = [spatial_loop, spatial_loop]


        mem_level = temporal_loop.loop_levels
        output_pre = loop.output_precision
        output_dis = loop.output_distinguish
        if mem_bw_bit is None:
            mem_bw_bit = {'W': [[1, 1]] * mem_level['W'],
                          'I': [[1, 1]] * mem_level['I'],
                          'O': [[1, 1]] * mem_level['O']}
        elif type(mem_bw_bit['W'][0][0]) in [list, tuple]:
            mem_bw_bit = iterative_data_format_clean(mem_bw_bit)
        elif clk_domain != {}:
            for operand in ['W', 'I', 'O']:
                for level in range(mem_level[operand]):
                    mem_bw_bit[operand][level][0] /= clk_domain[operand][level]
                    mem_bw_bit[operand][level][1] /= clk_domain[operand][level + 1]
        mem_utilize = {'W': [],
                       'I': [],
                       'O': []}

        '''
        mem_utilize: memory utilization is calculated by req_mem_size*100 (bit) / real_mem_size (bit). (Unit: percent)
        
        Note that for output, the precision for partial/final output are different. 
        So firstly, for certain level of output memory, whether it is used to store partial/final output is verified.
        
        Random level of memory sharing is supported.
        '''

        ''' Weight and Input '''

        for operand in ['W', 'I']:
            for level in range(mem_level[operand]):
                req_mem_size = precision[operand] * loop.req_mem_size[operand][level]
                real_mem_size = mem_size_bit[operand][level]
                mem_utilize[operand].append(req_mem_size / real_mem_size)

        ''' Output '''

        R_W_LOW = 0
        for level in range(mem_level['O']):

            if loop.mem_access_elem['O_partial'][level][R_W_LOW] == (0, 0):
                req_mem_size = precision['O_final'] * loop.req_mem_size['O'][level]

            else:
                req_mem_size = precision['O'] * loop.req_mem_size['O'][level]

            real_mem_size = mem_size_bit['O'][level]
            mem_utilize['O'].append(req_mem_size / real_mem_size)

        ''' Memory sharing '''
        mem_utilize_shared = copy.deepcopy(mem_utilize)
        if mem_share:
            for idx, shared_mem_list in mem_share.items():
                va = 0
                for operand, lv in shared_mem_list:
                    va += mem_utilize[operand][lv]
                for operand, lv in shared_mem_list:
                    mem_utilize_shared[operand][lv] = va

        for operand in ['W', 'I', 'O']:
            for le, mem_utilize_shared_single in enumerate(mem_utilize_shared[operand]):
                if mem_utilize_shared_single > 1:
                    raise ValueError('%s memory level %d is too small to hold assigned loops.'%(operand, le))

        '''
        mac_utilize_spatial: MAC array utilization is spatially dropped in two cases:
        
        Case 1: When the spatially-unrolled loop index (w/wo replication) is not equal to
        the corresponding MAC array dimension. 
        During the whole computation, part of the array is idle.
        
            E.g. (without replication) unrolling ['C' 3 times] on a array dimension of '4'. 
            During the whole computation, 1 out of 4 PEs are idle.
            
            E.g. (with replication) unrolling ['C' 3 times & 'OX' 3 times] on a array dimension of '10'.
            During the whole computation, 1 out of 10 PEs are idle.
        
        Case 2: When the total size of the loop index, which is being spatially unrolled (w/wo replication), 
        is not a multiple value of the corresponding MAC array dimension. 
        During the last array-mapping cycles, part of the array is idle.
        
            E.g. (without replication) total K = 61 and Ku = 4 (on a array dimension of '4'). 
            During the last array-mapping cycles, 3 out of 4 PEs are idle.
            
            E.g. (with replication) total K = 61, OX = 28 and Ku = 3, OXu = 3 (on a array dimension of '9'). 
            During the last array-mapping cycles of Ku, 3 out of 4 PEs are idle.
            During the last array-mapping cycles of OXu, 3 out of 4 PEs are idle.
            Thus, the array utilization drop is the result of both and need to be multiplied together. 
        
        Case 1&2 can be added on top of each other, i.e. multiplying utilization values together.
        '''

        # spatial_loop is a list, in which spatial_loop[0] is the rounded one and spatial_loop[1] is the fractional one.

        ''' Case 1 '''
        num_of_used_mac = spatial_loop[1].unit_count['W'][0]
        num_of_actual_mac = np.prod(mac_array_info['array_size']).item()
        mac_utilize_spatial1 = num_of_used_mac / num_of_actual_mac

        # ''' Case 2 '''
        # index = 0
        # value = 1
        # for unrolled_loop in spatial_loop[1].spatial_loop_list:
        #     quotient = layer.size_list[unrolled_loop[index]] // unrolled_loop[value]
        #     quotient_ceiling = (layer.size_list[unrolled_loop[index]] + unrolled_loop[value] - 1) // unrolled_loop[value]
        #     remainder = layer.size_list[unrolled_loop[index]] % unrolled_loop[value]
        #     mac_utilize_spatial2 = (quotient + (remainder / unrolled_loop[value])) / quotient_ceiling
        mac_utilize_spatial2 = 1

        mac_utilize_spatial = mac_utilize_spatial1 * mac_utilize_spatial2
        if mac_utilize_spatial1 > 1 or mac_utilize_spatial2 > 1:
            raise ValueError("Your spatial-unrolling size is larger than your array size!")

        # TODO this is a temporary solution for calculating temporal mac utilization drop based on mac_array_stall.
        '''
        mac_utilize_temporal: MAC array utilization could temporally dropped at certain moments:
        
        Moment 1: (computing schedule limited) --- mac_utilize_temporal1
            In the beginning/end of the computation, the array is activated/deactivated part by part. 
            E.g. TPU-like computation, activating PEs row by row when starting.
        
        Moment 2: (memory bandwidth limited) --- mac_utilize_temporal2
            In the middle of the computation, when certain operand's stationary cycles end, 
            MAC array may stall certain cycles to transfer data in and out.
        '''

        ''' Moment 1 '''
        mac_utilize_temporal1 = 1
        if mac_array_stall['systolic'] == 1 or mac_array_stall['systolic'] == 2:
            ''' row-by-row (1) or column-by-column (2) '''
            mac_utilize_temporal1 = temporal_loop.total_cycles / \
                                    (temporal_loop.total_cycles +
                                     mac_array_info['array_size'][mac_array_stall['systolic']-1])

        elif mac_array_stall['systolic'] == 3:
            ''' diagonally (3) '''
            row = 0
            column = 1
            mac_utilize_temporal1 = temporal_loop.total_cycles / \
                                    (temporal_loop.total_cycles +
                                     mac_array_info['array_size'][row] + mac_array_info['array_size'][column] - 1)

        ''' Moment 2 '''
        # TODO memory schedule optimization

        ''' Before computation starts, # of clock cycle for loading data (W/I).  '''

        '''
        Case 1:
        From outside to the whole memory system (W/I), if the top memory level wrt bw is not set to math.inf.
        (The # of cc of loading the top level memory is taking into account.)
        
        Case 2:
        From the top level memory (W/I) to all memory (W/I) below, if the top memory level wrt bw is set to math.inf.
        (The # of cc of loading the top level memory isn't taking into account.)
        '''
        cc_load = {'W': [],
                   'I': []}
        rd_bw = 0
        wr_bw = 1
        for operand in ['W', 'I']:
            for level in range(mem_level[operand]):
                if level == mem_level[operand] - 1:
                    '''Don't count the loading cc for the top level mem.'''
                    # cc_load[operand].append(0)
                    '''Count the loading cc for the top level mem.'''
                    # for top-level memory loading, the loading cycle only depends on its writing bw.
                    cc_load[operand].append(ceil(loop.req_mem_size[operand][level] * precision[operand] /
                                                 mem_bw_bit[operand][level][wr_bw]))

                else:
                    # for non-top-level memory loading, the loading cycle depends on the current level
                    # memory's writing bw and above level memory's reading bw (taking the bw_boost factor into account).
                    cc_load[operand].append(ceil(loop.req_mem_size[operand][level] * precision[operand] /
                                                 min(mem_bw_bit[operand][level][wr_bw],
                                                     mem_bw_bit[operand][level + 1][rd_bw] /
                                                     spatial_loop[0].real_bw_boost[operand][level + 1])))

        cc_load_comb = {'W': cc_load['W'][-1],
                        'I': cc_load['I'][-1]}
        for operand in ['W', 'I']:
            if cc_load[operand][mem_level[operand] - 1] == 0:
                # without top-level memory loading
                for level in reversed(range(mem_level[operand])):
                    if level == 0:
                        break
                    if mem_type[operand][level] in [2, 3]:
                        # dual-port memory can be written and read in parallel
                        cc_load_comb[operand] = max(cc_load_comb[operand],
                                                    cc_load[operand][level - 1] + (mem_level[operand] - level - 1))
                    else:
                        cc_load_comb[operand] = cc_load_comb[operand] + cc_load[operand][level - 1]
            else:
                # with top-level memory loading
                for level in reversed(range(mem_level[operand])):
                    if level == 0:
                        break
                    if mem_type[operand][level] in [2, 3]:
                        # dual-port memory can be written and read in parallel
                        cc_load_comb[operand] = max(cc_load_comb[operand],
                                                    cc_load[operand][level - 1] + (mem_level[operand] - level))
                    else:
                        cc_load_comb[operand] = cc_load_comb[operand] + cc_load[operand][level - 1]

        cc_load_tot = max(cc_load_comb['W'], cc_load_comb['I'])

        if mem_share:
            for idx, shared_mem in mem_share.items():
                mem_share_collect = []
                for mem in shared_mem:
                    mem_share_collect.extend(mem)
                if 'I' in mem_share_collect and 'W' in mem_share_collect:
                    cc_load_tot += min(cc_load['W'][mem_share_collect[mem_share_collect.index('W') + 1]],
                                       cc_load['I'][mem_share_collect[mem_share_collect.index('I') + 1]])

        ''' 
        During computation, # of stalled clock cycle for the whole nested loop.
        stall_cc describes the edge event (between two levels) !!!
        
        Format: [a, [b,c], [d,e], ...]
        'a' means the stall during the innermost mem talks to MAC.
        [b, c] means the stall during the innermost mem talks to its above mem.
        'b' is the stall caused by the innermost mem.
        'c' is the stall caused by its above mem.
        '''
        stall_cc = {'W': [],
                    'I': [],
                    'O': []}

        '''
        trans_time: Data transmitting time for req_inst_mem_bw, assuming independent memory system for W/I/O.
        (ideal duration, ideal period)
        Input/Weight: duration is, at each continuous data transfer time block, 
        the # of cc for current level of memory to write to the level below.
        
        E.g. [1,3] means every 3 cycles, the current level of memory writes to below level 1 cycle.
        NOTE: Output need to be treat differently since psum can cause bidirectional data movement. 
        Use output_dis as flag to distinguish.
        (not using output_pre to distinguish because something people put the same precision to psum and fsum)
        
        trans_time describes the edge event (between two levels) !!!
        '''
        # initializing trans_time with [1, 1] indicates that the lowest level of memory talk to MAC all the time (every clock cycle).

        trans_time = {'W': [[1, 1]],
                      'I': [[1, 1]],
                      'O': [[1, 1]]}

        for operand in ['W', 'I', 'O']:
            for level in range(mem_level[operand]):
                if mem_type[operand][level] == 2:
                    duration = temporal_loop.loop_cycles[operand][level] / loop.top_ir[operand][level]
                else:
                    duration = temporal_loop.loop_cycles[operand][level]
                period = temporal_loop.loop_cycles[operand][level]
                trans_time[operand].append([duration, period])

        req_aver_bw = {}
        req_aver_bw_bit = {}
        req_inst_bw = {}
        req_inst_bw_bit = {}
        elem_count = 0
        cycle_count = 1
        rd = 0
        wr = 1
        to_low = 0
        to_high = 1
        for operand, li in loop.req_aver_mem_bw.items():
            req_aver_bw[operand] = []
            req_aver_bw_bit[operand] = []
            for level, rd_wr_bw in enumerate(li):
                if not rd_wr_bw:
                    rd_wr_bw = [(0, 1), (0, 1)]
                if not rd_wr_bw[rd]:
                    rd_wr_bw[rd] = (0, 1)
                if not rd_wr_bw[wr]:
                    rd_wr_bw[wr] = (0, 1)
                to_low_bw = rd_wr_bw[rd][elem_count] / rd_wr_bw[rd][cycle_count]
                to_high_bw = rd_wr_bw[wr][elem_count] / rd_wr_bw[wr][cycle_count]
                req_aver_bw[operand].append([to_low_bw, to_high_bw])
                if operand == 'O':
                    req_aver_bw_bit[operand].append([to_low_bw * output_pre[level][to_low],
                                                     to_high_bw * output_pre[level][to_high]])
                elif operand == 'O_partial':
                    req_aver_bw_bit[operand].append([to_low_bw * precision['O'],
                                                     to_high_bw * precision['O']])
                else:
                    req_aver_bw_bit[operand].append([to_low_bw * precision[operand],
                                                     to_high_bw * precision[operand]])

        for operand, li in loop.req_inst_mem_bw.items():
            req_inst_bw[operand] = []
            req_inst_bw_bit[operand] = []
            for level, rd_wr_bw in enumerate(li):
                if not rd_wr_bw:
                    rd_wr_bw = [(0, 1), (0, 1)]
                if not rd_wr_bw[rd]:
                    rd_wr_bw[rd] = (0, 1)
                if not rd_wr_bw[wr]:
                    rd_wr_bw[wr] = (0, 1)
                to_low_bw = rd_wr_bw[rd][elem_count] / rd_wr_bw[rd][cycle_count]
                to_high_bw = rd_wr_bw[wr][elem_count] / rd_wr_bw[wr][cycle_count]
                req_inst_bw[operand].append([to_low_bw, to_high_bw])
                if operand == 'O':
                    req_inst_bw_bit[operand].append([to_low_bw * output_pre[level][to_low],
                                                     to_high_bw * output_pre[level][to_high]])
                elif operand == 'O_partial':
                    req_inst_bw_bit[operand].append([to_low_bw * precision['O'],
                                                     to_high_bw * precision['O']])
                else:
                    req_inst_bw_bit[operand].append([to_low_bw * precision[operand],
                                                     to_high_bw * precision[operand]])

        '''
        req_mem_bw depends on memory type
        req_mem_bw describes vertex event (focus on each level) !!!
        '''
        req_mem_bw = {}
        req_mem_bw_bit = {}

        duration = 0
        period = 1
        rd = 0
        wr = 1
        for operand in ['W', 'I']:
            req_mem_bw[operand] = []
            req_mem_bw_bit[operand] = []
            for level in range(mem_level[operand]):
                if mem_level[operand] == 1:
                    ''' consider the only memory level talking to MAC '''
                    req_mem_bw_rd = req_aver_bw[operand][level][rd]
                    req_mem_bw[operand].append([req_mem_bw_rd, 0])

                    req_mem_bw_rd_bit = req_aver_bw_bit[operand][level][rd]
                    req_mem_bw_bit[operand].append([req_mem_bw_rd_bit, 0])

                    data_block_bit = req_mem_bw_rd_bit * trans_time[operand][level][duration]
                    stall_cc[operand].append([(data_block_bit / mem_bw_bit[operand][level][rd] -
                                               trans_time[operand][level][duration]) *
                                              temporal_loop.total_cycles / trans_time[operand][level][period]])
                elif level == mem_level[operand] - 1:
                    ''' consider the top level of memory talking to the below level of memory '''
                    if mem_type[operand][level - 1] == 2:
                        req_mem_bw_wr = req_inst_bw[operand][level - 1][wr]
                        req_mem_bw[operand][-1].append(req_mem_bw_wr)
                        req_mem_bw_wr_bit = req_inst_bw_bit[operand][level - 1][wr]
                        req_mem_bw_bit[operand][-1].append(req_mem_bw_wr_bit)

                        req_mem_bw_rd = req_inst_bw[operand][level][rd]
                        req_mem_bw[operand].append([req_mem_bw_rd, 0])
                        req_mem_bw_rd_bit = req_inst_bw_bit[operand][level][rd]
                        req_mem_bw_bit[operand].append([req_mem_bw_rd_bit, 0])

                    else:
                        req_mem_bw_wr = req_aver_bw[operand][level - 1][wr]
                        req_mem_bw[operand][-1].append(req_mem_bw_wr)
                        req_mem_bw_wr_bit = req_aver_bw_bit[operand][level - 1][wr]
                        req_mem_bw_bit[operand][-1].append(req_mem_bw_wr_bit)

                        req_mem_bw_rd = req_aver_bw[operand][level][rd]
                        req_mem_bw[operand].append([req_mem_bw_rd, 0])
                        req_mem_bw_rd_bit = req_aver_bw_bit[operand][level][rd]
                        req_mem_bw_bit[operand].append([req_mem_bw_rd_bit, 0])

                    '''
                    write to the below level memory (the object is the below level memory).
                    data_block_wr: data block being written.
                    stall_cc_wr: stalled clock cycle come from writing.
                    '''

                    data_block_wr_bit = req_mem_bw_wr_bit * trans_time[operand][level][duration]
                    stall_cc_wr = (data_block_wr_bit / mem_bw_bit[operand][level - 1][wr] -
                                   trans_time[operand][level][duration]) * \
                                  temporal_loop.total_cycles / trans_time[operand][level][period]

                    '''
                    read from the current level memory (the object is the current level memory).
                    data_block_re: data block being read.
                    stall_cc_re: stalled clock cycle come from reading.
                    '''

                    data_block_rd_bit = req_mem_bw_rd_bit * trans_time[operand][level][duration]
                    stall_cc_rd = (data_block_rd_bit / mem_bw_bit[operand][level][rd] -
                                   trans_time[operand][level][duration]) * \
                                  temporal_loop.total_cycles / trans_time[operand][level][period]

                    stall_cc[operand].append([stall_cc_wr, stall_cc_rd])

                elif level == 0:
                    ''' consider the lowest level of memory talking to MAC '''
                    req_mem_bw_rd = req_aver_bw[operand][level][rd]
                    req_mem_bw[operand].append([req_mem_bw_rd])

                    req_mem_bw_rd_bit = req_aver_bw_bit[operand][level][rd]
                    req_mem_bw_bit[operand].append([req_mem_bw_rd_bit])

                    data_block_bit = req_mem_bw_rd_bit * trans_time[operand][level][duration]
                    stall_cc[operand].append([(data_block_bit / mem_bw_bit[operand][level][rd] -
                                               trans_time[operand][level][duration]) *
                                              temporal_loop.total_cycles / trans_time[operand][level][period]])
                else:
                    ''' consider the current level of memory talking to the below level of memory '''
                    if mem_type[operand][level - 1] == 2:
                        req_mem_bw_wr = req_inst_bw[operand][level - 1][wr]
                        req_mem_bw[operand][-1].append(req_mem_bw_wr)
                        req_mem_bw_wr_bit = req_inst_bw_bit[operand][level - 1][wr]
                        req_mem_bw_bit[operand][-1].append(req_mem_bw_wr_bit)

                        req_mem_bw_rd = req_inst_bw[operand][level][rd]
                        req_mem_bw[operand].append([req_mem_bw_rd])
                        req_mem_bw_rd_bit = req_inst_bw_bit[operand][level][rd]
                        req_mem_bw_bit[operand].append([req_mem_bw_rd_bit])

                    else:
                        req_mem_bw_wr = req_aver_bw[operand][level - 1][wr]
                        req_mem_bw[operand][-1].append(req_mem_bw_wr)
                        req_mem_bw_wr_bit = req_aver_bw_bit[operand][level - 1][wr]
                        req_mem_bw_bit[operand][-1].append(req_mem_bw_wr_bit)

                        req_mem_bw_rd = req_aver_bw[operand][level][rd]
                        req_mem_bw[operand].append([req_mem_bw_rd])
                        req_mem_bw_rd_bit = req_aver_bw_bit[operand][level][rd]
                        req_mem_bw_bit[operand].append([req_mem_bw_rd_bit])

                    '''
                    write to the below level memory (the object is the below level memory).
                    data_block_wr: data block being written.
                    stall_cc_wr: stalled clock cycle come from writing.
                    '''

                    data_block_wr_bit = req_mem_bw_wr_bit * trans_time[operand][level][duration]
                    stall_cc_wr = (data_block_wr_bit / mem_bw_bit[operand][level - 1][wr] -
                                   trans_time[operand][level][duration]) * \
                                  temporal_loop.total_cycles / trans_time[operand][level][period]

                    '''
                    read from the current level memory (the object is the current level memory).
                    data_block_re: data block being read.
                    stall_cc_re: stalled clock cycle come from reading.
                    '''

                    data_block_rd_bit = req_mem_bw_rd_bit * trans_time[operand][level][duration]
                    stall_cc_rd = (data_block_rd_bit / mem_bw_bit[operand][level][rd] -
                                   trans_time[operand][level][duration]) * \
                                  temporal_loop.total_cycles / trans_time[operand][level][period]

                    stall_cc[operand].append([stall_cc_wr, stall_cc_rd])

        to_high = 1
        to_low = 0
        for operand in ['O_raw']:
            req_mem_bw[operand] = []
            req_mem_bw_bit[operand] = []
            for level in range(mem_level['O']):
                if level == mem_level['O'] - 1:
                    ''' consider the top level of memory talking to the below level of memory '''
                    if (mem_type['O'][level - 1] == 2 and mem_type['O'][level] == 1) \
                            and output_dis[level][to_low] == 'psum':

                        # When need to transmit partial sum for output
                        # between "dual-port no double buffering level" and "single-port double buffering level"
                        # the required bandwidth in between doubles (based on inst_bw).

                        # Object: (level-1) memory
                        req_mem_bw_H = 2 * req_inst_bw['O'][level - 1][to_high]
                        req_mem_bw[operand][-1].append(req_mem_bw_H)
                        req_mem_bw_H_bit = 2 * req_inst_bw_bit['O'][level - 1][to_high]
                        req_mem_bw_bit[operand][-1].append(req_mem_bw_H_bit)
                        # Object: top level memory
                        req_mem_bw_L = 2 * req_inst_bw['O'][level][to_low]
                        req_mem_bw[operand].append([req_mem_bw_L, 0])
                        req_mem_bw_L_bit = 2 * req_inst_bw_bit['O'][level][to_low]
                        req_mem_bw_bit[operand].append([req_mem_bw_L_bit, 0])

                    elif (mem_type['O'][level - 1] == 1 or mem_type['O'][level] == 1) \
                            and output_dis[level][to_low] == 'psum':

                        # When need to transmit partial sum for output between single-port double buffering level
                        # (either the current level or below level is single-port double buffering),
                        # the required bandwidth in between doubles (based on aver_bw).

                        # Object: (level-1) memory
                        req_mem_bw_H = 2 * req_aver_bw['O'][level - 1][to_high]
                        req_mem_bw[operand][-1].append(req_mem_bw_H)
                        req_mem_bw_H_bit = 2 * req_aver_bw_bit['O'][level - 1][to_high]
                        req_mem_bw_bit[operand][-1].append(req_mem_bw_H_bit)
                        # Object: top level memory
                        req_mem_bw_L = 2 * req_aver_bw['O'][level][to_low]
                        req_mem_bw[operand].append([req_mem_bw_L, 0])
                        req_mem_bw_L_bit = 2 * req_aver_bw_bit['O'][level][to_low]
                        req_mem_bw_bit[operand].append([req_mem_bw_L_bit, 0])

                    elif mem_type['O'][level - 1] == 2:
                        # Object: (level-1) memory
                        req_mem_bw_H = req_inst_bw['O'][level - 1][to_high]
                        req_mem_bw[operand][-1].append(req_mem_bw_H)
                        req_mem_bw_H_bit = req_inst_bw_bit['O'][level - 1][to_high]
                        req_mem_bw_bit[operand][-1].append(req_mem_bw_H_bit)
                        # Object: top level memory
                        req_mem_bw_L = req_inst_bw['O'][level][to_low]
                        req_mem_bw[operand].append([req_mem_bw_L, 0])
                        req_mem_bw_L_bit = req_inst_bw_bit['O'][level][to_low]
                        req_mem_bw_bit[operand].append([req_mem_bw_L_bit, 0])

                    else:
                        # Object: (level-1) memory
                        req_mem_bw_H = req_aver_bw['O'][level - 1][to_high]
                        req_mem_bw[operand][-1].append(req_mem_bw_H)
                        req_mem_bw_H_bit = req_aver_bw_bit['O'][level - 1][to_high]
                        req_mem_bw_bit[operand][-1].append(req_mem_bw_H_bit)
                        # Object: top level memory
                        req_mem_bw_L = req_aver_bw['O'][level][to_low]
                        req_mem_bw[operand].append([req_mem_bw_L, 0])
                        req_mem_bw_L_bit = req_aver_bw_bit['O'][level][to_low]
                        req_mem_bw_bit[operand].append([req_mem_bw_L_bit, 0])

                elif level == 0:
                    ''' consider the lowest level of memory talking to MAC '''
                    ''' p_sum here for sure, i.e. data flows in two direction.'''
                    if mem_type['O'][level] == 1:
                        ''' if the lowest level of memory is single-port double buffering, required bw doubles. '''
                        req_mem_bw_L = 2 * req_aver_bw['O'][level][to_low]
                        req_mem_bw[operand].append([req_mem_bw_L])

                        req_mem_bw_L_bit = 2 * req_aver_bw_bit['O'][level][to_low]
                        req_mem_bw_bit[operand].append([req_mem_bw_L_bit])

                    else:
                        ''' if the lowest level of memory is dual-port (wi/wo double buffering), 
                        required bw has no need to double. '''
                        req_mem_bw_L = req_aver_bw['O'][level][to_low]
                        req_mem_bw[operand].append([req_mem_bw_L])

                        req_mem_bw_L_bit = req_aver_bw_bit['O'][level][to_low]
                        req_mem_bw_bit[operand].append([req_mem_bw_L_bit])

                else:
                    ''' consider the current level of memory talking to the below level of memory '''
                    if (mem_type['O'][level - 1] == 2 and mem_type['O'][level] == 1) \
                            and output_dis[level][to_low] == 'psum':

                        # When need to transmit partial sum for output
                        # between "dual-port no double buffering level" and "single-port double buffering level"
                        # the required bandwidth in between doubles (based on inst_bw).

                        # Object: (level-1) memory
                        req_mem_bw_H = 2 * req_inst_bw['O'][level - 1][to_high]
                        req_mem_bw[operand][-1].append(req_mem_bw_H)
                        req_mem_bw_H_bit = 2 * req_inst_bw_bit['O'][level - 1][to_high]
                        req_mem_bw_bit[operand][-1].append(req_mem_bw_H_bit)
                        # Object: current level memory
                        req_mem_bw_L = 2 * req_inst_bw['O'][level][to_low]
                        req_mem_bw[operand].append([req_mem_bw_L])
                        req_mem_bw_L_bit = 2 * req_inst_bw_bit['O'][level][to_low]
                        req_mem_bw_bit[operand].append([req_mem_bw_L_bit])

                    elif (mem_type['O'][level - 1] == 1 or mem_type['O'][level] == 1) \
                            and output_dis[level][to_low] == 'psum':

                        # When need to transmit partial sum for output between single-port double buffering level
                        # (either the current level or below level is single-port double buffering),
                        # the required bandwidth in between doubles (based on aver_bw).

                        # Object: (level-1) memory
                        req_mem_bw_H = 2 * req_aver_bw['O'][level - 1][to_high]
                        req_mem_bw[operand][-1].append(req_mem_bw_H)
                        req_mem_bw_H_bit = 2 * req_aver_bw_bit['O'][level - 1][to_high]
                        req_mem_bw_bit[operand][-1].append(req_mem_bw_H_bit)
                        # Object: current level memory
                        req_mem_bw_L = 2 * req_aver_bw['O'][level][to_low]
                        req_mem_bw[operand].append([req_mem_bw_L])
                        req_mem_bw_L_bit = 2 * req_aver_bw_bit['O'][level][to_low]
                        req_mem_bw_bit[operand].append([req_mem_bw_L_bit])

                    elif mem_type['O'][level - 1] == 2:
                        # Object: (level-1) memory
                        req_mem_bw_H = req_inst_bw['O'][level - 1][to_high]
                        req_mem_bw[operand][-1].append(req_mem_bw_H)
                        req_mem_bw_H_bit = req_inst_bw_bit['O'][level - 1][to_high]
                        req_mem_bw_bit[operand][-1].append(req_mem_bw_H_bit)
                        # Object: current level memory
                        req_mem_bw_L = req_inst_bw['O'][level][to_low]
                        req_mem_bw[operand].append([req_mem_bw_L])
                        req_mem_bw_L_bit = req_inst_bw_bit['O'][level][to_low]
                        req_mem_bw_bit[operand].append([req_mem_bw_L_bit])

                    else:
                        # Object: (level-1) memory
                        req_mem_bw_H = req_aver_bw['O'][level - 1][to_high]
                        req_mem_bw[operand][-1].append(req_mem_bw_H)
                        req_mem_bw_H_bit = req_aver_bw_bit['O'][level - 1][to_high]
                        req_mem_bw_bit[operand][-1].append(req_mem_bw_H_bit)
                        # Object: current level memory
                        req_mem_bw_L = req_aver_bw['O'][level][to_low]
                        req_mem_bw[operand].append([req_mem_bw_L])
                        req_mem_bw_L_bit = req_aver_bw_bit['O'][level][to_low]
                        req_mem_bw_bit[operand].append([req_mem_bw_L_bit])

        ''' 
        Compute stalled cycle from Output.
        data_block_bl_bit: data block transferred at the below edge (current level mem talks to the below level mem).
        data_block_al_bit: data block transferred at the above edge (current level mem talks to the above level mem).
        '''
        for lv, bw_list in enumerate(req_mem_bw_bit['O_raw']):
            if lv == 0:
                # psum at level 0 for sure.
                data_block_bl_bit = bw_list[0] * trans_time['O'][lv][duration]
                stall_cc_bl = (data_block_bl_bit / mem_bw_bit['O'][lv][to_low] -
                               trans_time['O'][lv][duration]) * \
                              temporal_loop.total_cycles / trans_time['O'][lv][period]
                stall_cc['O'].append([stall_cc_bl])

                data_block_al_bit = bw_list[1] * trans_time['O'][lv + 1][duration]
                if mem_size_bit['O'][lv] <= precision['O'] or mem_type['O'][lv] == 3:
                    # only can store one psum / dual-port double buffering, no need to use residual BW.
                    stall_cc_al = (data_block_al_bit / mem_bw_bit['O'][lv][to_high] -
                                   trans_time['O'][lv + 1][duration]) * \
                                  temporal_loop.total_cycles / trans_time['O'][lv + 1][period]
                    stall_cc['O'].append([stall_cc_al])

                else:
                    # if data need to be transferred bidirectionally, use residual BW.
                    if mem_bw_bit['O'][lv][to_high] - req_mem_bw_bit['O_raw'][lv][to_low] > 0:

                        # While talking to low level, current mem still has spare bw to talk to high level.
                        stall_cc_al_1 = (data_block_al_bit / (
                                mem_bw_bit['O'][lv][to_high] - req_mem_bw_bit['O_raw'][lv][to_low]) -
                                         trans_time['O'][lv + 1][duration]) * \
                                        temporal_loop.total_cycles / trans_time['O'][lv + 1][period]

                        # While talking to low level, current mem doesn't have spare bw to talk to high level.
                        # Thus computation stalls when it needs to talk to high level.
                        stall_cc_al_2 = trans_time['O'][lv + 1][duration] * temporal_loop.total_cycles / \
                                        trans_time['O'][lv + 1][period]
                        stall_cc['O'].append([min(stall_cc_al_1, stall_cc_al_2)])
                    else:
                        stall_cc_al_2 = trans_time['O'][lv + 1][duration] * temporal_loop.total_cycles / \
                                        trans_time['O'][lv + 1][period]
                        stall_cc['O'].append([stall_cc_al_2])

            else:
                # Not level 0
                data_block_bl_bit = bw_list[0] * trans_time['O'][lv][duration]
                stall_cc_bl = (data_block_bl_bit / mem_bw_bit['O'][lv][to_low] -
                               trans_time['O'][lv][duration]) * \
                              temporal_loop.total_cycles / trans_time['O'][lv][period]
                stall_cc['O'][-1].append(stall_cc_bl)

                if lv == mem_level['O'] - 1:
                    # ends here (when all the output are written to the top memory)
                    # because for now, we don't care how the top memory talk to the outside world.
                    break

                data_block_al_bit = bw_list[1] * trans_time['O'][lv + 1][duration]
                if output_dis[lv] == ('fsum', 'fsum') or mem_type['O'][lv] == 3:
                    # only final sum exists at current level, meaning data flows uni-directionally, same as W and I.
                    stall_cc_al = (data_block_al_bit / mem_bw_bit['O'][lv][to_high] -
                                   trans_time['O'][lv + 1][duration]) * \
                                  temporal_loop.total_cycles / trans_time['O'][lv + 1][period]
                    stall_cc['O'].append([stall_cc_al])
                else:
                    # if data need to be transferred bidirectionally, use residual BW.
                    if mem_bw_bit['O'][lv][to_high] - req_mem_bw_bit['O_raw'][lv][to_low] > 0:
                        # While talking to low level, current mem still has spare bw to talk to high level.
                        stall_cc_al_1 = (data_block_al_bit / (
                                mem_bw_bit['O'][lv][to_high] - req_mem_bw_bit['O_raw'][lv][to_low]) -
                                         trans_time['O'][lv + 1][duration]) * \
                                        temporal_loop.total_cycles / trans_time['O'][lv + 1][period]

                        # While talking to low level, current mem doesn't have spare bw to talk to high level.
                        # Thus computation stalls when it needs to talk to high level.
                        stall_cc_al_2 = (data_block_al_bit / mem_bw_bit['O'][lv][
                            to_high]) * temporal_loop.total_cycles / trans_time['O'][lv + 1][period]
                        stall_cc['O'].append([min(stall_cc_al_1, stall_cc_al_2)])
                    else:
                        stall_cc_al_2 = (data_block_al_bit / mem_bw_bit['O'][lv][
                            to_high]) * temporal_loop.total_cycles / trans_time['O'][lv + 1][period]
                        stall_cc['O'].append([stall_cc_al_2])

        ''' Take the memory sharing into account '''
        # Preparation step: edge event -> vertex event (so as to handle individual memory unit easily)
        stall_cc_temp = copy.deepcopy(stall_cc)
        stall_cc_flat = {'W': [], 'I': [], 'O': []}
        for operand in ['W', 'I', 'O']:
            stall_cc_temp[operand].append(0)
            stall_cc_flat[operand] = list(flatten(stall_cc_temp[operand]))

        stall_cc_vertex = {'W': [], 'I': [], 'O': []}
        for operand in ['W', 'I', 'O']:
            for idx, cc in enumerate(stall_cc_flat[operand]):
                if idx % 2 == 0:
                    stall_cc_vertex[operand].append([cc])
                else:
                    stall_cc_vertex[operand][-1].append(cc)

        # Formal step:
        stall_cc_vertex_share = copy.deepcopy(stall_cc_vertex)
        if mem_share:
            for idx, shared_mem_list in mem_share.items():
                mem_share_collect = []
                for mem in shared_mem_list:
                    mem_share_collect.extend(mem)

                stall_cc_update_L = 0
                stall_cc_update_H = 0
                # Initialize all to the (- total # of cycle in ideal case).
                stall_cc_single = {'W': [-temporal_loop.total_cycles, -temporal_loop.total_cycles],
                                   'I': [-temporal_loop.total_cycles, -temporal_loop.total_cycles],
                                   'O': [-temporal_loop.total_cycles, -temporal_loop.total_cycles]}
                for operand, lv in shared_mem_list:
                    stall_cc_single[operand] = stall_cc_vertex[operand][lv]

                if 'O' in mem_share_collect:
                    shared_output_lv = mem_share_collect[mem_share_collect.index('O') + 1]
                    if output_dis[shared_output_lv] == ('psum', 'psum'):
                        # (psum, psum)
                        for operand in ['W', 'I', 'O']:
                            stall_cc_update_L += temporal_loop.total_cycles + stall_cc_single[operand][to_low]
                            stall_cc_update_H += temporal_loop.total_cycles + stall_cc_single[operand][to_high]

                        stall_cc_update_L -= temporal_loop.total_cycles
                        stall_cc_update_H -= temporal_loop.total_cycles
                        stall_cc_vertex_share['O'][shared_output_lv] = [stall_cc_update_L, stall_cc_update_H]
                        if 'I' in mem_share_collect:
                            shared_input_lv = mem_share_collect[mem_share_collect.index('I') + 1]
                            stall_cc_vertex_share['I'][shared_input_lv] = [stall_cc_update_L, stall_cc_update_H]
                        if 'W' in mem_share_collect:
                            shared_weight_lv = mem_share_collect[mem_share_collect.index('W') + 1]
                            stall_cc_vertex_share['W'][shared_weight_lv] = [stall_cc_update_L, stall_cc_update_H]

                    elif output_dis[shared_output_lv] == ('psum', 'fsum'):
                        # (psum, fsum)
                        for operand in ['W', 'I']:
                            stall_cc_update_L += temporal_loop.total_cycles + stall_cc_single[operand][to_low]
                            stall_cc_update_H += temporal_loop.total_cycles + stall_cc_single[operand][to_high]
                        for operand in ['O']:
                            stall_cc_update_L += temporal_loop.total_cycles + stall_cc_single[operand][to_low]

                        stall_cc_update_L -= temporal_loop.total_cycles
                        stall_cc_update_H -= temporal_loop.total_cycles
                        stall_cc_vertex_share['O'][shared_output_lv][to_low] = stall_cc_update_L
                        if 'I' in mem_share_collect:
                            shared_input_lv = mem_share_collect[mem_share_collect.index('I') + 1]
                            stall_cc_vertex_share['I'][shared_input_lv] = [stall_cc_update_L, stall_cc_update_H]
                        if 'W' in mem_share_collect:
                            shared_weight_lv = mem_share_collect[mem_share_collect.index('W') + 1]
                            stall_cc_vertex_share['W'][shared_weight_lv] = [stall_cc_update_L, stall_cc_update_H]

                    else:
                        # (fsum, fsum)
                        for operand in ['W', 'I']:
                            stall_cc_update_L += temporal_loop.total_cycles + stall_cc_single[operand][to_low]
                            stall_cc_update_H += temporal_loop.total_cycles + stall_cc_single[operand][to_high]

                        stall_cc_update_L -= temporal_loop.total_cycles
                        stall_cc_update_H -= temporal_loop.total_cycles
                        if 'I' in mem_share_collect:
                            shared_input_lv = mem_share_collect[mem_share_collect.index('I') + 1]
                            stall_cc_vertex_share['I'][shared_input_lv] = [stall_cc_update_L, stall_cc_update_H]
                        if 'W' in mem_share_collect:
                            shared_weight_lv = mem_share_collect[mem_share_collect.index('W') + 1]
                            stall_cc_vertex_share['W'][shared_weight_lv] = [stall_cc_update_L, stall_cc_update_H]

                else:
                    # 'O' NOT in mem_share_collect
                    for operand in ['W', 'I']:
                        stall_cc_update_L += temporal_loop.total_cycles + stall_cc_single[operand][to_low]
                        stall_cc_update_H += temporal_loop.total_cycles + stall_cc_single[operand][to_high]

                    stall_cc_update_L -= temporal_loop.total_cycles
                    stall_cc_update_H -= temporal_loop.total_cycles

                    if 'I' in mem_share_collect:
                        shared_input_lv = mem_share_collect[mem_share_collect.index('I') + 1]
                        stall_cc_vertex_share['I'][shared_input_lv] = [stall_cc_update_L, stall_cc_update_H]
                    if 'W' in mem_share_collect:
                        shared_weight_lv = mem_share_collect[mem_share_collect.index('W') + 1]
                        stall_cc_vertex_share['W'][shared_weight_lv] = [stall_cc_update_L, stall_cc_update_H]

        stall_cc_mem_share = {'W': [], 'I': [], 'O': []} #, 'output_direction': []}
        cc_mem_stall_list = {'W': [], 'I': [], 'O': []}
        for operand in ['W', 'I', 'O']:
            for idx, li in enumerate(stall_cc_vertex_share[operand]):
                if idx == 0:
                    stall_cc_mem_share[operand].append([li[0]])
                    cc_mem_stall_list[operand].append(li[0])
                else:
                    stall_cc_mem_share[operand].append([stall_cc_vertex_share[operand][idx - 1][1], li[0]])
                    cc_mem_stall_list[operand].extend([stall_cc_vertex_share[operand][idx - 1][1], li[0]])
        if clk_domain == {}:
            cc_mem_stall_tot = max(
                [ceil(max(cc_mem_stall_list['W'] + cc_mem_stall_list['I'] + cc_mem_stall_list['O'])), 0])
        else:
            # when W and I, O stall cannot be overlapped (AiMC cases).
            cc_mem_stall_tot = max([ceil(max(cc_mem_stall_list['W'][1:])), 0]) + max(
                [ceil(max(cc_mem_stall_list['I'] + cc_mem_stall_list['O'])), 0])

        # for j in output_dis:
        #     if j == ('psum', 'psum') or j == ('psum', 'fsum'):
        #         stall_cc_mem_share['output_direction'].append(['psum'])
        #     else:
        #         stall_cc_mem_share['output_direction'].append(['fsum'])

        ''' 
        After computation finishes, # of clock cycle for offloading output data to the top level 'O' memory. 
        
        * part of offloading stall has been included in the computation part 
          (when there is stall in 'O' memory during computation).
        
        * here we calculate the rest non-including part. 
          (when there is no stall in 'O' memory during computation.)
        '''

        # '''
        # offloading time = offloading-within-computation cc + offloading-after-computation cc.
        # offloading-after-computation cc is what we want to get here,
        # which contributes to offloading stall, i.e. utilization drop.
        # offload_within_compu_cc = total_cycles - offload_start_time
        # offload_start_time = SUM( (ir_loop_range - 1)*covered_cc_below )
        # '''
        #
        # ''' Calculating offload_within_compu_cc '''
        # l_type = 0
        # l_range = 1
        # offload_start_time = 0
        # for level, loops in enumerate(temporal_loop.temporal_loop['O']):
        #     if not loops:
        #         continue
        #     else:
        #         for idx, lp in enumerate(loops):
        #             if lp[l_type] in [1, 2, 5]:
        #                 offload_start_time += (lp[l_range] - 1) * loop.dt_bloc_below['O'][level][idx][1]
        #
        # offload_within_compu_cc = temporal_loop.total_cycles - (offload_start_time + 1)
        #
        # ''' Calculating offload_total_cc '''
        # # fsum_gen_speed: final output generating speed at the lowest Output memory level (unit: cc/bit).
        # level0 = 0
        # to_high = 1
        # data = 0
        # cycle = 1
        # data_count = spatial_loop[0].real_bw_boost_low['O'][level0]
        # fsum_gen_speed = loop.req_aver_mem_bw['O'][level0][to_high][cycle] / \
        #                  (loop.req_aver_mem_bw['O'][level0][to_high][data]*data_count*precision['O_final']
        #
        # mem_bw_cc_per_bit = {'W': [],
        #                      'I': [],
        #                      'O': []}
        # for operand in ['O']:
        #     for level, bw in enumerate(mem_bw_bit[operand]):
        #         mem_bw_cc_per_bit[operand].append([1/(bw[0]*loop.req_mem_count[operand][level]),
        #                                            1/(bw[1]*loop.req_mem_count[operand][level])])

        # level0 = 0
        # level1 = 1
        # from_low = 0
        # to_high = 1
        # data = 0
        #
        # if not loop.dt_bloc_below['O'][level0]:
        #     data_count_lowest_level = loop.dt_bloc_below['O'][level1][from_low][data] * \
        #                               precision['O_final']
        # else:
        #     data_count_lowest_level = loop.dt_bloc_below['O'][level0][to_high][data] * \
        #                               spatial_loop[0].real_bw_boost['O'][level0] * precision['O_final']
        #
        #
        # wr = 0
        # rd = 1
        # cc_offload_list = []
        # for level in reversed(range(mem_level['O'])):
        #     if level == 0:
        #         cc_offload_list.append(0)
        #     else:
        #         cc_offload_list.append(data_count_lowest_level /
        #                                min(mem_bw_bit['O'][level][wr] * spatial_loop.unit_unique['O'][level + 1],
        #                                    mem_bw_bit['O'][level - 1][rd] * spatial_loop.unit_unique['O'][level]))
        #
        # cc_offload_comb = cc_offload_list[0]
        # for level in range(mem_level['O']):
        #     if level == 0 or level == mem_level['O'] - 1:
        #         continue
        #     else:
        #         if mem_type['O'][level] in [2, 3]:
        #             cc_offload_comb = max(cc_offload_comb, cc_offload_list[level] + 1)
        #         else:
        #             cc_offload_comb = cc_offload_comb + cc_offload_list[level]
        #
        # cc_offload = cc_offload_comb

        # TODO here we assume output have to always obey the memory hierarchy, always offloading level by level.
        #  But actually, final output can jump from bottom level directly to the top level,
        #  skipping all the intermediate 'O' memory levels, in which case,
        #  last round of periodic stall cycles from intermediate levels can be excluded from computation stall part,
        #  and for offloading cc, the same principle applies.

        ''' Integrate required memory bandwidth '''
        # combine to_high & to_low Output memory (read and write) BW
        req_mem_bw_bit['O'] = []
        for lv, bw_list in enumerate(req_mem_bw_bit['O_raw']):
            if mem_size_bit['O'][lv] <= precision['O'] or mem_type['O'][lv] == 3:
                # only can store one psum OR dual-port double buffering
                req_mem_bw_bit['O'].append([max(bw_list), max(bw_list)])
            elif output_dis[lv] == ('psum', 'psum'):
                # append [total read bw, total write bw]
                req_mem_bw_bit['O'].append([bw_list[to_low] + bw_list[to_high], bw_list[to_low] + bw_list[to_high]])
            elif output_dis[lv] == ('psum', 'fsum'):
                req_mem_bw_bit['O'].append([bw_list[to_low] + bw_list[to_high], bw_list[to_low]])
            else:
                req_mem_bw_bit['O'].append([bw_list[to_high], bw_list[to_low]])

        req_sh_mem_bw_bit = copy.deepcopy(req_mem_bw_bit)
        if mem_share:
            for idx, shared_mem_list in mem_share.items():
                rd_bw_bit = 0
                wr_bw_bit = 0
                for operand, lv in shared_mem_list:
                    rd_bw_bit += req_mem_bw_bit[operand][lv][0]
                    wr_bw_bit += req_mem_bw_bit[operand][lv][1]
                for operand, lv in shared_mem_list:
                    req_sh_mem_bw_bit[operand][lv][0] = rd_bw_bit
                    req_sh_mem_bw_bit[operand][lv][1] = wr_bw_bit
        del req_sh_mem_bw_bit['O_raw']
        del req_mem_bw_bit['O_raw']

        '''
        max_req_bw_bit:
        Maximum amount of data that the current level mem/mac can receive/consume.
        to punish underutilized data fetching when actural BW > n*max_req_mem_bw.

        aftpac_mem_bw_bit: after-packing maximum required memory bw (n*max_req_mem_bw).
        It is multiple times of max_req_mem_bw, based on the actual memory bw, which is larger than max_req_mem_bw.
        '''

        # max_req_bw_bit starting from MAC level !!!
        max_req_bw_bit = {}
        for operand in ['I', 'W', 'O']:
            max_req_bw_bit[operand] = []
            for level, li in enumerate(req_mem_bw_bit[operand]):
                if level == 0:
                    max_req_bw_bit[operand].append(req_mem_bw_bit[operand][level][0])
                else:
                    max_req_bw_bit[operand].append(mem_bw_bit[operand][level - 1][1])
            max_req_bw_bit[operand].append(mem_bw_bit[operand][level][1])

        '''
        pun_factor: 
        punishment factor to punish underutilized data fetching bandwidth, which will create energy overhead. 
        It happens when actural BW > aftpac_req_mem_bw.
        '''
        pun_factor = {}
        for operand in ['W', 'I', 'O']:
            pun_factor[operand] = []
            for level, li in enumerate(req_mem_bw_bit[operand]):
                if max_req_bw_bit[operand][level] * spatial_loop[0].real_bw_boost[operand][level] >= \
                        mem_bw_bit[operand][level][0]:
                    pun_factor[operand].append(1)
                else:
                    aftpac_mem_bw_bit = mem_bw_bit[operand][level][0] - mem_bw_bit[operand][level][0] % (
                            max_req_bw_bit[operand][level] * spatial_loop[0].real_bw_boost[operand][level])
                    pun_factor[operand].append(mem_bw_bit[operand][level][0] / aftpac_mem_bw_bit)

        '''
        pun_factor_sh: punishment factor for shared memory.
        '''
        pun_factor_sh = []
        if mem_share:
            max_req_sh_bw_bit = 0
            for idx, shared_mem_list in mem_share.items():
                for mem in shared_mem_list:
                    operand = mem[0]
                    level = mem[1]
                    max_req_sh_bw_bit += max_req_bw_bit[operand][level] * spatial_loop[0].real_bw_boost[operand][level]
                if max_req_sh_bw_bit >= mem_bw_bit[operand][level][0]:
                    pun_factor_sh.append(1)
                else:
                    aftpac_mem_bw_bit = mem_bw_bit[operand][level][0] - \
                                        mem_bw_bit[operand][level][0] % max_req_sh_bw_bit
                    pun_factor_sh.append(mem_bw_bit[operand][level][0] / aftpac_mem_bw_bit)

        ideal_total_cycles = temporal_loop.total_cycles
        '''clock domian transfer'''
        if clk_domain != {}:
            ideal_total_cycles = temporal_loop.total_cycles * clk_domain['W'][0]

        ''' Final information integration '''
        # mac_utilize_temporal2 = ideal_total_cycles / \
        #                         (ideal_total_cycles + cc_load_tot + cc_mem_stall_tot + cc_offload)
        mac_utilize_temporal2 = ideal_total_cycles / \
                                (temporal_loop.total_cycles + cc_load_tot + cc_mem_stall_tot)
        mac_utilize_temporal = mac_utilize_temporal1 * mac_utilize_temporal2
        mac_utilize = mac_utilize_spatial * mac_utilize_temporal

        # Utilization without beginning stage data loading
        mac_utilize_temporal2_no_load = ideal_total_cycles / \
                                        (temporal_loop.total_cycles + cc_mem_stall_tot)
        mac_utilize_temporal_no_load = mac_utilize_temporal1 * mac_utilize_temporal2_no_load
        mac_utilize_no_load = mac_utilize_spatial * mac_utilize_temporal_no_load

        self.latency_tot = temporal_loop.total_cycles + cc_load_tot + cc_mem_stall_tot
        self.latency_no_load = temporal_loop.total_cycles + cc_mem_stall_tot

        self.cc_load = cc_load
        self.cc_load_comb = cc_load_comb
        self.cc_load_tot = cc_load_tot
        self.cc_mem_stall_tot = cc_mem_stall_tot

        self.stall_cc_mem_share = stall_cc_mem_share
        self.stall_cc = stall_cc

        self.pun_factor = pun_factor
        self.pun_factor_sh = pun_factor_sh
        self.req_aver_bw = req_aver_bw
        self.req_aver_bw_bit = req_aver_bw_bit

        self.req_inst_bw = req_inst_bw
        self.req_inst_bw_bit = req_inst_bw_bit

        # self.req_mem_bw = req_mem_bw
        self.req_mem_bw_bit = req_mem_bw_bit
        # self.req_sh_mem_bw = req_sh_mem_bw
        self.req_sh_mem_bw_bit = req_sh_mem_bw_bit

        self.mem_utilize = mem_utilize
        self.mem_utilize_shared = mem_utilize_shared

        self.mac_utilize_spatial = mac_utilize_spatial
        self.mac_utilize_temporal = mac_utilize_temporal
        self.mac_utilize = mac_utilize

        self.mac_utilize_temporal_no_load = mac_utilize_temporal_no_load
        self.mac_utilize_no_load = mac_utilize_no_load

    @classmethod
    def get_utilization(cls, layer, temporal_loop, spatial_loop, loop, mac_array_info, mem_size_bit, mem_share,
                        mem_type, mac_array_stall, precision, mem_bw_bit):
        return cls(layer, temporal_loop, spatial_loop, loop, mac_array_info, mem_size_bit, mem_share,
                   mem_type, mac_array_stall, precision, mem_bw_bit, clk_domain={})

import utils as u
import importlib.machinery
from tabulate import tabulate
from itertools import combinations
import numpy as np
import copy
import matplotlib.pyplot as plt
from math import floor


def no_tile_case(layer_spec, layers, pool_list):
    buffer_req = []
    base_buffer_req = []
    comp_cycles = []
    latency_offset = [0]
    latency_offset_cc = [0]
    cc = 10
    row_table = []
    tile_size = []
    pe_array_size = []
    for ii_l, l_index in enumerate(layers):
        l1 = u.Layer(layer_spec.layer_info[l_index],pool_list[ii_l]) 
        tile_size.append(tuple([l1.OX, l1.OY]))
        us = [tuple([1,l1.FX]),tuple([2,l1.FY]),tuple([5,l1.C]),tuple([6,l1.K])]
        pe_array_size.append([us[0][1] * us[1][1] * us[2][1], us[3][1]])
        buffer_req.append(u.depth_first_buffer_requirement(l1.C,l1.IX,l1.FX,l1.FY,l1.SX,l1.SY,l1.poolx,us,8))
        base_buffer_req.append(u.buffer_requirement(l1.C,l1.IX,l1.IY,8))
        comp_cycles.append(u.computing_cycles(l1, us, cc))
        if ii_l == 0:
            latency_offset.append(0)
            latency_offset_cc.append(0)
        else:
            latency_offset.append(u.depth_first_latency_offset_analog(l1,us,1))
            latency_offset_cc.append(u.depth_first_latency_offset_analog(l1,us,cc))
        cc *= u.cycle_to_output(l1, us)
        row_table.append([l_index, pool_list[ii_l], l1.IX,cc,latency_offset[-1], latency_offset_cc[-1], comp_cycles[-1],  buffer_req[-1]/(8*1024),base_buffer_req[-1]/(8*1024)])
        
    print(tabulate(row_table, headers=['LAYER','POOL','IX','CC','OFFSET','OFFSET*CC','COMPCYCLES','BUFFER[kB]','BASEBUFFER[kB]']))
    # print( sum([x/(8*1024) for x in buffer_req]), sum(latency_offset_cc)+comp_cycles[-1])
    return sum([x/(8*1024) for x in buffer_req]), sum(latency_offset_cc)+comp_cycles[-1], \
        buffer_req, tile_size, pe_array_size


def tile_case(layer_spec, layers, pool_list, start_pipe_index, pipe_length, tile_size):
    # TILE SIZE REFERS TO OX TILE SIZE!
    layer_list = []
    
    for ii_l, l_index in enumerate(layers):
        l1 = u.Layer(layer_spec.layer_info[l_index],pool_list[ii_l])
        layer_list.append(l1)
    # !!! THIS TS REFERS TO INPUTS IFMAP !!!
    ts =  u.tiling_size(layer_list, start_pipe_index, pipe_length, (tile_size,tile_size))
    print(ts)
    buffer_req = []
    comp_cycles = []
    comp_cycles_tile = []
    latency_offset = []
    latency_offset_cc = []
    cc = 10
    layer_list = []
    row_table = []
    pe_array_size = []
    
    for ii_l, l_index in enumerate(layers):#[start_pipe_index:start_pipe_index + pipe_length]):
        l1 = u.Layer(layer_spec.layer_info[l_index],pool_list[ii_l])
        layer_list.append(l1)
        
    for ii_l, l in enumerate(layer_list):
        if ii_l in list(range(start_pipe_index,start_pipe_index + pipe_length + 1)):
            l1 = copy.deepcopy(l)
            l1.IX = ts[ii_l][0]
            l1.IY = ts[ii_l][1]
            # IF IX == REAL IX ADD PADDING!
            # THIS OX DOES NOT INCLUDE THE POOLING BECAUSE IT IS USED ONLY
            # FOR COMPUTING *COMPUTING CLOCK CYCLES* WHICH REFER TO OX
            # BEFORE POOLING
            if l1.IX == l.IX:
                l1.OX = ((l1.IX - l1.FX + 2 * l1.PX) / (l1.SX) + 1)
                l1.OY = ((l1.IY - l1.FY + 2 * l1.PY) / (l1.SY) + 1) 
            else:
                l1.OX = ((l1.IX - l1.FX) / (l1.SX) + 1) 
                l1.OY = ((l1.IY - l1.FY) / (l1.SY) + 1) 
            print('IX, OX',l1.IX, l1.OX)
        else:
            l1 = copy.deepcopy(l)
        us = [tuple([1,l1.FX]),tuple([2,l1.FY]),tuple([5,l1.C]),tuple([6,l1.K])]
        pe_array_size.append([us[0][1] * us[1][1] * us[2][1], us[3][1]])
        buffer_req.append(u.depth_first_buffer_requirement(l1.C,l1.IX,l1.FX,l1.FY,l1.SX,l1.SY,l1.poolx,us,8))
        comp_cycles.append(u.computing_cycles(l1, us, cc))
        if ii_l == 0:
            latency_offset.append(0)
            latency_offset_cc.append(0)
        else:
            latency_offset.append(u.depth_first_latency_offset_analog(l1,us,1))
            latency_offset_cc.append(u.depth_first_latency_offset_analog(l1,us,cc))
        row_table.append([layers[ii_l], pool_list[ii_l], cc, ts[ii_l][0],l.IX ,\
                          latency_offset[-1], latency_offset_cc[-1], comp_cycles[-1], buffer_req[-1]/(8*1024)])
        
        us = [tuple([1,l1.FX]),tuple([2,l1.FY]),tuple([5,l1.C]),tuple([6,l1.K])]
        cc *= u.cycle_to_output(l1, us)
    print(tabulate(row_table, headers=['LAYER','POOL','CC','TS','IX','OFF.','OFF.*CC','COMPC','BUFF[kB]']))
    print('TOTAL BUFFER REQ', sum(buffer_req[1:])/(8*1024))
    # !!!!!TS REFERS TO IX,IY DIMENSIONS!!!!
    return buffer_req, latency_offset_cc, comp_cycles, ts, pe_array_size
    

def tiling_comb_sweep(layer_spec, layers, pool_list):
    layer_list = []
    l_ix_list = []
    for ii_l, l_index in enumerate(layers):
        l1 = u.Layer(layer_spec.layer_info[l_index],pool_list[ii_l])
        layer_list.append(l1)
        l_ix_list.append(l1.IX)
    tcomp_list = []
    buf_list = []
    tile_layer_list = []
    tile_ix_sizes = []
    tile_ox_sizes = []
    pe_array_sizes = []
    # loop through the layers in layer list
    # each iteration corresponds to a different tiling sequence length
    # for ii_l, l in enumerate(layer_list):
    l = copy.deepcopy(layer_list[-1])
    
    tsize_list = u.prime_factors(l.OX)
    tsize_list_temp = []

    # generate combinations of LPF so as to evaluate
    # all tilings of the last layer of the chunk
    for k in range(1,len(tsize_list)):
        t_combs = set(combinations(tsize_list,k))
        tsize_list_temp += list([np.prod(x) for x in t_combs])
    # add also the cases in which tile size = 1 and tile size = OX (no tiling)
    tsize_list_temp.append(1)
    tsize_list_temp.append(l.OX)
    tsize_list = copy.deepcopy(tsize_list_temp)
    print(tsize_list)
    # for a given sequence length and for a given tiling size
    # compute required buffer, latencies
    for tsox in tsize_list:
        # if ii_l == 0, then no offset (it is assumed that data comes directly from L1)
        # and no input buffer required (since data comes directly from L1)
        # if 1:#ii_l > 0:
        start_pipe = 0
        ts = (tsox - 1) * l.SX + l.FX 
        print("##########################")
        print('LAYER', ii_l + 1, 'TILE SIZE', tsox, 'START PIPE', start_pipe, 'PIPE LENGTH', ii_l - start_pipe)
        buffer_req, lat_off, comp_cycles, tile_sizes, pe_array = tile_case(layer_spec, layers, pool_list,start_pipe,ii_l - start_pipe+1,tsox)
        tile_steps = (floor(l.OX/tsox))**2
        # modify start of tiling if tiling is bigger than ifmap of certain layers
        start_pipe_aux = 0
        for ii_lx, lx in enumerate(layer_list):
            if lx.IX != tile_sizes[ii_lx][0]:
                start_pipe_aux = ii_lx
                break
        # total computations cycles required (without cc now)    
        total_comp = sum(lat_off[0:start_pipe_aux]) + \
            (sum(lat_off[start_pipe_aux:ii_l+1]) + comp_cycles[ii_l]) * tile_steps
        # the following if condition covers the case in which there is only a SINGLE layer in the sequence (pipe length = 1)
        if lat_off[ii_l+1:]:
            total_comp += sum(lat_off[ii_l+1:]) + comp_cycles[-1]
            # print('TOTAL_COMP',  sum(lat_off[0:start_pipe_aux]) ,'+', (sum(lat_off[start_pipe_aux:ii_l+1]) + comp_cycles[ii_l]) * tile_steps,'+', sum(lat_off[ii_l+1:]) + comp_cycles[-1],'=',total_comp)
        # else:
            # print('TOTAL_COMP',  sum(lat_off[0:start_pipe_aux]) ,'+', (sum(lat_off[start_pipe_aux:ii_l+1]) + comp_cycles[ii_l]) * tile_steps,'=',total_comp)

        tcomp_list.append(total_comp)
        buf_list.append(buffer_req)
        pe_array_sizes.append(pe_array)
        
        # TILE SIZES OX IS RETURNED
        # TILE SIZES IS USED TO UPDATE LAYER INFO FILE
        # IT IS REQUIRED THAT THE TS_OX DOES NOT INCLUDE POOL
        tile_sizes_ox = []
        for tsx in tile_sizes:
            if tsx == l.IX:
                ts_ox = (tsx[0] - l.FX + 2 * l.PX) / l.SX + 1
            else:
                ts_ox = (tsx[0] - l.FX) / l.SX + 1
            tile_sizes_ox.append(tuple([ts_ox, ts_ox]))
            tile_ix_sizes.append(copy.deepcopy(tile_sizes))
            tile_ox_sizes.append(copy.deepcopy(tile_sizes_ox))
                
    #buffer_size_kb = [x/(8*1024) for x in buf_list]
    
    return buf_list, tcomp_list, tile_ox_sizes, pe_array_sizes








# if buffer_req == min_buffer_req and total_comp < min_total_comp:
#                     min_total_comp = total_comp
#                     min_buffer_req = buffer_req
#                     best_ts = ts
#                     best_l = ii_l
#                     best_start_pipe = start_pipe
#                     best_pipe_length = ii_l - start_pipe_index
#                 if buffer_req < min_buffer_req:
#                     min_total_comp = total_comp
#                     min_buffer_req = buffer_req
#                     best_ts = ts
#                     best_l = ii_l
#                     best_start_pipe = start_pipe
#                     best_pipe_length = ii_l - start_pipe
                





# print()
# print('MIN BUFFER REQ',min_buffer_req/(8*1024))
# print('BEST L',best_l+1)
# print('BEST TS',best_ts)
# print('BEST START PIPE',best_start_pipe)
# print('BEST PIPE LENGTH', best_pipe_length)
# print('MIN TOTAL COMP',min_total_comp)
    

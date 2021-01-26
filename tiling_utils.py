import importlib.machinery
from tabulate import tabulate
from itertools import combinations
from math import floor, ceil
import numpy as np
import copy

class Layer:
    def __init__(self, layer, pool):
        self.FX = layer['FX']
        self.FY = layer['FY']
        self.OX = layer['OX']
        self.OY = layer['OY']
        self.IX = (layer['OX']-1)*layer['SX'] + layer['FX'] - 2 * layer['PX']
        self.IY = (layer['OY']-1)*layer['SY'] + layer['FY'] - 2 * layer['PY']
        self.C = layer['C']
        self.K = layer['K']
        self.B = layer['B']
        self.SX = layer['SX']
        self.SY = layer['SY']
        self.PX = layer['PX']
        self.PY = layer['PY']
        self.poolx = pool[0]
        self.poolstride = pool[1]

        
def depth_first_buffer_requirement(c, ix, fx, fy, sx, sy, poolx, us, precision):
    # COMPUTES BUFFER REQUIREMENT IF DEPTH FIRST
    # FOR BUFFER FOR *INPUTS* OF LAYER
    
    OXu = np.prod([x[1] for x in us if x[0] == 3]) 
    main_part = (c * ix) * (fy - sy)
    footer = c * (sy * poolx) * (fx + (poolx - 1) * sx + sx * (OXu - 1) * poolx)
    return (main_part + footer) * precision


def buffer_requirement(c,ix,iy,precision):
    return c * ix * iy * precision


def tiling_size(pipelined_layer_list, start_pipe_index, pipe_length, final_tile_size):
    # pll : pipelined layers list
    # ts  : tile size (TILE_X, TILE_Y) of OX,OY
    # computes the tiling size of the layers provided in pll
    # starting from the desired tile size of the last layer in pll
    # COVERS: FX, STRIDE; DOES NOT COVER: PADDING
    # OX of L-1 required to run L = (OX_{L} - 1) * SX_{L} + FX_{L} = OX_{L-1}

    # !!!!RETURNS TILES OF IX DIMENSIONS!!!!!!
    
    pll = pipelined_layer_list
    ts = final_tile_size
    spi = start_pipe_index
    pl = pipe_length
    tile_sizes = []
    
    for ii_l, l in enumerate(pll[0:start_pipe_index]):
        tile_sizes.append(tuple([l.IX,l.IY]))
    pl_list = []
    for ii_l, l in enumerate(pll[start_pipe_index: start_pipe_index + pipe_length]):
        pl_list.append(l)
    pl_list.reverse()
    if ts[0] != l.IX:
        tsx = tuple([(ts[0] - 1) * pl_list[0].SX + pl_list[0].FX, \
                     (ts[1] - 1) * pl_list[0].SY + pl_list[0].FY])
    else:
        tsx = tuple([(ts[0] - 1) * pl_list[0].SX + pl_list[0].FX - 2 * pl_list[0].PX, \
                     (ts[1] - 1) * pl_list[0].SY + pl_list[0].FY - 2 * pl_list[0].PY])
    tile_sizes.append(tsx)
    print(tile_sizes)
    ts_index = tile_sizes.__len__()
    for ii_l, l in enumerate(pl_list[1:]):
        if tile_sizes[ts_index - 1][0] != pl_list[ii_l].IX:
            ts_x1 = ((tile_sizes[ts_index - 1][0] - 1) * l.SX + l.FX - 1) * l.poolstride + l.poolx 
            ts_y1 = ((tile_sizes[ts_index - 1][1] - 1) * l.SY + l.FY - 1) * l.poolstride + l.poolx
        else:
            ts_x1 = ((tile_sizes[ts_index - 1][0] - 1) * l.SX + l.FX - 2 * l.PX - 1) * l.poolstride + l.poolx
            ts_y1 = ((tile_sizes[ts_index - 1][1] - 1) * l.SY + l.FY - 2 * l.PX - 1) * l.poolstride + l.poolx
        if ts_x1 <= pl_list[ii_l + 1].IX:
            tile_sizes.insert(ts_index - 1, tuple([ts_x1, ts_y1]))
        else:
            print('a', ts_x1, pl_list[ii_l + 1].IX, tile_sizes[ts_index-1][0], l.SX, l.poolx, l.FX, l.PX)
            tile_sizes.insert(ts_index - 1, tuple([pl_list[ii_l + 1].IX, pl_list[ii_l + 1].IY]))

    for ii_l, l in enumerate(pll[start_pipe_index+pipe_length:]):
        tile_sizes.append(tuple([l.IX,l.IY]))
    print(start_pipe_index, pipe_length)
    print(tile_sizes)
    return tile_sizes


def depth_first_latency_offset_analog(layer, unrolling_scheme, comp_cycles_previous):
    # Given unrolling scheme of L and specs of layer L and L-1
    # Computes the latency offset for layer L
    
    l = copy.deepcopy(layer)
    cc = comp_cycles_previous
    us = copy.deepcopy(unrolling_scheme)

    OXu = np.prod([x[1] for x in us if x[0] == 3])
    main_offset = l.IX * (l.FY - l.SY - l.PY)
    filter_offset = l.poolx * l.SY * (l.FX - l.PX + (l.poolx - 1) * l.SX + l.SX * (OXu - 1) * l.poolx)
    
    return (main_offset + filter_offset) * cc


def computing_cycles(layer, unrolling_scheme, cycle):

    l = copy.deepcopy(layer)
    us = copy.deepcopy(unrolling_scheme)
    cc = cycle
    l.K /= np.prod([x[1] for x in us if x[0] == 6])
    l.C /= np.prod([x[1] for x in us if x[0] == 5])
    l.OY /= np.prod([x[1] for x in us if x[0] == 4])
    l.OX /= np.prod([x[1] for x in us if x[0] == 3])
    l.FY /= np.prod([x[1] for x in us if x[0] == 2])
    l.FX /= np.prod([x[1] for x in us if x[0] == 1])
    res = l.K * l.C * l.OX * l.OY * l.FX * l.FY * cc
    
    return res


def cycle_to_output(layer, unrolling_scheme):
    l = copy.deepcopy(layer)
    us = copy.deepcopy(unrolling_scheme)
    OXu = np.prod([x[1] for x in us if x[0] == 3])
    l.K /= np.prod([x[1] for x in us if x[0] == 6])
    l.C /= np.prod([x[1] for x in us if x[0] == 5])
    l.FX /= np.prod([x[1] for x in us if x[0] == 2])
    l.FY /= np.prod([x[1] for x in us if x[0] == 1])
    return (l.K * l.C * l.FX * l.FY * l.SX * l.SY * l.poolx * l.poolx) / OXu
    

def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors



import utils as u
import tiling_funcs as tf
import importlib.machinery
from tabulate import tabulate
from itertools import combinations
import numpy as np
import copy
import matplotlib.pyplot as plt





if __name__ == "__main__":
    layer_path = "cost_model_input/layer/"
    layer_filename = "VGG16"
    rows = 1024
    cols = 512
    layer_spec = importlib.machinery.SourceFileLoader('%s' % (layer_filename),
                                                      '%s%s.py' % (layer_path, layer_filename)).load_module()

    ii_l = 1;
    if layer_filename == 'VGG16':
        layers = [1,2,3,4,5,6,7,8,9,10,11,12,13]
        pool_list = [[1,1],[2,2],[1,1],[2,2],[1,1],[1,1],[2,2],[1,1],[1,1],[2,2],[1,1],[1,1],[2,2],[1,1],[1,1],[1,1]]
    if layer_filename == 'AlexNet':
        layers = [1,2,3,4,5]
        pool_list = [[2,2],[2,2],[1,1],[1,1],[2,2]]
    if layer_filename == 'ResNet20':
        layers = list(range(1,30))
        pool_list = [[1,1]]*layers.__len__()
     

    pipe_length = 0
    tile_size = 1
    start_pipe_index = 0
    
    for chunk_size in range(1,len(layers)+1):
        print("############")
        print("CHUNK SIZE ", chunk_size)
        print("############")
        total_latency = 0
        layer_chunks_list = []
        pool_chunks_list = []
        for i in range(0, len(layers), chunk_size):
            slice_item = slice(i, i + chunk_size, 1)
            layer_chunks_list.append(layers[slice_item])
            pool_chunks_list.append(pool_list[slice_item])
        chunks_list = [[lc,pool_chunks_list[ii_lc]] for ii_lc, lc in enumerate(layer_chunks_list)]
        layer_chunk_pos = []
        for i in range(0, len(layers)):
            if chunk_size == 1:
                layer_chunk_pos.append(1)
            else:
                if i == (len(layers) - 1):
                    layer_chunk_pos.append(1)
                elif (i+1) % chunk_size == 0 and i != 0:
                    layer_chunk_pos.append(1)
                else:
                    layer_chunk_pos.append(0)
        buffer_req_list = []
        tcomp_list = []
        tile_sizes_ox_list = []
        pe_array_list = []
        
        for layer_chunk,pool_chunk  in chunks_list:
            print()
            print('LAYER CHUNK',layer_chunk)
            print('POOL CHUNK',pool_chunk)
            # b, t = tf.no_tile_case(layer_spec, layer_chunk, pool_chunk)
            buffer_req, tcomp, tile_sizes_ox, pe_array = tf.tiling_comb_sweep(layer_spec, layer_chunk, pool_chunk)
            # print(buffer_req)
            # print(tile_sizes_ox)
            # print(pe_array)
            buffer_req_list += copy.deepcopy(buffer_req)
            tcomp_list += copy.deepcopy(tcomp)
            tile_sizes_ox_list += copy.deepcopy(tile_sizes_ox)
            pe_array_list += copy.deepcopy(pe_array)
        # print()
        # print("CHUNK SIZE", chunk_size)
        # print(buffer_req_list)
        # print(tile_sizes_ox_list)
        # print(layer_chunk_pos)
        


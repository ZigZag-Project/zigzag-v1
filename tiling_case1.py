import utils as u
import tiling_funcs as tf
import input_scripts as isc
import importlib.machinery
from tabulate import tabulate
from itertools import combinations
import numpy as np
import copy
import matplotlib.pyplot as plt
import yaml

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

    layer_chunk = []
    for i in range(0, len(layers)):
        if i != len(layers) - 1:
            layer_chunk.append(0)
        else:
            layer_chunk.append(1)
            
    pipe_length = 0
    tile_size = 1
    start_pipe_index = 0
    b,t, buffer_req, tile_size, pe_array = tf.no_tile_case(layer_spec, layers, pool_list)
    print(pe_array)
    cfg = "no_tiling"
    isc.generate_zigzag_inputs(cfg, layer_filename, layers, layer_chunk, buffer_req, tile_size, pe_array)
    with open("settings_AlexNet_1_no_tiling.yaml") as f:
        list_doc = yaml.safe_load(f)

    # print(type(list_doc['layer_indices']))
    
    #buffer_req, tcomp, tile_sizes_ox, pe_array = tf.tiling_comb_sweep(layer_spec, layers, pool_list)
    # for ii_tsx,tsx in enumerate(tile_sizes_ox):
    #     cfg = "tiling_"+str(ii_tsx)
    #     isc.generate_zigzag_inputs(cfg, layer_filename, layers, layer_chunk, buffer_req[ii_tsx], tsx, pe_array[ii_tsx])

    # generate settings for each tile size, layer
    # tile_sizes_ox = []
    # for tsx in tile_sizes:
    #     tile_sizes_ox.append(tsx - l.FX + 2*l.PX)/l.SX + 1)
    #     print('TILE OX', tile_sizes_ox)
                
                

                








    # plt.scatter(b,t)
    # plt.xlabel('BUFFER SIZE [kB]')
    # plt.ylabel('COMP CYCLES [cc]')
    # plt.yscale('log')
    # plt.title('AlexNet')
    # plt.tight_layout()
    # plt.show()
    
    

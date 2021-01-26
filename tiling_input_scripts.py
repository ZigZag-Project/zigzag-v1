import yaml
import os
from math import log, ceil
import importlib.machinery

def layer_settings(layer_name, layer_origin, cfg, layer_indices, tile_size):
    layer_spec = importlib.machinery.SourceFileLoader('%s' % (layer_origin),
                                                      '%s.py' % (layer_origin)).load_module()
    layer_info = layer_spec.layer_info
    print(tile_size)
    if tile_size:
        for ii_l, l in enumerate(layer_indices):
            layer_info[l]['OX'] = int(tile_size[ii_l][0])
            layer_info[l]['OY'] = int(tile_size[ii_l][1])

    if not os.path.exists("input_files/"+layer_name+"_"+cfg):
        os.makedirs("input_files/"+layer_name+"_"+cfg)
    f = open("input_files/"+layer_name+"_"+cfg+"/"+layer_name+ "_" + cfg + ".py", "w")
    f.write('layer_info = ' + repr(layer_info) + '\n')
    f.close()

    
def settings_cfg(settings_origin, cfg, network, layer_index):
    with open(settings_origin + ".yaml") as f:
        settings_doc = yaml.safe_load(f)

    settings_doc["result_filename"] = network + "_" + cfg
    settings_doc["layer_filename"] = "./NN_layers/"+network
    settings_doc["layer_indices"] = [layer_index]

    if not os.path.exists("input_files/"+network+"_"+cfg):
        os.makedirs("input_files/"+network+"_"+cfg)
        
    with open("input_files/"+network+"_"+cfg +"/"+settings_origin + "_" + network + "_" + str(layer_index) + "_" + cfg  + ".yaml", "w") as f:
        yaml.dump(settings_doc, f)

        
def mapping_cfg(mapping_origin, us, cfg):
    with open("my_file.yaml") as f:
        list_doc = yaml.safe_load(f)

    for sense in list_doc:
        if sense["name"] == "sense2":
            sense["value"] = 1234

    with open("my_file.yaml", "w") as f:
        yaml.dump(list_doc, f)


def arch_cfg(arch_origin, cfg, network, layer_index, pe_array, input_buffer, output_buffer):
    with open(arch_origin + ".yaml") as f:
        list_doc = yaml.safe_load(f)

    list_doc['PE_array']['Col'] = pe_array[1]
    list_doc['PE_array']['Row'] = pe_array[0]
    
    memory_pool = {1:"sram1kb", 2:"sram2kb", 4:"sram4kb", 8:"sram8kb",\
                   16:"sram16kb", 32:"sram32kb", 64:"sram64kb", 128:"sram128kb"}
    
    if input_buffer:
        list_doc['memory_hierarchy']['buf_input']['memory_instance'] = memory_pool[2**ceil(log(input_buffer/8192,2))]
    else:
        list_doc['memory_hierarchy']['buf_input']['memory_instance'] = "sram1Mb"
        
    if output_buffer:
        list_doc['memory_hierarchy']['buf_output']['memory_instance'] = memory_pool[2**ceil(log(output_buffer/8192,2))]
    else:
        list_doc['memory_hierarchy']['buf_output']['memory_instance'] = "sram1Mb"

    if not os.path.exists("input_files/"+network+"_"+cfg):
        os.makedirs("input_files/"+network+"_"+cfg)
    
    with open("input_files/"+network+"_"+cfg+"/"+arch_origin + "_" + network + "_" + str(layer_index) + "_" + cfg + ".yaml", "w") as f:
        yaml.dump(list_doc, f)


def generate_zigzag_inputs(cfg, network, layer_indices, layer_chunks, buffer_req, tile_size, pe_array):

    # CFG MUST CONTAIN TILING/NO_TILING + TILE_SIZE_INDEX + LAYER_CHUNKS_INDEX
    layer_origin = "cost_model_input/layer/"+network
    layer_settings(network, layer_origin, cfg, layer_indices, tile_size)
    print(cfg, buffer_req)
    for ii_l, l in enumerate(layer_indices):
        settings_origin = "settings"
        settings_cfg(settings_origin, cfg, network, l)
        arch_origin = "architecture"
        if ii_l == 0:
            arch_cfg(arch_origin, cfg, network, l, pe_array[ii_l], None, buffer_req[ii_l+1])
        elif layer_chunks[ii_l - 1] == 1:    
            arch_cfg(arch_origin, cfg, network, l, pe_array[ii_l], None, buffer_req[ii_l+1])
        elif layer_chunks[ii_l] == 1:
            arch_cfg(arch_origin, cfg, network, l, pe_array[ii_l], buffer_req[ii_l], None)
        else:
            arch_cfg(arch_origin, cfg, network, l, pe_array[ii_l], buffer_req[ii_l], buffer_req[ii_l + 1])

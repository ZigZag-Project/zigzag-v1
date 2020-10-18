import yaml
import sys
from msg import MemoryNode, MemorySchemeNode, MemoryScheme


class InputSettings:

    def __init__(self, results_path, results_filename, layer_filename, layer_number, layer_parallel_processing, precision,
                 mac_array_info, mac_array_stall, mem_hierarchy_single_simulation, mem_scheme_parallel_processing,
                 mem_scheme_single, fixed_spatial_unrolling, spatial_unrolling_single, flooring_single,
                 fixed_temporal_mapping, temporal_mapping_single, tmg_search_method, temporal_mapping_multiprocessing,
                 drc_enabled, PE_RF_size_threshold, PE_RF_depth, CHIP_depth, max_area, utilization_rate_area,
                 memory_hierarchy_ratio, mem_pool, banking, L1_size, L2_size, unrolling_size_list, unrolling_scheme_list,
                 unrolling_scheme_list_text, memory_scheme_hint, spatial_utilization_threshold, spatial_unrolling_mode,
                 stationary_optimization_enable, su_parallel_processing, arch_search_result_saving, su_search_result_saving,
                 tm_search_result_saving, result_print_mode, im2col_enable):
        self.results_path = results_path
        self.results_filename = results_filename
        self.layer_filename = layer_filename
        self.layer_number = layer_number
        self.layer_parallel_processing = layer_parallel_processing
        self.precision = precision
        self.mac_array_info = mac_array_info
        self.mac_array_stall = mac_array_stall
        self.energy_over_utilization = True
        self.mem_hierarchy_single_simulation = mem_hierarchy_single_simulation
        self.mem_scheme_parallel_processing = mem_scheme_parallel_processing
        self.mem_scheme_single = mem_scheme_single
        self.fixed_spatial_unrolling = fixed_spatial_unrolling
        self.spatial_unrolling_single = spatial_unrolling_single
        self.flooring_single = flooring_single
        self.fixed_temporal_mapping = fixed_temporal_mapping
        self.temporal_mapping_single = temporal_mapping_single
        self.tmg_search_method = tmg_search_method
        self.temporal_mapping_multiprocessing = temporal_mapping_multiprocessing
        self.drc_enabled = drc_enabled
        self.prune_PE_RF = True
        self.mem_hierarchy_iterative_search = False
        self.unrolling_size_list = unrolling_size_list
        self.unrolling_scheme_list = unrolling_scheme_list
        self.unrolling_scheme_list_text = unrolling_scheme_list_text
        self.PE_RF_size_threshold = PE_RF_size_threshold
        self.PE_RF_depth = PE_RF_depth
        self.CHIP_depth = CHIP_depth
        self.utilization_optimizer_pruning = False
        self.max_area = max_area
        self.utilization_rate_area = utilization_rate_area
        self.memory_hierarchy_ratio = memory_hierarchy_ratio
        self.mem_pool = mem_pool
        self.L1_size = L1_size
        self.L2_size = L2_size
        self.memory_scheme_hint = memory_scheme_hint
        self.banking = banking
        self.spatial_utilization_threshold = spatial_utilization_threshold
        self.spatial_unrolling_mode = spatial_unrolling_mode
        self.stationary_optimization_enable = stationary_optimization_enable
        self.su_parallel_processing = su_parallel_processing
        self.arch_search_result_saving = arch_search_result_saving
        self.su_search_result_saving = su_search_result_saving
        self.tm_search_result_saving = tm_search_result_saving
        self.result_print_mode = result_print_mode
        self.im2col_enable = im2col_enable


def get_input_settings(setting_path, mapping_path, memory_pool_path, architecure_path):
    settings_file = open(setting_path)
    memory_pool_file = open(memory_pool_path)
    architecture_file = open(architecure_path)
    mapping_file = open(mapping_path)
    fl = yaml.full_load(settings_file)

    if fl['result_print_mode'] not in ['concise', 'complete']:
        raise ValueError('result_print_mode is not correctly set. Please check the setting file.')

    tm_fixed_flag = fl['fixed_temporal_mapping']
    sm_fixed_flag = fl['fixed_spatial_unrolling']
    arch_fixed_flag = fl['fixed_architecture']
    fl = yaml.full_load(memory_pool_file)
    memory_pool = []
    for m in fl:
        mt = fl[m]['mem_type']
        mt_tmp = 0
        if mt == 'dual_port_double_buffered': mt_tmp = 3
        if mt == 'dual_port_single_buffered': mt_tmp = 2
        if mt == 'single_port_double_buffered': mt_tmp = 1
        if mt_tmp == 0:
            raise ValueError("In memory pool, some memory's memory type is not correctly defined.")
        mbw = []
        for mb in fl[m]['mem_bw']:
            mbw.append([mb, mb])
        m_tmp = {
            'name': m,
            'size_bit': fl[m]['size_bit'],
            'mem_bw': mbw,
            'area': fl[m]['area'],
            'utilization_rate': fl[m]['utilization_rate'],
            'mem_type': mt_tmp,
            'cost': [list(a) for a in zip(fl[m]['cost']['read_word'], fl[m]['cost']['write_word'])],
            'unroll': 1,
            'mem_fifo': False
        }
        memory_pool.append(m_tmp)
    fl = yaml.full_load(architecture_file)
    L1_size = fl['L1_size']
    L2_size = fl['L2_size']
    banking = fl['banking']
    area_max_arch = fl['area_max']
    area_utilization_arch = fl['area_utilization']
    mem_ratio = fl['mem_ratio']
    PE_depth = fl['PE_memory_depth']
    CHIP_depth = fl['CHIP_memory_depth']
    PE_RF_size_threshold = fl['PE_threshold']
    mac_array_info = {}
    mac_array_stall = {}
    mac_array_info['array_size'] = [fl['PE_array']['Col'], fl['PE_array']['Row']]
    memory_scheme_hint = MemorySchemeNode([])
    mh_name = {'W': [], 'I': [], 'O': []}

    if arch_fixed_flag:
        for m in fl['memory_hierarchy']:
            m_tmp = [x for x in memory_pool if x['name'] == fl['memory_hierarchy'][m]['memory_instance']]
            if not m_tmp:
                raise Exception("Memory instance " + str(m) + " in hierarchy is not found in memory pool")
            m_tmp = m_tmp[0]
            m_tmp = MemoryNode(m_tmp, (), 0, 1)
            m_tmp.memory_level['unroll'] = fl['memory_hierarchy'][m]['bank_instances']
            m_tmp.memory_level['nbanks'] = fl['memory_hierarchy'][m]['nbanks']
            m_tmp.operand = tuple(fl['memory_hierarchy'][m]['operand_stored'])
            memory_scheme_hint.memory_scheme.add(m_tmp)
            for operand in ['W', 'I', 'O']:
                if operand in m_tmp.operand:
                    mh_name[operand].append(tuple([m, m_tmp.memory_level['size_bit']]))
        for operand in ['W', 'I', 'O']:
            mh_name[operand].sort(key=lambda tup: tup[1])
            mh_name[operand] = [x[0] for x in mh_name[operand]]
    else:
        if fl['memory_hint']:
            for m in fl['memory_hint']:
                m_tmp = [x for x in memory_pool if x['name'] == fl['memory_hint'][m]['memory_instance']]
                if not m_tmp:
                    raise Exception("Memory instance " + str(m) + " in hint is not found in memory pool")
                m_tmp = m_tmp[0]
                memory_pool.remove(m_tmp)
                m_tmp = MemoryNode(m_tmp, (), 0, 1)
                m_tmp.memory_level['unroll'] = fl['memory_hint'][m]['bank_instances']
                m_tmp.memory_level['nbanks'] = 1
                m_tmp.operand = tuple(fl['memory_hint'][m]['operand_stored'])

                memory_scheme_hint.memory_scheme.add(m_tmp)

    precision = {'W': fl['precision']['W'], 'I': fl['precision']['I'], 'O': fl['precision']['O_partial'],
                 'O_final': fl['precision']['O_final']}
    mac_array_info['single_mac_energy'] = fl['single_mac_energy']
    mac_array_info['idle_mac_energy'] = fl['idle_mac_energy']
    p_aux = [precision['W'], precision['I']]
    mac_array_info['precision'] = max(p_aux)
    mac_array_info['headroom'] = precision['O'] - precision['O_final']

    uwe = []
    for i in range(0, 10):
        uwe.append(0)
    mac_array_info['unit_wire_energy'] = uwe
    mac_array_stall['systolic'] = fl['mac_array_stall']['systolic']
    fl = yaml.full_load(mapping_file)
    tm_fixed = {'W': [], 'I': [], 'O': []}
    sm_fixed = {'W': [], 'I': [], 'O': []}
    flooring_fixed = {'W': [], 'I': [], 'O': []}
    unrolling_scheme_list = []
    unrolling_size_list = []
    i2a = {'B': 7, 'K': 6, 'C': 5, 'OY': 4, 'OX': 3, 'FY': 2, 'FX': 1}
    unrolling_scheme_list_text = []
    if tm_fixed_flag:
        for op in fl['temporal_mapping_fixed']:
            if op == 'weight': operand = 'W'
            if op == 'input': operand = 'I'
            if op == 'output': operand = 'O'
            tm_fixed[operand] = [[] for x in fl['temporal_mapping_fixed'][op]]
            for ii, lev in enumerate(fl['temporal_mapping_fixed'][op]):
                index_lev = mh_name[operand].index(lev)
                for pf in fl['temporal_mapping_fixed'][op][lev]:
                    tm_fixed[operand][index_lev].append(tuple([i2a[pf[0]], pf[1]]))
    if sm_fixed_flag:
        for op in fl['spatial_mapping_fixed']:
            if op == 'weight': operand = 'W'
            if op == 'input': operand = 'I'
            if op == 'output': operand = 'O'
            sm_fixed[operand] = [[] for x in fl['spatial_mapping_fixed'][op]]
            flooring_fixed[operand] = [[] for x in fl['spatial_mapping_fixed'][op]]
            for lev in fl['spatial_mapping_fixed'][op]:
                ii_lev = 0
                if lev == 'MAC' : ii_lev = 0
                else : ii_lev = lev + 1
                flooring_fixed[operand][ii_lev] = [[] for d in fl['spatial_mapping_fixed'][op][lev]]
                for dim in fl['spatial_mapping_fixed'][op][lev]:
                    ii_dim = 0
                    if dim == 'Col': ii_dim = 0
                    if dim == 'Row': ii_dim = 1
                    for pf in fl['spatial_mapping_fixed'][op][lev][dim]:
                        sm_fixed[operand][ii_lev].append(tuple([i2a[pf[0]], pf[1]]))
                        flooring_fixed[operand][ii_lev][ii_dim].append(i2a[pf[0]])
    else:
        unrolling_scheme_list_text = fl['spatial_mapping_list']
        for us in fl['spatial_mapping_list']:
            unrolling_scheme_list.append([])
            unrolling_scheme_list[-1] = [[] for x in us]
            unrolling_size_list.append([])
            unrolling_size_list[-1] = [[] for x in us]
            for dim in us:
                ii_dim = 0
                dimx = next(iter(dim))
                if dimx == 'Col': ii_dim = 0
                if dimx == 'Row': ii_dim = 1
                for pf in dim[dimx]:
                    pf_type = list(pf.split('_'))[0]
                    unrolling_scheme_list[-1][ii_dim].append(i2a[pf_type])
                    try:
                        pf_size = list(pf.split('_'))[1]
                        unrolling_size_list[-1][ii_dim].append(int(pf_size))
                    except:
                        pf_size = None
                        unrolling_size_list[-1][ii_dim].append(pf_size)

    settings_file = open(setting_path)
    fl = yaml.full_load(settings_file)
    if fl['temporal_mapping_search_method'] == 'exhaustive':
        tmg_search_method = 1
        stationary_optimization_enable = False
        data_reuse_threshold = 0
    elif fl['temporal_mapping_search_method'] == 'iterative':
        tmg_search_method = 0
        stationary_optimization_enable = True
        data_reuse_threshold = 1
    elif fl['temporal_mapping_search_method'] == 'heuristic_v1':
        tmg_search_method = 1
        stationary_optimization_enable = True
        data_reuse_threshold = 0
    elif fl['temporal_mapping_search_method'] == 'heuristic_v2':
        tmg_search_method = 1
        stationary_optimization_enable = True
        data_reuse_threshold = 1
    elif fl['temporal_mapping_search_method'] == 'c++':
        tmg_search_method = 2
        stationary_optimization_enable = None
        data_reuse_threshold = None
    else:
        raise ValueError('temporal_mapping_search_method is not correctly set. Please check the setting file.')

    sumode = ['exhaustive', 'heuristic_v1', 'heuristic_v2', 'hint_driven', 'hint_driven_with_greedy_mapping']
    if not fl['fixed_spatial_unrolling']:
        sumx = sumode.index(fl['spatial_unrolling_search_method'])
    else:
        sumx = 0
    input_settings = InputSettings(fl['result_path'], fl['result_filename'], fl['layer_filename'],
                                   fl['layer_indices'], fl['layer_multiprocessing'], precision,
                                   mac_array_info, mac_array_stall, fl['fixed_architecture'],
                                   fl['architecture_search_multiprocessing'], memory_scheme_hint,
                                   fl['fixed_spatial_unrolling'], sm_fixed, flooring_fixed,
                                   fl['fixed_temporal_mapping'], tm_fixed, tmg_search_method,
                                   fl['temporal_mapping_multiprocessing'],
                                   data_reuse_threshold, PE_RF_size_threshold, PE_depth,
                                   CHIP_depth, area_max_arch, area_utilization_arch,
                                   mem_ratio, memory_pool, banking, L1_size, L2_size, unrolling_size_list,
                                   unrolling_scheme_list, unrolling_scheme_list_text, memory_scheme_hint,
                                   fl['spatial_utilization_threshold'], sumx, stationary_optimization_enable,
                                   fl['spatial_unrolling_multiprocessing'], fl['save_all_architecture_result'],
                                   fl['save_all_spatial_unrolling_result'], fl['save_all_temporal_mapping_result'],
                                   fl['result_print_mode'], fl['im2col_enable'])

    return input_settings

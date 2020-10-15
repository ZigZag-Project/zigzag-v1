import os
import sys
import xml.etree.cElementTree as ET
import pandas as pd
import ast
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QTableView
from copy import deepcopy
from pathlib import Path
from datetime import datetime
import classes as cls
import time

Qt = QtCore.Qt


class PandasModel(QtCore.QAbstractTableModel):
    def __init__(self, data, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._data = data

    def rowCount(self, parent=None):
        return len(self._data.values)

    def columnCount(self, parent=None):
        return self._data.columns.size

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return QtCore.QVariant(str(
                    self._data.values[index.row()][index.column()]))
        return QtCore.QVariant()

    def headerData(self, rowcol, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self._data.columns[rowcol]
        if orientation == QtCore.Qt.Vertical and role == QtCore.Qt.DisplayRole:
            return self._data.index[rowcol]
        return None


def t_loop_name_transfer(mapping):
    loop_name = {1: 'FX', 2: 'FY', 3: 'OX', 4: 'OY', 5: 'C', 6: 'K', 7: 'B'}
    post_mapping = []
    for outer_idx, outer_li in enumerate(mapping):
        post_mapping.append([])
        if outer_li:
            for inner_idx, inner_li in enumerate(outer_li):
                post_mapping[-1].append((loop_name[inner_li[0]], inner_li[1]))
    return post_mapping


def s_loop_name_transfer(mapping):
    loop_name = {1: 'FX', 2: 'FY', 3: 'OX', 4: 'OY', 5: 'C', 6: 'K', 7: 'B'}
    post_mapping = []
    for outer_idx, outer_li in enumerate(mapping):
        post_mapping.append([])
        if outer_li:
            for inner_idx, inner_li in enumerate(outer_li):
                post_mapping[-1].append([])
                for va in inner_li:
                    post_mapping[-1][-1].append((loop_name[va[0]], va[1]))
    return post_mapping


def flooring_name_transfer(flooring):
    loop_name = {1: 'FX', 2: 'FY', 3: 'OX', 4: 'OY', 5: 'C', 6: 'K', 7: 'B'}
    for operand in ['W', 'I', 'O']:
        for outer_idx, outer_li in enumerate(flooring[operand]):
            if outer_li is []:
                continue
            else:
                for inner_idx, inner_li in enumerate(outer_li):
                    for i in range(len(inner_li)):
                        flooring[operand][outer_idx][inner_idx][i] = loop_name[
                            flooring[operand][outer_idx][inner_idx][i]]
                    flooring[operand][outer_idx][inner_idx] = tuple(flooring[operand][outer_idx][inner_idx])
    return flooring


def digit_truncate(data_in, num):
    if type(data_in) is dict:
        if 'O_partial' in list(data_in.keys()):
            try:
                for operand in ['W', 'I']:
                    for outer_idx, outer_va in enumerate(data_in[operand]):
                        for inner_idx, inner_va in enumerate(outer_va):
                            data_in[operand][outer_idx][inner_idx] = int(inner_va)
                for operand in ['O', 'O_partial', 'O_final']:
                    for outer_idx, outer_va in enumerate(data_in[operand]):
                        for inner_idx, inner_va in enumerate(outer_va):
                            data_in[operand][outer_idx][inner_idx] = (int(inner_va[0]), int(inner_va[1]))
                return data_in
            except:
                for operand in ['W', 'I', 'O', 'O_partial', 'O_final']:
                    for idx, va in enumerate(data_in[operand]):
                        data_in[operand][idx] = round(va)
                return data_in
        else:
            try:
                if num == 0:
                    for operand in ['W', 'I', 'O']:
                        for outer_idx, outer_va in enumerate(data_in[operand]):
                            for inner_idx, inner_va in enumerate(outer_va):
                                data_in[operand][outer_idx][inner_idx] = round(inner_va)
                    return data_in
                else:
                    for operand in ['W', 'I', 'O']:
                        for outer_idx, outer_va in enumerate(data_in[operand]):
                            for inner_idx, inner_va in enumerate(outer_va):
                                data_in[operand][outer_idx][inner_idx] = round(inner_va, num)
                    return data_in
            except:
                for operand in ['W', 'I', 'O']:
                    for idx, va in enumerate(data_in[operand]):
                        data_in[operand][idx] = round(va, num)
                return data_in
    else:
        for idx, va in enumerate(data_in):
            data_in[idx] = round(data_in[idx], num)
        return data_in


def energy_clean(energy_breakdown):
    try:
        energy_breakdown_clean = {'W': [], 'I': [], 'O': []}
        for operand in ['W', 'I', 'O']:
            for li in energy_breakdown[operand]:
                energy_breakdown_clean[operand].append(round(li[1], 1))
        return energy_breakdown_clean
    except:
        return energy_breakdown


def mem_bw_req_meet_check(req_bw, act_bw):
    meet_list = {'W': [], 'I': [], 'O': []}
    for operand in ['W', 'I', 'O']:
        for lv in range(len(req_bw[operand])):
            if (req_bw[operand][lv][0] <= act_bw[operand][lv][0]) and \
                    (req_bw[operand][lv][1] <= act_bw[operand][lv][1]):
                meet_list[operand].append([True, True])
            elif (req_bw[operand][lv][0] > act_bw[operand][lv][0]) and \
                    (req_bw[operand][lv][1] <= act_bw[operand][lv][1]):
                meet_list[operand].append([False, True])
            elif (req_bw[operand][lv][0] <= act_bw[operand][lv][0]) and \
                    (req_bw[operand][lv][1] > act_bw[operand][lv][1]):
                meet_list[operand].append([True, False])
            else:
                meet_list[operand].append([False, False])
    return meet_list


def spatial_loop_same_term_merge(unrolling, flooring):
    spatial_list = {'W': [], 'I': [], 'O': []}
    for operand in ['W', 'I', 'O']:
        for level, level_list in enumerate(flooring[operand]):
            spatial_list[operand].append([])
            if not level_list:
                continue
            else:
                for XY_idx, XY_list in enumerate(level_list):
                    spatial_list[operand][-1].append([])
                    for va in XY_list:
                        spatial_list[operand][-1][-1].append(list(unrolling[operand][level].pop(0)))

    spatial_list_clean = deepcopy(spatial_list)
    for operand in ['W', 'I', 'O']:
        for level, level_list in enumerate(spatial_list[operand]):
            if not level_list:
                continue
            else:
                for XY_idx, XY_list in enumerate(level_list):
                    if len(XY_list) in [1, 0]:
                        continue
                    else:
                        va_clean_idx = 0
                        for va_idx in range(1, len(XY_list)):
                            if XY_list[va_idx - 1][0] == XY_list[va_idx][0]:
                                spatial_list_clean[operand][level][XY_idx][va_clean_idx][1] *= XY_list[va_idx][1]
                                spatial_list_clean[operand][level][XY_idx].remove(XY_list[va_idx])
                                va_clean_idx -= 1
                            va_clean_idx += 1
    return spatial_list_clean


def iterative_data_format_clean(original_dict):
    new_dict = {'W': [], 'I': [], 'O': []}
    for operand in ['W', 'I', 'O']:
        for li in original_dict[operand]:
            new_dict[operand].append(li[0])
    return new_dict


def mem_unroll_format(unit_count):
    return {'W': unit_count['W'][1:], 'I': unit_count['I'][1:], 'O': unit_count['O'][1:]}


def mem_share_reformat(mem_share, mem_name):
    new_mem_share = {}
    for i, shared_list in mem_share.items():
        new_mem_share[i + 1] = []
        for li in shared_list:
            new_mem_share[i + 1].append((li[0], mem_name[li[0]][li[1]]))
    return new_mem_share


def add_mem_type_name(mem_type_list):
    name_transfer = {1: 'sp_db', 2: 'dp_sb', 3: 'dp_db'}
    new_mem_type_list = {'W': [], 'I': [], 'O': []}
    for operand in ['W', 'I', 'O']:
        for li in mem_type_list[operand]:
            new_mem_type_list[operand].append(name_transfer[li])
    return new_mem_type_list


def elem2bit(elem_list, precision):
    bit_list = {'W': [], 'I': [], 'O': [], 'O_partial': [], 'O_final': []}
    for operand in ['W', 'I', 'O_partial', 'O_final']:
        for va in elem_list[operand]:
            bit_list[operand].append(va * precision[operand])
    for idx, va in enumerate(bit_list['O_partial']):
        if va != 0:
            bit_list['O'].append(va)
        else:
            bit_list['O'].append(bit_list['O_final'][idx])
    return bit_list


def create_printing_block(row, col):
    return [[' '] * col for _ in range(row)]


def modify_printing_block(old_block, start_row, start_col, new_str):
    new_block = deepcopy(old_block)
    new_block[start_row][start_col:start_col + len(new_str)] = new_str
    return new_block


def print_printing_block(file_path_name, printing_block, mode):
    orig_stdout = sys.stdout
    f = open(file_path_name, mode)
    sys.stdout = f
    print()
    for i in range(len(printing_block)):
        print(''.join(printing_block[i]))
    sys.stdout = orig_stdout
    f.close()


def print_good_tm_format(tm, mem_name, file_path_name):
    lp_name = {1: 'FX', 2: 'FY', 3: 'OX', 4: 'OY', 5: 'C', 6: 'K', 7: 'B'}
    tm_list = [tp for li in tm['W'] for tp in li]

    # get required interval between 'W', 'I', 'O', based on actual mem name length
    max_mem_name_len = 0
    for operand in ['W', 'I', 'O']:
        for lv in range(len(mem_name[operand])):
            if len(mem_name[operand][lv]) > max_mem_name_len:
                max_mem_name_len = len(mem_name[operand][lv])
    interval = max_mem_name_len + 10

    tot_row = 2 * (len(tm_list) + 1) + 8
    tot_col = int(2 * (len(tm_list) + 1) + 3.75 * interval)
    tot_col_cut = 2 * (len(tm_list) + 1) + interval
    tm_block = create_printing_block(tot_row, tot_col)
    dash = '*' * int((tot_col - len(' Temporal Mapping Visualization ')) / 2)
    tm_block = modify_printing_block(tm_block, 1, 0, dash + ' Temporal Mapping Visualization ' + dash)
    tm_block = modify_printing_block(tm_block, 2, 1, 'W: ' + str(t_loop_name_transfer(tm['W'])))
    tm_block = modify_printing_block(tm_block, 3, 1, 'I: ' + str(t_loop_name_transfer(tm['I'])))
    tm_block = modify_printing_block(tm_block, 4, 1, 'O: ' + str(t_loop_name_transfer(tm['O'])))
    tm_block = modify_printing_block(tm_block, 6, 0, '-' * tot_col)
    tm_block = modify_printing_block(tm_block, 7, 1, 'Temporal Loops')
    tm_block = modify_printing_block(tm_block, 8, 0, '-' * tot_col)
    finish_row = 2 * len(tm_list) + 7
    for i, li in enumerate(tm_list):
        tm_block = modify_printing_block(tm_block, finish_row - 2 * i, len(tm_list) - i,
                                         'for ' + str(lp_name[li[0]]) + ' in ' + '[0:' + str(li[1]) + ')')
        tm_block = modify_printing_block(tm_block, 2 * (i + 1) + 1 + 7, 0, '-' * tot_col)

    # print mem name to each level
    for idx, operand in enumerate(['W', 'I', 'O']):
        column_position = tot_col_cut + idx * interval
        tm_block = modify_printing_block(tm_block, 7, column_position, operand)
        i = 0
        for level, lv_li in enumerate(tm[operand]):
            for _ in lv_li:
                tm_block = modify_printing_block(tm_block, finish_row - 2 * i, column_position,
                                                 str(mem_name[operand][level]))
                i += 1
    tm_block = modify_printing_block(tm_block, finish_row + 2, 1,
                                     "(Notes: Temporal Mapping starts from the innermost memory level. MAC level is out of Temporal Mapping's scope.)")
    print_printing_block(file_path_name, tm_block, 'a+')


def print_good_su_format(su, mem_name, file_path_name):
    lp_name = {1: 'FX', 2: 'FY', 3: 'OX', 4: 'OY', 5: 'C', 6: 'K', 7: 'B'}
    su_list = [sp for lv_li in su['W'] for xy_li in lv_li for sp in xy_li]
    mem_name = {'W': ['MAC'] + mem_name['W'], 'I': ['MAC'] + mem_name['I'], 'O': ['MAC'] + mem_name['O']}
    # get required interval between 'W', 'I', 'O', based on actual mem name length
    max_mem_name_len = 0
    for operand in ['W', 'I', 'O']:
        for lv in range(len(mem_name[operand])):
            if len(mem_name[operand][lv]) > max_mem_name_len and su[operand][lv] != []:
                max_mem_name_len = len(mem_name[operand][lv])
    interval = max_mem_name_len + 13

    tot_row = 2 * (len(su_list) + 1) + 13
    tot_col = int(2 * (len(su_list) + 1) + 3.75 * interval)
    tot_col_cut = 2 * (len(su_list) + 1) + interval
    su_block = create_printing_block(tot_row, tot_col)
    dash = '*' * int((tot_col - len(' Levels In The System')) / 2)
    su_block = modify_printing_block(su_block, 0, 0, dash + ' Levels In The System ' + dash)
    su_block = modify_printing_block(su_block, 1, 1, 'W: ' + str(mem_name['W']))
    su_block = modify_printing_block(su_block, 2, 1, 'I: ' + str(mem_name['I']))
    su_block = modify_printing_block(su_block, 3, 1, 'O: ' + str(mem_name['O']))
    dash = '*' * int((tot_col - len(' Spatial Unrolling Visualization ')) / 2)
    su_block = modify_printing_block(su_block, 6, 0, dash + ' Spatial Unrolling Visualization ' + dash)
    su_block = modify_printing_block(su_block, 7, 1, 'W: ' + str(s_loop_name_transfer(su['W'])))
    su_block = modify_printing_block(su_block, 8, 1, 'I: ' + str(s_loop_name_transfer(su['I'])))
    su_block = modify_printing_block(su_block, 9, 1, 'O: ' + str(s_loop_name_transfer(su['O'])))
    su_block = modify_printing_block(su_block, 11, 0, '-' * tot_col)
    su_block = modify_printing_block(su_block, 12, 1, "Unrolled Loops")
    su_block = modify_printing_block(su_block, 13, 0, '-' * tot_col)
    finish_row = 2 * len(su_list) + 12
    for i, li in enumerate(su_list):
        su_block = modify_printing_block(su_block, finish_row - 2 * i, 1,
                                         'unroll ' + str(lp_name[li[0]]) + ' in ' + '[0:' + str(li[1]) + ')')
        su_block = modify_printing_block(su_block, 2 * (i + 1) + 1 + 12, 0, '-' * tot_col)

    su_block = modify_printing_block(su_block, finish_row + 2, 1,
                                     "(Notes: Unrolled loops' order doesn't matters; D1 and D2 indicate 2 PE array dimensions.)")
    # print mem name to each level
    XY_name = {0: 'D1', 1: 'D2'}
    for idx, operand in enumerate(['W', 'I', 'O']):
        column_position = tot_col_cut + idx * interval
        su_block = modify_printing_block(su_block, 12, column_position, operand)
        i = 0
        for level, lv_li in enumerate(su[operand]):
            for xy, xy_li in enumerate(lv_li):
                for _ in enumerate(xy_li):
                    su_block = modify_printing_block(su_block, finish_row - 2 * i, column_position,
                                                     str(mem_name[operand][level]) + ' (' + XY_name[xy] + ')')
                    i += 1
    print_printing_block(file_path_name, su_block, 'w+')


def print_xml(results_filename, layer_specification, mem_scheme, cost_model_output, common_settings,
              hw_pool_sizes, elapsed_time, result_print_mode):
    dir_path = ''
    dir_path_list = results_filename.split('/')
    for i in range(0, len(dir_path_list) - 1):
        dir_path += dir_path_list[i] + '/'
    ''' Create result folder if it does not exist. '''
    Path(dir_path).mkdir(parents=True, exist_ok=True)

    ''' 
    Newly generated result will be appended after existed ones. 
    Create new result file if there is no existed one.
    '''
    if not os.path.isfile(results_filename + '.xml'):
        root = ET.Element('root')
        tree = ET.ElementTree(root)
        tree.write(results_filename + '.xml')

    tree = ET.parse(results_filename + '.xml')
    root = tree.getroot()
    sim = ET.SubElement(root, 'simulation')
    result_generate_time = ET.SubElement(sim, 'result_generated_time')
    result_generate_time.tail = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if cost_model_output not in [None, [], {}]:
        if result_print_mode == 'complete':
            layer = ET.SubElement(sim, 'layer')
            # layer_index = ET.SubElement(layer, 'layer_index')
            # layer_index.tail = str(common_settings.layer_index)
            layer_spec = ET.SubElement(layer, 'layer_spec')
            layer_spec.tail = str(layer_specification.size_list_output_print)
            total_MAC_op = ET.SubElement(layer, 'total_MAC_operation')
            total_MAC_op.tail = str(layer_specification.total_MAC_op)
            total_data_size = ET.SubElement(layer, 'total_data_size_element')
            total_data_size.tail = str(layer_specification.total_data_size)
            total_data_reuse = ET.SubElement(layer, 'total_data_reuse')
            total_data_reuse.tail = str(layer_specification.total_data_reuse)

            search_engine = ET.SubElement(sim, 'search_engine')

            memory_hier_search_engine = ET.SubElement(search_engine, 'mem_hierarchy_search')
            memory_hier_search_mode = ET.SubElement(memory_hier_search_engine, 'mode')
            memory_hier_search_mode.tail = str(common_settings.search_mode['Mem'])
            if not common_settings.search_mode['Mem'] == 'fixed':
                total_area = ET.SubElement(memory_hier_search_engine, 'area_constraint')
                total_area.tail = str(common_settings.area_constraint['max_area'])
                area_utilization_threshold = ET.SubElement(memory_hier_search_engine, 'area_utilization_threshold')
                area_utilization_threshold.tail = '\u2265 ' + str(common_settings.area_constraint['area_th'])
                # mem_pool = ET.SubElement(memory_hier_search_engine, 'mem_pool')
                # mem_pool.tail = str(common_settings.mem_pool)
                memory_hierarchy_ratio = ET.SubElement(memory_hier_search_engine, 'consecutive_memory_level_size_ratio')
                memory_hierarchy_ratio.tail = '\u2265 ' + str(common_settings.memory_hierarchy_ratio)
                max_inner_PE_mem_size = ET.SubElement(memory_hier_search_engine, 'max_inner_PE_mem_size_bit')
                max_inner_PE_mem_size.tail = str(common_settings.max_inner_PE_mem_size)
                max_inner_PE_mem_level = ET.SubElement(memory_hier_search_engine, 'max_inner_PE_mem_level')
                max_inner_PE_mem_level.tail = str(common_settings.max_inner_PE_mem_level)
                max_outer_PE_mem_level = ET.SubElement(memory_hier_search_engine, 'max_outer_PE_mem_level')
                max_outer_PE_mem_level.tail = str(common_settings.max_outer_PE_mem_level)
            valid_mem_scheme_count = ET.SubElement(memory_hier_search_engine, 'mem_scheme_index')
            valid_mem_scheme_count.tail = str(common_settings.mem_scheme_count)

            spatial_mapping_search_engine = ET.SubElement(search_engine, 'spatial_mapping_search')
            spatial_search_mode = ET.SubElement(spatial_mapping_search_engine, 'mode')
            spatial_search_mode.tail = str(common_settings.search_mode['Spatial'])
            if not common_settings.search_mode['Spatial'] == 'fixed':
                if common_settings.spatial_unrolling_mode != 3:
                    spatial_utilization_threshold = ET.SubElement(spatial_mapping_search_engine,
                                                                  'spatial_utilization_threshold')
                    spatial_utilization_threshold.tail = str(common_settings.SU_threshold)
                else:
                    spatial_mapping_hint_list = ET.SubElement(spatial_mapping_search_engine,
                                                              'spatial_mapping_hint_list')
                    spatial_mapping_hint_list.tail = str(common_settings.spatial_mapping_hint_list)
            valid_unrolling_scheme_count = ET.SubElement(spatial_mapping_search_engine, 'unrolling_scheme_index')
            valid_unrolling_scheme_count.tail = str(common_settings.spatial_count)

            temporal_mapping_search_engine = ET.SubElement(search_engine, 'temporal_mapping_search')
            temporal_search_mode = ET.SubElement(temporal_mapping_search_engine, 'mode')
            temporal_search_mode.tail = str(common_settings.search_mode['Temporal'])
            if not common_settings.search_mode['Temporal'] == 'fixed':
                memory_utilization_threshold = ET.SubElement(temporal_mapping_search_engine, 'memory_utilization_hint')
                memory_utilization_threshold.tail = str(common_settings.mem_utilization_rate)
            valid_temporal_mapping_count = ET.SubElement(temporal_mapping_search_engine, 'valid_temporal_mapping_found')
            valid_temporal_mapping_count.tail = str(hw_pool_sizes)

            hw_spec = ET.SubElement(sim, 'hw_spec')

            PE_array = ET.SubElement(hw_spec, 'PE_array')
            precision = ET.SubElement(PE_array, 'precision_bit')
            precision.tail = str(common_settings.precision)
            array_size = ET.SubElement(PE_array, 'array_size')
            array_size.tail = str(common_settings.array_size)

            memory_hierarchy = ET.SubElement(hw_spec, 'memory_hierarchy')
            mem_name = ET.SubElement(memory_hierarchy, 'mem_name_in_the_hierarchy')
            mem_name_W = ET.SubElement(mem_name, 'W')
            mem_name_W.tail = str(mem_scheme.mem_name['W'])
            mem_name_I = ET.SubElement(mem_name, 'I')
            mem_name_I.tail = str(mem_scheme.mem_name['I'])
            mem_name_O = ET.SubElement(mem_name, 'O')
            mem_name_O.tail = str(mem_scheme.mem_name['O'])

            mem_size_bit = ET.SubElement(memory_hierarchy, 'mem_size_bit')
            mem_size_bit.tail = str(mem_scheme.mem_size)

            mem_access_cost = deepcopy(common_settings.mem_access_cost)
            if type(mem_scheme.mem_bw['W'][0][0]) in [list, tuple]:
                mem_scheme.mem_bw = iterative_data_format_clean(mem_scheme.mem_bw)
            if type(common_settings.mem_access_cost['W'][0][0]) in [list, tuple]:
                mem_access_cost = iterative_data_format_clean(mem_access_cost)
            if type(mem_scheme.mem_area['W'][0]) in [list, tuple]:
                mem_scheme.mem_area = iterative_data_format_clean(mem_scheme.mem_area)
            mem_bw = ET.SubElement(memory_hierarchy, 'mem_bw_bit_per_cycle_or_mem_wordlength')
            mem_bw.tail = str(mem_scheme.mem_bw)
            mem_cost_word = ET.SubElement(memory_hierarchy, 'mem_access_energy_per_word')
            mem_cost_word.tail = str(mem_access_cost)
            # pun_factor = ET.SubElement(memory_hierarchy, 'pun_factor')
            # pun_factor.tail = str(cost_model_output.utilization.pun_factor)
            mem_type = ET.SubElement(memory_hierarchy, 'mem_type')
            mem_type.tail = str(add_mem_type_name(mem_scheme.mem_type))
            mem_share = ET.SubElement(memory_hierarchy, 'mem_share')
            mem_share.tail = str(mem_share_reformat(mem_scheme.mem_share, mem_scheme.mem_name))
            mem_area = ET.SubElement(memory_hierarchy, 'mem_area_single_module')
            mem_area.tail = str(mem_scheme.mem_area)
            mem_unroll = ET.SubElement(memory_hierarchy, 'mem_unroll')
            mem_unroll.tail = str(mem_unroll_format(cost_model_output.spatial_loop.unit_count))

            results = ET.SubElement(sim, 'results')
            basic_info = ET.SubElement(results, 'basic_info')

            spatial_mapping = ET.SubElement(basic_info, 'spatial_unrolling')
            spatial_mapping_W = ET.SubElement(spatial_mapping, 'W')
            spatial_mapping_I = ET.SubElement(spatial_mapping, 'I')
            spatial_mapping_O = ET.SubElement(spatial_mapping, 'O')
            try:
                spatial_mapping_W.tail = str(s_loop_name_transfer(cost_model_output.spatial_scheme['W']))
                spatial_mapping_I.tail = str(s_loop_name_transfer(cost_model_output.spatial_scheme['I']))
                spatial_mapping_O.tail = str(s_loop_name_transfer(cost_model_output.spatial_scheme['O']))
            except:
                cost_model_output.spatial_scheme = \
                    spatial_loop_same_term_merge(cost_model_output.spatial_scheme, cost_model_output.flooring)
                spatial_mapping_W.tail = str(s_loop_name_transfer(cost_model_output.spatial_scheme['W']))
                spatial_mapping_I.tail = str(s_loop_name_transfer(cost_model_output.spatial_scheme['I']))
                spatial_mapping_O.tail = str(s_loop_name_transfer(cost_model_output.spatial_scheme['O']))
            temporal_mapping = ET.SubElement(basic_info, 'temporal_mapping')
            temporal_mapping_W = ET.SubElement(temporal_mapping, 'W')
            temporal_mapping_W.tail = str(t_loop_name_transfer(cost_model_output.temporal_scheme['W']))
            temporal_mapping_I = ET.SubElement(temporal_mapping, 'I')
            temporal_mapping_I.tail = str(t_loop_name_transfer(cost_model_output.temporal_scheme['I']))
            temporal_mapping_O = ET.SubElement(temporal_mapping, 'O')
            temporal_mapping_O.tail = str(t_loop_name_transfer(cost_model_output.temporal_scheme['O']))

            data_reuse = ET.SubElement(basic_info, 'data_reuse')
            try:
                del (cost_model_output.loop.data_reuse['I_base'])
                del (cost_model_output.loop.data_reuse['I_zigzag'])
            except:
                pass
            data_reuse.tail = str(digit_truncate(cost_model_output.loop.data_reuse, 2))
            i_fifo = ET.SubElement(basic_info, 'I_pr_diagonally_broadcast_or_fifo_effect')
            i_fifo.tail = "'I': " + str(cost_model_output.loop.I_fifo)
            req_mem_size = ET.SubElement(basic_info, 'used_mem_size_bit')
            req_mem_size.tail = str(elem2bit(cost_model_output.loop.req_mem_size, common_settings.precision))
            mem_utilize = ET.SubElement(basic_info, 'actual_mem_utilization_individual')
            mem_utilize.tail = str(digit_truncate(cost_model_output.utilization.mem_utilize, 2))
            mem_utilize_shared = ET.SubElement(basic_info, 'actual_mem_utilization_shared')
            mem_utilize_shared.tail = str(digit_truncate(cost_model_output.utilization.mem_utilize_shared, 2))
            effective_mem_size = ET.SubElement(basic_info, 'effective_mem_size_bit')
            effective_mem_size.tail = str(elem2bit(digit_truncate(cost_model_output.loop.effective_mem_size, 0),
                                                   common_settings.precision))
            unit_count = ET.SubElement(basic_info, 'total_unit_count')
            unit_count.tail = str(cost_model_output.spatial_loop.unit_count)
            unit_unique = ET.SubElement(basic_info, 'unique_unit_count')
            unit_unique.tail = str(cost_model_output.spatial_loop.unit_unique)
            unit_duplicate = ET.SubElement(basic_info, 'duplicate_unit_count')
            unit_duplicate.tail = str(cost_model_output.spatial_loop.unit_duplicate)

            try:
                del (cost_model_output.loop.mem_access_elem['I_base'])
                del (cost_model_output.loop.mem_access_elem['I_zig_zag'])
            except:
                pass
            access_count = ET.SubElement(basic_info, 'mem_access_count_elem')
            cost_model_output.loop.mem_access_elem = digit_truncate(cost_model_output.loop.mem_access_elem, 0)
            access_count_W = ET.SubElement(access_count, 'W')
            access_count_W.tail = str(cost_model_output.loop.mem_access_elem['W'])
            access_count_I = ET.SubElement(access_count, 'I')
            access_count_I.tail = str(cost_model_output.loop.mem_access_elem['I'])
            access_count_O = ET.SubElement(access_count, 'O')
            access_count_O.tail = str(cost_model_output.loop.mem_access_elem['O'])
            access_count_O_partial = ET.SubElement(access_count, 'O_partial')
            access_count_O_partial.tail = str(cost_model_output.loop.mem_access_elem['O_partial'])
            access_count_O_final = ET.SubElement(access_count, 'O_final')
            access_count_O_final.tail = str(cost_model_output.loop.mem_access_elem['O_final'])
            # array_element_distance = ET.SubElement(basic_info, 'array_element_distance')
            # array_element_distance.tail = str(cost_model_output.loop.array_wire_distance)

            energy = ET.SubElement(results, 'energy')
            minimum_cost = ET.SubElement(energy, 'total_energy')
            minimum_cost.tail = str(round(cost_model_output.total_cost, 1))
            energy_breakdown = ET.SubElement(energy, 'mem_energy_breakdown')
            cost_model_output.operand_cost = energy_clean(cost_model_output.operand_cost)
            energy_breakdown_W = ET.SubElement(energy_breakdown, 'W')
            energy_breakdown_W.tail = str(cost_model_output.operand_cost['W'])
            energy_breakdown_I = ET.SubElement(energy_breakdown, 'I')
            energy_breakdown_I.tail = str(cost_model_output.operand_cost['I'])
            energy_breakdown_O = ET.SubElement(energy_breakdown, 'O')
            energy_breakdown_O.tail = str(cost_model_output.operand_cost['O'])

            mac_cost = ET.SubElement(energy, 'mac_energy')
            mac_cost.tail = 'active: ' + str(round(cost_model_output.mac_cost[0], 1)) + \
                            ', idle: ' + str(round(cost_model_output.mac_cost[1], 1))

            performance = ET.SubElement(results, 'performance')
            utilization = ET.SubElement(performance, 'mac_array_utilization')
            mac_utilize = ET.SubElement(utilization, 'utilization_with_data_loading')
            mac_utilize.tail = str(round(cost_model_output.utilization.mac_utilize, 4))
            mac_utilize_no_load = ET.SubElement(utilization, 'utilization_without_data_loading')
            mac_utilize_no_load.tail = str(round(cost_model_output.utilization.mac_utilize_no_load, 4))

            mac_utilize_spatial = ET.SubElement(utilization, 'utilization_spatial')
            mac_utilize_spatial.tail = str(round(cost_model_output.utilization.mac_utilize_spatial, 4))
            mac_utilize_temporal = ET.SubElement(utilization, 'utilization_temporal_with_data_loading')
            mac_utilize_temporal.tail = str(round(cost_model_output.utilization.mac_utilize_temporal, 4))
            mac_utilize_temporal_no_load = ET.SubElement(utilization, 'mac_utilize_temporal_without_data_loading')
            mac_utilize_temporal_no_load.tail = str(
                round(cost_model_output.utilization.mac_utilize_temporal_no_load, 4))

            latency = ET.SubElement(performance, 'latency')
            latency_tot = ET.SubElement(latency, 'latency_cycle_with_data_loading')
            latency_tot.tail = str(cost_model_output.utilization.latency_tot)
            latency_no_load = ET.SubElement(latency, 'latency_cycle_without_data_loading')
            latency_no_load.tail = str(cost_model_output.utilization.latency_no_load)
            total_cycles = ET.SubElement(latency, 'ideal_computing_cycle')
            total_cycles.tail = str(cost_model_output.temporal_loop.total_cycles)

            data_loading = ET.SubElement(latency, 'data_loading')
            cc_load_tot = ET.SubElement(data_loading, 'load_cycle_total')
            cc_load_tot.tail = str(cost_model_output.utilization.cc_load_tot)
            cc_load = ET.SubElement(data_loading, 'load_cycle_individual')
            cc_load.tail = str(cost_model_output.utilization.cc_load)
            cc_load_comb = ET.SubElement(data_loading, 'load_cycle_combined')
            cc_load_comb.tail = str(cost_model_output.utilization.cc_load_comb)

            mem_stalling = ET.SubElement(latency, 'mem_stalling')
            cc_mem_stall_tot = ET.SubElement(mem_stalling, 'mem_stall_cycle_total')
            cc_mem_stall_tot.tail = str(cost_model_output.utilization.cc_mem_stall_tot)
            stall_cc = ET.SubElement(mem_stalling, 'mem_stall_cycle_individual')
            stall_cc.tail = str(cost_model_output.utilization.stall_cc)
            stall_cc_mem_share = ET.SubElement(mem_stalling, 'mem_stall_cycle_shared')
            stall_cc_mem_share.tail = str(cost_model_output.utilization.stall_cc_mem_share)

            req_mem_bw_bit = ET.SubElement(mem_stalling, 'req_mem_bw_bit_per_cycle_individual')
            req_mem_bw_bit.tail = str(digit_truncate(cost_model_output.utilization.req_mem_bw_bit, 1))
            req_sh_mem_bw_bit = ET.SubElement(mem_stalling, 'req_mem_bw_bit_per_cycle_shared')
            req_sh_mem_bw_bit.tail = str(digit_truncate(cost_model_output.utilization.req_sh_mem_bw_bit, 1))
            mem_bw_req_meet_list = mem_bw_req_meet_check(cost_model_output.utilization.req_sh_mem_bw_bit,
                                                         mem_scheme.mem_bw)
            mem_bw_req_meet = ET.SubElement(mem_stalling, 'mem_bw_requirement_meet')
            mem_bw_req_meet.tail = str(mem_bw_req_meet_list)

            area = ET.SubElement(results, 'area')
            area.tail = str(cost_model_output.area)
        else:
            layer = ET.SubElement(sim, 'layer')
            # layer_index = ET.SubElement(layer, 'layer_index')
            # layer_index.tail = str(common_settings.layer_index)
            layer_spec = ET.SubElement(layer, 'layer_spec')
            layer_spec.tail = str(layer_specification.size_list_output_print)
            total_MAC_op = ET.SubElement(layer, 'total_MAC_operation')
            total_MAC_op.tail = str(layer_specification.total_MAC_op)
            total_data_size = ET.SubElement(layer, 'total_data_size_element')
            total_data_size.tail = str(layer_specification.total_data_size)
            # total_data_reuse = ET.SubElement(layer, 'total_data_reuse')
            # total_data_reuse.tail = str(layer_specification.total_data_reuse)

            hw_spec = ET.SubElement(sim, 'hw_spec')
            PE_array = ET.SubElement(hw_spec, 'PE_array')
            precision = ET.SubElement(PE_array, 'precision_bit')
            precision.tail = str(common_settings.precision)
            array_size = ET.SubElement(PE_array, 'array_size')
            array_size.tail = str(common_settings.array_size)

            memory_hierarchy = ET.SubElement(hw_spec, 'memory_hierarchy')
            mem_name = ET.SubElement(memory_hierarchy, 'mem_name_in_the_hierarchy')
            mem_name_W = ET.SubElement(mem_name, 'W')
            mem_name_W.tail = str(mem_scheme.mem_name['W'])
            mem_name_I = ET.SubElement(mem_name, 'I')
            mem_name_I.tail = str(mem_scheme.mem_name['I'])
            mem_name_O = ET.SubElement(mem_name, 'O')
            mem_name_O.tail = str(mem_scheme.mem_name['O'])

            mem_size_bit = ET.SubElement(memory_hierarchy, 'mem_size_bit')
            mem_size_bit.tail = str(mem_scheme.mem_size)

            mem_access_cost = deepcopy(common_settings.mem_access_cost)
            if type(mem_scheme.mem_bw['W'][0][0]) in [list, tuple]:
                mem_scheme.mem_bw = iterative_data_format_clean(mem_scheme.mem_bw)
            if type(common_settings.mem_access_cost['W'][0][0]) in [list, tuple]:
                mem_access_cost = iterative_data_format_clean(mem_access_cost)
            if type(mem_scheme.mem_area['W'][0]) in [list, tuple]:
                mem_scheme.mem_area = iterative_data_format_clean(mem_scheme.mem_area)
            mem_bw = ET.SubElement(memory_hierarchy, 'mem_bw_bit_per_cycle_or_mem_wordlength')
            mem_bw.tail = str(mem_scheme.mem_bw)
            mem_cost_word = ET.SubElement(memory_hierarchy, 'mem_access_energy_per_word')
            mem_cost_word.tail = str(mem_access_cost)
            mem_type = ET.SubElement(memory_hierarchy, 'mem_type')
            mem_type.tail = str(add_mem_type_name(mem_scheme.mem_type))
            mem_share = ET.SubElement(memory_hierarchy, 'mem_share')
            mem_share.tail = str(mem_share_reformat(mem_scheme.mem_share, mem_scheme.mem_name))
            mem_area = ET.SubElement(memory_hierarchy, 'mem_area_single_module')
            mem_area.tail = str(mem_scheme.mem_area)
            mem_unroll = ET.SubElement(memory_hierarchy, 'mem_unroll')
            mem_unroll.tail = str(mem_unroll_format(cost_model_output.spatial_loop.unit_count))

            results = ET.SubElement(sim, 'results')
            basic_info = ET.SubElement(results, 'basic_info')

            spatial_mapping = ET.SubElement(basic_info, 'spatial_unrolling')
            spatial_mapping_W = ET.SubElement(spatial_mapping, 'W')
            spatial_mapping_I = ET.SubElement(spatial_mapping, 'I')
            spatial_mapping_O = ET.SubElement(spatial_mapping, 'O')
            try:
                spatial_mapping_W.tail = str(s_loop_name_transfer(cost_model_output.spatial_scheme['W']))
                spatial_mapping_I.tail = str(s_loop_name_transfer(cost_model_output.spatial_scheme['I']))
                spatial_mapping_O.tail = str(s_loop_name_transfer(cost_model_output.spatial_scheme['O']))
            except:
                cost_model_output.spatial_scheme = \
                    spatial_loop_same_term_merge(cost_model_output.spatial_scheme, cost_model_output.flooring)
                spatial_mapping_W.tail = str(s_loop_name_transfer(cost_model_output.spatial_scheme['W']))
                spatial_mapping_I.tail = str(s_loop_name_transfer(cost_model_output.spatial_scheme['I']))
                spatial_mapping_O.tail = str(s_loop_name_transfer(cost_model_output.spatial_scheme['O']))
            temporal_mapping = ET.SubElement(basic_info, 'temporal_mapping')
            temporal_mapping_W = ET.SubElement(temporal_mapping, 'W')
            temporal_mapping_W.tail = str(t_loop_name_transfer(cost_model_output.temporal_scheme['W']))
            temporal_mapping_I = ET.SubElement(temporal_mapping, 'I')
            temporal_mapping_I.tail = str(t_loop_name_transfer(cost_model_output.temporal_scheme['I']))
            temporal_mapping_O = ET.SubElement(temporal_mapping, 'O')
            temporal_mapping_O.tail = str(t_loop_name_transfer(cost_model_output.temporal_scheme['O']))

            energy = ET.SubElement(results, 'energy')
            minimum_cost = ET.SubElement(energy, 'total_energy')
            minimum_cost.tail = str(round(cost_model_output.total_cost, 1))
            energy_breakdown = ET.SubElement(energy, 'energy_breakdown')
            cost_model_output.operand_cost = energy_clean(cost_model_output.operand_cost)
            energy_breakdown_W = ET.SubElement(energy_breakdown, 'mem_energy_W')
            energy_breakdown_W.tail = str(cost_model_output.operand_cost['W'])
            energy_breakdown_I = ET.SubElement(energy_breakdown, 'mem_energy_I')
            energy_breakdown_I.tail = str(cost_model_output.operand_cost['I'])
            energy_breakdown_O = ET.SubElement(energy_breakdown, 'mem_energy_O')
            energy_breakdown_O.tail = str(cost_model_output.operand_cost['O'])

            mac_cost = ET.SubElement(energy, 'mac_energy')
            mac_cost.tail = 'active: ' + str(round(cost_model_output.mac_cost[0], 1)) + \
                            ', idle: ' + str(round(cost_model_output.mac_cost[1], 1))

            performance = ET.SubElement(results, 'performance')
            utilization = ET.SubElement(performance, 'mac_array_utilization')
            # mac_utilize = ET.SubElement(utilization, 'utilization_with_data_loading')
            # mac_utilize.tail = str(round(cost_model_output.utilization.mac_utilize, 4))
            # mac_utilize_no_load = ET.SubElement(utilization, 'utilization_without_data_loading')
            utilization.tail = str(round(cost_model_output.utilization.mac_utilize_no_load, 4))

            # mac_utilize_spatial = ET.SubElement(utilization, 'utilization_spatial')
            # mac_utilize_spatial.tail = str(round(cost_model_output.utilization.mac_utilize_spatial, 4))
            # mac_utilize_temporal = ET.SubElement(utilization, 'utilization_temporal_with_data_loading')
            # mac_utilize_temporal.tail = str(round(cost_model_output.utilization.mac_utilize_temporal, 4))
            # mac_utilize_temporal_no_load = ET.SubElement(utilization, 'utilization_temporal_without_data_loading')
            # mac_utilize_temporal_no_load.tail = str(round(cost_model_output.utilization.mac_utilize_temporal_no_load, 4))

            latency = ET.SubElement(performance, 'latency')
            # latency_tot = ET.SubElement(latency, 'latency_cycle_with_data_loading')
            # latency_tot.tail = str(cost_model_output.utilization.latency_tot)
            # latency_no_load = ET.SubElement(latency, 'latency_cycle_without_data_loading')
            latency.tail = str(cost_model_output.utilization.latency_no_load)
            # total_cycles = ET.SubElement(latency, 'ideal_computing_cycle')
            # total_cycles.tail = str(cost_model_output.temporal_loop.total_cycles)

            area = ET.SubElement(results, 'area')
            area.tail = str(cost_model_output.area)
    elapsed_time_node = ET.SubElement(sim, 'elapsed_time_second')
    elapsed_time_node.tail = str(round(elapsed_time, 3))
    tree = ET.ElementTree(root)
    tree.write(results_filename + '.xml')

    if cost_model_output in [None, [], {}]:
        print('L: ', common_settings.layer_index, ' M: ', common_settings.mem_scheme_count,
              ' SU: ', common_settings.spatial_count, ' cost model output', cost_model_output)
        return
    else:
        print_good_su_format(cost_model_output.spatial_scheme, mem_scheme.mem_name, results_filename + '.mapping')
        print_good_tm_format(cost_model_output.temporal_scheme, mem_scheme.mem_name, results_filename + '.mapping')

def print_helper(input_settings, layers, multi_manager):

    # Use this for other print types (such as yaml) in the future
    print_type = 'xml'

    save_all_arch = input_settings.arch_search_result_saving
    save_all_su = input_settings.su_search_result_saving

    # Set mode based on input settings
    fixed_mem = input_settings.mem_hierarchy_single_simulation
    fixed_su = input_settings.fixed_spatial_unrolling
    fixed_tm = input_settings.fixed_temporal_mapping

    rf_base = input_settings.results_path + '%s' + input_settings.results_filename
    rf_ending_en = '_min_en'
    rf_ending_ut = '_max_ut'

    # if mode == 1: # HW Cost
    #     sub_path = '/fixed_tm_for_fixed_su/'
    # elif mode == 2: # TM 
    #     sub_path = '/best_tm_for_fixed_su/'
    # elif mode == 3: # SU + TM
    #     sub_path = '/best_su_for_fixed_mem/'
    # elif mode == 4: # Arch + TM
    #     sub_path = '/best_su_for_fixed_mem/'
    # elif mode == 5: # Arch + SU + TM
    #     sub_path = '/best_su_for_fixed_mem/'

    list_min_en_output = multi_manager.list_min_en_output
    list_tm_count_en = multi_manager.list_tm_count_en
    list_sim_time = multi_manager.list_sim_time
    list_max_ut_output = multi_manager.list_max_ut_output
    list_tm_count_ut = multi_manager.list_tm_count_ut
    list_su_count = multi_manager.list_su_count
    list_sim_time_en = multi_manager.list_sim_time_en
    list_sim_time_ut = multi_manager.list_sim_time_ut

    # Iterate through the processed layers
    for i, layer_index in enumerate(input_settings.layer_number):

        layer_idx_str = 'L_%d' % layer_index
        layer = cls.Layer.extract_layer_info(multi_manager.layer_spec.layer_info[layer_index])

        if save_all_arch or input_settings.mem_hierarchy_single_simulation:
            for mem_scheme_index in range(multi_manager.mem_scheme_count):

                mem_scheme_str = 'M_%d' % (mem_scheme_index + 1)
                msc = multi_manager.mem_scheme_sim[mem_scheme_index]

                if save_all_su:
                    # Save all the SU + best TM combinations
                    su_count = list_su_count[mem_scheme_str][layer_idx_str]

                    if su_count is None:
                        # su count is None for duplicate layers, get su count from parent
                        parent_str = 'L_%d' % layers[i].parent
                        su_count = list_su_count[mem_scheme_str][parent_str]

                    for i in range(1, su_count + 1):
                        mem_scheme_su_str = '%s_SU_%d_%d' % (mem_scheme_str, su_count, i)

                        best_output_energy = list_min_en_output[mem_scheme_str][layer_idx_str]['best_tm_each_su'][mem_scheme_su_str]
                        tm_count_en = list_tm_count_en[mem_scheme_str][layer_idx_str]['best_tm_each_su'][mem_scheme_su_str]
                        sim_time = list_sim_time[mem_scheme_str][layer_idx_str]['best_tm_each_su'][mem_scheme_su_str]

                        best_output_utilization = list_max_ut_output[mem_scheme_str][layer_idx_str]['best_tm_each_su'][mem_scheme_su_str]
                        tm_count_ut = list_tm_count_ut[mem_scheme_str][layer_idx_str]['best_tm_each_su'][mem_scheme_su_str]

                        mem_scheme_count_str = '%d/%d' % (mem_scheme_index + 1, multi_manager.mem_scheme_count)
                        spatial_unrolling_count = str(i) + '/' + str(su_count)
                        common_settings = CommonSetting(input_settings, layer_index, mem_scheme_count_str, spatial_unrolling_count, msc)

                        mem_scheme_su_save_str = '_M%d_SU%d' % (mem_scheme_index + 1, i)

                        sub_path = '/all_su_best_tm/'

                        rf_en = (rf_base % sub_path) + '_L' + str(layer_index) + mem_scheme_su_save_str + rf_ending_en
                        rf_ut = (rf_base % sub_path) + '_L' + str(layer_index) + mem_scheme_su_save_str + rf_ending_ut

                        print_xml(rf_en, layer, msc, best_output_energy, common_settings, tm_count_en, sim_time, input_settings.result_print_mode)
                        print_xml(rf_ut, layer, msc, best_output_utilization, common_settings, tm_count_ut, sim_time, input_settings.result_print_mode)
    
                # Save the best SU + TM combination
                [[mem_scheme_su_str_en, best_output_energy]] = list_min_en_output[mem_scheme_str][layer_idx_str]['best_su_each_mem'].items()
                [[mem_scheme_su_str_en, tm_count_en]] = list_tm_count_en[mem_scheme_str][layer_idx_str]['best_su_each_mem'].items()
                sim_time_en = list_sim_time[mem_scheme_str][layer_idx_str]['best_su_each_mem'][mem_scheme_su_str_en]

                [[mem_scheme_su_str_ut, best_output_utilization]] = list_max_ut_output[mem_scheme_str][layer_idx_str]['best_su_each_mem'].items()
                [[mem_scheme_su_str_ut, tm_count_ut]] = list_tm_count_ut[mem_scheme_str][layer_idx_str]['best_su_each_mem'].items()
                sim_time_ut = list_sim_time[mem_scheme_str][layer_idx_str]['best_su_each_mem'][mem_scheme_su_str_ut]
                

                mem_scheme_count_str = '%d/%d' % (mem_scheme_index + 1, multi_manager.mem_scheme_count)
                spatial_unrolling_count_en = str(mem_scheme_su_str_en.split('_')[-1]) + '/' + str(mem_scheme_su_str_en.split('_')[-2])
                spatial_unrolling_count_ut = str(mem_scheme_su_str_ut.split('_')[-1]) + '/' + str(mem_scheme_su_str_ut.split('_')[-2])
                common_settings_en = CommonSetting(input_settings, layer_index, mem_scheme_count_str, spatial_unrolling_count_en, msc)
                common_settings_ut = CommonSetting(input_settings, layer_index, mem_scheme_count_str, spatial_unrolling_count_ut, msc)

                mem_scheme_su_save_str_en = '_M%d_SU%s' %(mem_scheme_index + 1, str(mem_scheme_su_str_en.split('_')[-1]))
                mem_scheme_su_save_str_ut = '_M%d_SU%s' %(mem_scheme_index + 1, str(mem_scheme_su_str_ut.split('_')[-1]))

                sub_path = '/best_su_best_tm/'

                rf_en = (rf_base % sub_path) + '_L' + str(layer_index) + mem_scheme_su_save_str_en + rf_ending_en
                rf_ut = (rf_base % sub_path) + '_L' + str(layer_index) + mem_scheme_su_save_str_ut + rf_ending_ut

                print_xml(rf_en, layer, msc, best_output_energy, common_settings_en, tm_count_en, sim_time_en, input_settings.result_print_mode)
                print_xml(rf_ut, layer, msc, best_output_utilization, common_settings_ut, tm_count_ut, sim_time_ut, input_settings.result_print_mode)

        else:
            # Only save the best memory + su + tm combination
            [[mem_scheme_su_str_en, best_output_energy]] = list_min_en_output['best_mem_each_layer'][layer_idx_str].items()
            [[mem_scheme_su_str_en, tm_count_en]] = list_tm_count_en['best_mem_each_layer'][layer_idx_str].items()
            sim_time_en = list_sim_time_en['best_mem_each_layer'][layer_idx_str]

            [[mem_scheme_su_str_ut, best_output_utilization]] = list_max_ut_output['best_mem_each_layer'][layer_idx_str].items()

            [[mem_scheme_su_str_ut, tm_count_ut]] = list_tm_count_ut['best_mem_each_layer'][layer_idx_str].items()
            sim_time_ut = list_sim_time_ut['best_mem_each_layer'][layer_idx_str]

            mem_scheme_index_en = int(mem_scheme_su_str_en.split('_')[1]) - 1
            mem_scheme_index_ut = int(mem_scheme_su_str_ut.split('_')[1]) - 1

            msc_en = multi_manager.mem_scheme_sim[mem_scheme_index_en]
            msc_ut = multi_manager.mem_scheme_sim[mem_scheme_index_ut]

            mem_scheme_count_str_en = '%d/%d' % (mem_scheme_index_en + 1, multi_manager.mem_scheme_count)
            mem_scheme_count_str_ut = '%d/%d' % (mem_scheme_index_ut + 1, multi_manager.mem_scheme_count)

            spatial_unrolling_count_en = str(mem_scheme_su_str_en.split('_')[-1]) + '/' + str(mem_scheme_su_str_en.split('_')[-2])
            spatial_unrolling_count_ut = str(mem_scheme_su_str_ut.split('_')[-1]) + '/' + str(mem_scheme_su_str_ut.split('_')[-2])
            common_settings_en = CommonSetting(input_settings, layer_index, mem_scheme_count_str_en, spatial_unrolling_count_en, msc_en)
            common_settings_ut = CommonSetting(input_settings, layer_index, mem_scheme_count_str_ut, spatial_unrolling_count_ut, msc_ut)

            mem_scheme_su_save_str_en = '_M%d_SU%s' %(mem_scheme_index_en + 1, str(mem_scheme_su_str_en.split('_')[-1]))
            mem_scheme_su_save_str_ut = '_M%d_SU%s' %(mem_scheme_index_ut + 1, str(mem_scheme_su_str_ut.split('_')[-1]))

            sub_path = '/best_mem_each_layer/'

            rf_en = (rf_base % sub_path) + '_L' + str(layer_index) + mem_scheme_su_save_str_en + rf_ending_en
            rf_ut = (rf_base % sub_path) + '_L' + str(layer_index) + mem_scheme_su_save_str_ut + rf_ending_ut

            print_xml(rf_en, layer, msc_en, best_output_energy, common_settings_en, tm_count_en, sim_time_en, input_settings.result_print_mode)
            print_xml(rf_ut, layer, msc_ut, best_output_utilization, common_settings_ut, tm_count_ut, sim_time_ut, input_settings.result_print_mode)
    

        if (len(input_settings.layer_number) > 1) and (multi_manager.mem_scheme_count > 1):
            # Save the best memory hierarchy for all layers in separate folder

            best_mem_scheme_idx_en = multi_manager.best_mem_scheme_index_en
            best_mem_scheme_idx_ut = multi_manager.best_mem_scheme_index_ut
            best_mem_scheme_str_en = 'M_%d' % (best_mem_scheme_idx_en + 1)
            best_mem_scheme_str_ut = 'M_%d' % (best_mem_scheme_idx_ut + 1)
            
            [[mem_scheme_su_str_en, best_output_energy]] = list_min_en_output[best_mem_scheme_str_en][layer_idx_str]['best_su_each_mem'].items()
            [[mem_scheme_su_str_en, tm_count_en]] = list_tm_count_en[best_mem_scheme_str_en][layer_idx_str]['best_su_each_mem'].items()
            sim_time_en = list_sim_time[best_mem_scheme_str_en][layer_idx_str]['best_su_each_mem'][mem_scheme_su_str_en]

            [[mem_scheme_su_str_ut, best_output_utilization]] = list_max_ut_output[best_mem_scheme_str_ut][layer_idx_str]['best_su_each_mem'].items()
            [[mem_scheme_su_str_ut, tm_count_ut]] = list_tm_count_ut[best_mem_scheme_str_ut][layer_idx_str]['best_su_each_mem'].items()
            sim_time_ut = list_sim_time[best_mem_scheme_str_ut][layer_idx_str]['best_su_each_mem'][mem_scheme_su_str_ut]

            msc_en = multi_manager.mem_scheme_sim[best_mem_scheme_idx_en]
            msc_ut = multi_manager.mem_scheme_sim[best_mem_scheme_idx_ut]

            mem_scheme_count_str_en = '%d/%d' % (best_mem_scheme_idx_en + 1, multi_manager.mem_scheme_count)
            mem_scheme_count_str_ut = '%d/%d' % (best_mem_scheme_idx_ut + 1, multi_manager.mem_scheme_count)

            spatial_unrolling_count_en = str(mem_scheme_su_str_en.split('_')[-1]) + '/' + str(mem_scheme_su_str_en.split('_')[-2])
            spatial_unrolling_count_ut = str(mem_scheme_su_str_ut.split('_')[-1]) + '/' + str(mem_scheme_su_str_ut.split('_')[-2])
            common_settings_en = CommonSetting(input_settings, layer_index, mem_scheme_count_str_en, spatial_unrolling_count_en, msc_en)
            common_settings_ut = CommonSetting(input_settings, layer_index, mem_scheme_count_str_ut, spatial_unrolling_count_ut, msc_ut)

            mem_scheme_su_save_str_en = '_M%d_SU%s' %(best_mem_scheme_idx_en + 1, str(mem_scheme_su_str_en.split('_')[-1]))
            mem_scheme_su_save_str_ut = '_M%d_SU%s' %(best_mem_scheme_idx_ut + 1, str(mem_scheme_su_str_ut.split('_')[-1]))

            sub_path = '/best_mem_network/'

            rf_en = (rf_base % sub_path) + '_L' + str(layer_index) + mem_scheme_su_save_str_en + rf_ending_en
            rf_ut = (rf_base % sub_path) + '_L' + str(layer_index) + mem_scheme_su_save_str_ut + rf_ending_ut

            print_xml(rf_en, layer, msc_en, best_output_energy, common_settings_en, tm_count_en, sim_time_en, input_settings.result_print_mode)
            print_xml(rf_ut, layer, msc_ut, best_output_utilization, common_settings_ut, tm_count_ut, sim_time_ut, input_settings.result_print_mode)



class CostModelOutput:

    def __init__(self, total_cost, operand_cost, mac_cost, temporal_scheme, spatial_scheme, flooring, loop,
                 spatial_loop, temporal_loop, area, utilization, shared_bw):
        self.operand_cost = operand_cost
        self.total_cost = total_cost
        self.temporal_scheme = temporal_scheme
        self.spatial_scheme = spatial_scheme
        self.flooring = flooring
        self.loop = loop
        self.mac_cost = mac_cost
        self.spatial_loop = spatial_loop
        self.temporal_loop = temporal_loop
        self.area = area
        self.utilization = utilization
        self.shared_bw = shared_bw


class CommonSetting:
    def __init__(self, input_settings, ii_layer_index, mem_scheme_count, spatial_unrolling_count, mem_scheme):
        self.search_mode = {
            'Mem': 'fixed' if input_settings.mem_hierarchy_single_simulation
            else 'iterative search' if input_settings.mem_hierarchy_iterative_search
            else 'exhaustive search',

            'Spatial': 'fixed' if input_settings.fixed_spatial_unrolling
            else 'exhaustive search' if input_settings.spatial_unrolling_mode == 0
            else 'heuristic search v1' if input_settings.spatial_unrolling_mode == 1
            else 'heuristic search v2' if input_settings.spatial_unrolling_mode == 2
            else 'hint-driven search',

            'Temporal': 'fixed' if input_settings.fixed_temporal_mapping
            else 'exhaustive search' if (
                    input_settings.tmg_search_method == 1 and input_settings.stationary_optimization_enable == False and input_settings.drc_enabled == False)
            else 'heuristic search v1' if (
                    input_settings.tmg_search_method == 1 and input_settings.stationary_optimization_enable == True and input_settings.drc_enabled == False)
            else 'heuristic search v2' if (
                    input_settings.tmg_search_method == 1 and input_settings.stationary_optimization_enable == True and input_settings.drc_enabled == True)
            else 'iterative search' if (
                    input_settings.tmg_search_method == 0)
            else 'c++ search'
        }
        self.area_constraint = {
            'max_area': input_settings.max_area,
            'area_th': input_settings.utilization_rate_area}
        self.mem_pool = input_settings.mem_pool
        self.memory_hierarchy_ratio = input_settings.memory_hierarchy_ratio
        self.max_inner_PE_mem_size = input_settings.PE_RF_size_threshold
        self.max_inner_PE_mem_level = input_settings.PE_RF_depth
        self.max_outer_PE_mem_level = input_settings.CHIP_depth
        self.mem_scheme_count = mem_scheme_count
        self.spatial_count = spatial_unrolling_count
        self.SU_threshold = input_settings.spatial_utilization_threshold
        self.mem_utilization_rate = mem_scheme.mem_utilization_rate_fixed
        self.precision = {'W': input_settings.precision['W'], 'I': input_settings.precision['I'],
                          'O_partial': input_settings.precision['O'],
                          'O_final': input_settings.precision['O_final']}
        self.array_size = {'Col': input_settings.mac_array_info['array_size'][0],
                           'Row': input_settings.mac_array_info['array_size'][1]}
        self.spatial_unrolling_mode = input_settings.spatial_unrolling_mode
        self.spatial_mapping_hint_list = input_settings.unrolling_scheme_list_text

        try:
            mem_access_cost = {'W': [], 'I': [], 'O': []}
            for operand in ['W', 'I', 'O']:
                for tup in mem_scheme.mem_cost[operand]:
                    if type(tup[0]) is tuple:
                        mem_access_cost[operand].append(list(tup[0]))
                    else:
                        mem_access_cost[operand].append(list(tup))
        except:
            mem_access_cost = deepcopy(mem_scheme.mem_cost)
        self.mem_access_cost = mem_access_cost
        self.layer_index = ii_layer_index


def xml_info_extraction():
    for m in range(1, 11):
        for su in range(1, 2):
            file_path_and_name = './results/case_study1_6/16x16/DarkNet19/best_tm_for_each_su/DarkNet19_L4_M%d_SU%d_min_en.xml' % (
                m, su)
            if os.path.isfile(file_path_and_name):
                tree = ET.parse(file_path_and_name)
                root = tree.getroot()
                for sim in root:
                    mem_size_bit = ast.literal_eval(sim.find("hw_spec/memory_hierarchy/mem_size_bit").tail)
                    print('M', m, ' SU', su, mem_size_bit)
                    # unrolling = ast.literal_eval(sim.find("results/basic_info/spatial_unrolling/W").tail)
                    # print('M', m, ' SU', su, unrolling)
                    area = ast.literal_eval(sim.find("results/area").tail)
                    print('M', m, ' SU', su, area)

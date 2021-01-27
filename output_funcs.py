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
from im2col_funcs import pw_layer_col2im
import time

# Standard library
from typing import Dict, Any    # Used for type hints
import json # Used to create the output YAML file

# External imports
import numpy as np  # Used to get access to numpy types in yaml_compatible

# Internal imports
from classes.layer import Layer # Used for print_yaml
from input_funcs import InputSettings   # Used for print_yaml
from msg import MemoryScheme # Used for print_yaml



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


def su_reformat_if_need(su):
    new_su = {'W': [], 'I': [], 'O': []}
    # extract unrolled dimension on row and on column
    for op in ['W', 'I', 'O']:
        for level_list in su[op]:
            # TODO assume that at least one operand at one level is fully 2D unrolled.
            if len(level_list) == 2:
                row_list = [x[0] for x in level_list[0]]
                col_list = [x[0] for x in level_list[1]]
                ok = True
                break
        if ok:
            break

    # for each 1D-unrolled mem/mac level, distinguish column from row
    for op in ['W', 'I', 'O']:
        for level_list in su[op]:
            if len(level_list) in [0,2]:
                new_su[op].append(level_list)
            elif len(level_list) == 1:
                if level_list[0][0][0] in row_list:
                    level_list.append([])
                elif level_list[0][0][0] in col_list:
                    level_list.insert(0, [])
                else:
                    print('su:', su)
                    raise ValueError('NO1. Please check the SU.')
                new_su[op].append(level_list)
            else:
                print('su:', su)
                raise ValueError('NO2. Please check the SU.')

    return new_su


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
    # print(su, mem_name, file_path_name)
    try:
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
                                         "(Notes: Unrolled loops' order doesn't matter; D1 and D2 are PE "
                                         "array's two geometric dimensions. )")
        # print mem name to each level
        XY_name = {0: 'D1', 1: 'D2'}
        position_save = [[], []]  # for I and O based on W

        for idx, operand in enumerate(['W']):
            column_position = tot_col_cut + idx * interval
            su_block = modify_printing_block(su_block, 12, column_position, operand)
            i = 0
            for level, lv_li in enumerate(su[operand]):
                for xy, xy_li in enumerate(lv_li):
                    for tt in xy_li:
                        position_save[0].extend([tt[0], XY_name[xy], finish_row - 2 * i])
                        position_save[1].extend([tt[0], XY_name[xy], finish_row - 2 * i])
                        su_block = modify_printing_block(su_block, finish_row - 2 * i, column_position,
                                                         str(mem_name[operand][level]) + ' (' + XY_name[xy] + ')')
                        # print_printing_block(file_path_name, su_block, 'w+')
                        i += 1

        for idx, operand in enumerate(['I', 'O']):
            column_position = tot_col_cut + (idx+1) * interval
            su_block = modify_printing_block(su_block, 12, column_position, operand)
            i = 0
            for level, lv_li in enumerate(su[operand]):
                for xy, xy_li in enumerate(lv_li):
                    for tt in xy_li:
                        indices = [i for i, x in enumerate(position_save[idx]) if x == XY_name[xy]]
                        for ind in indices:
                            # Check if loop type matches.
                            if position_save[idx][ind-1] == tt[0]:
                                line_position_index = ind
                                break
                        line_position = position_save[idx][line_position_index+1]
                        del position_save[idx][line_position_index]
                        del position_save[idx][line_position_index]
                        su_block = modify_printing_block(su_block, line_position, column_position,
                                                         str(mem_name[operand][level]) + ' (' + XY_name[xy] + ')')
                        # print_printing_block(file_path_name, su_block, 'w+')
                        i += 1
    except:
        su = su_reformat_if_need(su)
        su_list = [sp for lv_li in su['W'] for xy_li in lv_li for sp in xy_li]
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
                                         "(Notes: Unrolled loops' order doesn't matter; D1 and D2 are PE "
                                         "array's two geometric dimensions. )")
        # print mem name to each level
        XY_name = {0: 'D1', 1: 'D2'}
        position_save = [[], []]  # for I and O based on W

        for idx, operand in enumerate(['W']):
            column_position = tot_col_cut + idx * interval
            su_block = modify_printing_block(su_block, 12, column_position, operand)
            i = 0
            for level, lv_li in enumerate(su[operand]):
                for xy, xy_li in enumerate(lv_li):
                    for _ in xy_li:
                        position_save[0].extend([XY_name[xy], finish_row - 2 * i])
                        position_save[1].extend([XY_name[xy], finish_row - 2 * i])
                        su_block = modify_printing_block(su_block, finish_row - 2 * i, column_position,
                                                         str(mem_name[operand][level]) + ' (' + XY_name[xy] + ')')
                        # print_printing_block(file_path_name, su_block, 'w+')
                        i += 1

        for idx, operand in enumerate(['I', 'O']):
            column_position = tot_col_cut + (idx + 1) * interval
            su_block = modify_printing_block(su_block, 12, column_position, operand)
            i = 0
            for level, lv_li in enumerate(su[operand]):
                for xy, xy_li in enumerate(lv_li):
                    for _ in xy_li:
                        line_position_index = position_save[idx].index(XY_name[xy])
                        line_position = position_save[idx][line_position_index + 1]
                        del position_save[idx][line_position_index]
                        del position_save[idx][line_position_index]
                        su_block = modify_printing_block(su_block, line_position, column_position,
                                                         str(mem_name[operand][level]) + ' (' + XY_name[xy] + ')')
                        # print_printing_block(file_path_name, su_block, 'w+')
                        i += 1

    print_printing_block(file_path_name, su_block, 'w+')


def handle_grouped_convolutions(
    layer_specification: Layer, cost_model_output: "CostModelOutput", 
    ) -> Any:
    """
    Corrects various values to account for the grouped convolutions.

    Arguments
    =========
     - layer_specification: A description of the input layer that ZigZag has
        optimized.
     - cost_model_output: The cost computed by ZigZag for running the given
        layer on the hardware.

    Returns
    =======
    A tuple of 17 elements, in order:
     - size_list_output_print,
     - total_MAC_op_number,
     - total_data_size_number,
     - mem_access_elem,
     - total_cost,
     - operand_cost,
     - mac_cost_active,
     - mac_cost_idle,
     - latency_tot_number,
     - latency_no_load_number,
     - total_cycles_number,
     - cc_load_tot_number,
     - cc_load_number,
     - cc_load_comb_number,
     - cc_mem_stall_tot_number,
     - stall_cc_number,
     - stall_cc_mem_share_number,
    """
    # If layer has a number of groups > 1, transform relevant variables.
    # At this point, the results are those of one group, so multiply by number of groups.
    group_count = layer_specification.G

    # SPECIFICATION
    size_list_output_print = deepcopy(layer_specification.size_list_output_print)
    size_list_output_print['C'] *= group_count
    size_list_output_print['K'] *= group_count

    # COMPUTATIONS
    total_MAC_op_number = group_count * layer_specification.total_MAC_op

    # DATA SIZE
    total_data_size_number = deepcopy(layer_specification.total_data_size)
    total_data_size_number['W'] *= group_count
    total_data_size_number['I'] *= group_count
    total_data_size_number['O'] *= group_count

    # MEMORY ACCESS
    mem_access_elem = digit_truncate(deepcopy(cost_model_output.loop.mem_access_elem), 0)
    mem_access_elem['W'] = [[group_count * elem for elem in sublist] for sublist in mem_access_elem['W']]
    mem_access_elem['I'] = [[group_count * elem for elem in sublist] for sublist in mem_access_elem['I']]
    mem_access_elem['O'] = [[group_count * elem for elem in sublist] for sublist in mem_access_elem['O']]
    mem_access_elem['O_partial'] = [[group_count * elem for elem in sublist] for sublist in mem_access_elem['O_partial']]
    mem_access_elem['O_final'] = [[group_count * elem for elem in sublist] for sublist in mem_access_elem['O_final']]

    # ENERGY
    total_cost = round(group_count * cost_model_output.total_cost, 1)
    operand_cost = energy_clean(deepcopy(cost_model_output.operand_cost))
    operand_cost['W'] = [group_count * elem for elem in operand_cost['W']]
    operand_cost['I'] = [group_count * elem for elem in operand_cost['I']]
    operand_cost['O'] = [group_count * elem for elem in operand_cost['O']]
    mac_cost_active = round(group_count * cost_model_output.mac_cost[0], 1)
    mac_cost_idle = round(group_count * cost_model_output.mac_cost[1], 1)

    # LATENCY
    latency_tot_number = group_count * cost_model_output.utilization.latency_tot
    latency_no_load_number = group_count * cost_model_output.utilization.latency_no_load
    total_cycles_number = group_count * cost_model_output.temporal_loop.total_cycles

    cc_load_tot_number = group_count * cost_model_output.utilization.cc_load_tot
    cc_load_number = deepcopy(cost_model_output.utilization.cc_load)
    cc_load_number['W'] = [group_count * elem for elem in cc_load_number['W']]
    cc_load_number['I'] = [group_count * elem for elem in cc_load_number['I']]
    cc_load_comb_number = deepcopy(cost_model_output.utilization.cc_load_comb)
    cc_load_comb_number['W'] *= group_count
    cc_load_comb_number['I'] *= group_count

    cc_mem_stall_tot_number = group_count * cost_model_output.utilization.cc_mem_stall_tot
    stall_cc_number = deepcopy(cost_model_output.utilization.stall_cc)
    stall_cc_number['W'] = [[group_count * elem for elem in sublist] for sublist in stall_cc_number['W']]
    stall_cc_number['I'] = [[group_count * elem for elem in sublist] for sublist in stall_cc_number['I']]
    stall_cc_number['O'] = [[group_count * elem for elem in sublist] for sublist in stall_cc_number['O']]
    stall_cc_mem_share_number = deepcopy(cost_model_output.utilization.stall_cc_mem_share)
    stall_cc_mem_share_number['W'] = [[group_count * elem for elem in sublist] for sublist in stall_cc_mem_share_number['W']]
    stall_cc_mem_share_number['I'] = [[group_count * elem for elem in sublist] for sublist in stall_cc_mem_share_number['I']]
    stall_cc_mem_share_number['O'] = [[group_count * elem for elem in sublist] for sublist in stall_cc_mem_share_number['O']]

    # Returning the tuple of values corrected for the grouped convolution.
    return (
        size_list_output_print,
        total_MAC_op_number,
        total_data_size_number,
        mem_access_elem,
        total_cost,
        operand_cost,
        mac_cost_active,
        mac_cost_idle,
        latency_tot_number,
        latency_no_load_number,
        total_cycles_number,
        cc_load_tot_number,
        cc_load_number,
        cc_load_comb_number,
        cc_mem_stall_tot_number,
        stall_cc_number,
        stall_cc_mem_share_number,
    )



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

    try:
        tree = ET.parse(results_filename + '.xml')
    except:
        # Error thrown because the xml file is corrupt
        os.remove(results_filename + '.xml')
        print("Deleted corrupt xml file:", results_filename + '.xml')
        root = ET.Element('root')
        tree = ET.ElementTree(root)
        tree.write(results_filename + '.xml')
        tree = ET.parse(results_filename + '.xml')

    root = tree.getroot()
    sim = ET.SubElement(root, 'simulation')
    result_generate_time = ET.SubElement(sim, 'result_generated_time')
    result_generate_time.tail = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if cost_model_output not in [None, [], {}]:

        # Correcting the outputed values to account for the grouped convolution.
        (
            size_list_output_print,
            total_MAC_op_number,
            total_data_size_number,
            mem_access_elem,
            total_cost,
            operand_cost,
            mac_cost_active,
            mac_cost_idle,
            latency_tot_number,
            latency_no_load_number,
            total_cycles_number,
            cc_load_tot_number,
            cc_load_number,
            cc_load_comb_number,
            cc_mem_stall_tot_number,
            stall_cc_number,
            stall_cc_mem_share_number,
        ) = handle_grouped_convolutions(layer_specification, cost_model_output)

        if result_print_mode == 'complete':
            layer = ET.SubElement(sim, 'layer')
            # layer_index = ET.SubElement(layer, 'layer_index')
            # layer_index.tail = str(common_settings.layer_index)
            layer_spec = ET.SubElement(layer, 'layer_spec')
            # layer_spec.tail = str(layer_specification.size_list_output_print)
            layer_spec.tail = str(layer_specification.size_list_output_print)
            im2col_enable = ET.SubElement(layer, 'im2col_enable')
            im2col_enable.tail = str(common_settings.im2col_enable)
            total_MAC_op = ET.SubElement(layer, 'total_MAC_operation')
            # total_MAC_op.tail = str(layer_specification.total_MAC_op)
            total_MAC_op.tail = str(total_MAC_op_number)
            total_data_size = ET.SubElement(layer, 'total_data_size_element')
            # total_data_size.tail = str(layer_specification.total_data_size)
            total_data_size.tail = str(total_data_size_number)
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
            # if not common_settings.search_mode['Temporal'] == 'fixed':
            #     memory_utilization_threshold = ET.SubElement(temporal_mapping_search_engine, 'memory_utilization_hint')
            #     memory_utilization_threshold.tail = str(common_settings.mem_utilization_rate)
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
            greedy_mapping_flag = ET.SubElement(spatial_mapping, 'greedy_mapping')
            greedy_mapping_flag.tail = str(cost_model_output.greedy_mapping_flag)
            if cost_model_output.greedy_mapping_flag:
                footer_info = ET.SubElement(spatial_mapping, 'footer_info')
                footer_info.tail = str(cost_model_output.footer_info)
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
            # print('final memory ut', cost_model_output.utilization.mem_utilize_shared)
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
            # access_count_W.tail = str(cost_model_output.loop.mem_access_elem['W'])
            access_count_W.tail = str(mem_access_elem['W'])
            access_count_I = ET.SubElement(access_count, 'I')
            # access_count_I.tail = str(cost_model_output.loop.mem_access_elem['I'])
            access_count_I.tail = str(mem_access_elem['I'])
            access_count_O = ET.SubElement(access_count, 'O')
            # access_count_O.tail = str(cost_model_output.loop.mem_access_elem['O'])
            access_count_O.tail = str(mem_access_elem['O'])
            access_count_O_partial = ET.SubElement(access_count, 'O_partial')
            # access_count_O_partial.tail = str(cost_model_output.loop.mem_access_elem['O_partial'])
            access_count_O_partial.tail = str(mem_access_elem['O_partial'])
            access_count_O_final = ET.SubElement(access_count, 'O_final')
            # access_count_O_final.tail = str(cost_model_output.loop.mem_access_elem['O_final'])
            access_count_O_final.tail = str(mem_access_elem['O_final'])
            # array_element_distance = ET.SubElement(basic_info, 'array_element_distance')
            # array_element_distance.tail = str(cost_model_output.loop.array_wire_distance)

            energy = ET.SubElement(results, 'energy')
            minimum_cost = ET.SubElement(energy, 'total_energy')
            # minimum_cost.tail = str(round(cost_model_output.total_cost, 1))
            minimum_cost.tail = str(total_cost)
            energy_breakdown = ET.SubElement(energy, 'mem_energy_breakdown')
            cost_model_output.operand_cost = energy_clean(cost_model_output.operand_cost)
            energy_breakdown_W = ET.SubElement(energy_breakdown, 'W')
            # energy_breakdown_W.tail = str(cost_model_output.operand_cost['W'])
            energy_breakdown_W.tail = str(operand_cost['W'])
            energy_breakdown_I = ET.SubElement(energy_breakdown, 'I')
            # energy_breakdown_I.tail = str(cost_model_output.operand_cost['I'])
            energy_breakdown_I.tail = str(operand_cost['I'])
            energy_breakdown_O = ET.SubElement(energy_breakdown, 'O')
            # energy_breakdown_O.tail = str(cost_model_output.operand_cost['O'])
            energy_breakdown_O.tail = str(operand_cost['O'])

            mac_cost = ET.SubElement(energy, 'mac_energy')
            mac_cost.tail = 'active: ' + str(mac_cost_active) + \
                            ', idle: ' + str(mac_cost_idle)

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
            mac_utilize_temporal_no_load.tail = str(round(cost_model_output.utilization.mac_utilize_temporal_no_load, 4))


            latency = ET.SubElement(performance, 'latency')
            latency_tot = ET.SubElement(latency, 'latency_cycle_with_data_loading')
            # latency_tot.tail = str(cost_model_output.utilization.latency_tot)
            latency_tot.tail = str(latency_tot_number)
            latency_no_load = ET.SubElement(latency, 'latency_cycle_without_data_loading')
            # latency_no_load.tail = str(cost_model_output.utilization.latency_no_load)
            latency_no_load.tail = str(latency_no_load_number)
            total_cycles = ET.SubElement(latency, 'ideal_computing_cycle')
            # total_cycles.tail = str(cost_model_output.temporal_loop.total_cycles)
            total_cycles.tail = str(total_cycles_number)

            data_loading = ET.SubElement(latency, 'data_loading')
            cc_load_tot = ET.SubElement(data_loading, 'load_cycle_total')
            # cc_load_tot.tail = str(cost_model_output.utilization.cc_load_tot)
            cc_load_tot.tail = str(cc_load_tot_number)
            cc_load = ET.SubElement(data_loading, 'load_cycle_individual')
            # cc_load.tail = str(cost_model_output.utilization.cc_load)
            cc_load.tail = str(cc_load_number)
            cc_load_comb = ET.SubElement(data_loading, 'load_cycle_combined')
            # cc_load_comb.tail = str(cost_model_output.utilization.cc_load_comb)
            cc_load_comb.tail = str(cc_load_comb_number)

            mem_stalling = ET.SubElement(latency, 'mem_stalling')
            cc_mem_stall_tot = ET.SubElement(mem_stalling, 'mem_stall_cycle_total')
            # cc_mem_stall_tot.tail = str(cost_model_output.utilization.cc_mem_stall_tot)
            cc_mem_stall_tot.tail = str(cc_mem_stall_tot_number)
            stall_cc = ET.SubElement(mem_stalling, 'mem_stall_cycle_individual')
            # stall_cc.tail = str(cost_model_output.utilization.stall_cc)
            stall_cc.tail = str(stall_cc_number)
            stall_cc_mem_share = ET.SubElement(mem_stalling, 'mem_stall_cycle_shared')
            # stall_cc_mem_share.tail = str(cost_model_output.utilization.stall_cc_mem_share)
            stall_cc_mem_share.tail = str(stall_cc_mem_share_number)

            req_mem_bw_bit = ET.SubElement(mem_stalling, 'req_mem_bw_bit_per_cycle_individual')
            req_mem_bw_bit.tail = str(digit_truncate(cost_model_output.utilization.req_mem_bw_bit, 1))
            req_sh_mem_bw_bit = ET.SubElement(mem_stalling, 'req_mem_bw_bit_per_cycle_shared')
            req_sh_mem_bw_bit.tail = str(digit_truncate(cost_model_output.utilization.req_sh_mem_bw_bit, 1))
            mem_bw_req_meet_list = mem_bw_req_meet_check(cost_model_output.utilization.req_sh_mem_bw_bit,
                                                         mem_scheme.mem_bw)
            mem_bw_req_meet = ET.SubElement(mem_stalling, 'mem_bw_requirement_meet')
            mem_bw_req_meet.tail = str(mem_bw_req_meet_list)

            area = ET.SubElement(results, 'area')
            total_area = ET.SubElement(area, 'total_area')
            total_area.tail = str(round(cost_model_output.area[0],1))
            active_area = ET.SubElement(area, 'active_area')
            active_area.tail = str(round(cost_model_output.area[1],1))
            dark_silicon = ET.SubElement(area, 'dark_silicon_percentage')
            dark_silicon.tail = str(100-round(cost_model_output.area[1]/cost_model_output.area[0]*100, 1)) + ' %'
        else:
            layer = ET.SubElement(sim, 'layer')
            # layer_index = ET.SubElement(layer, 'layer_index')
            # layer_index.tail = str(common_settings.layer_index)
            layer_spec = ET.SubElement(layer, 'layer_spec')
            # layer_spec.tail = str(layer_specification.size_list_output_print)
            layer_spec.tail = str(size_list_output_print)
            im2col_enable = ET.SubElement(layer, 'im2col_enable')
            im2col_enable.tail = str(common_settings.im2col_enable)
            total_MAC_op = ET.SubElement(layer, 'total_MAC_operation')
            # total_MAC_op.tail = str(layer_specification.total_MAC_op)
            total_MAC_op.tail = str(total_MAC_op_number)
            total_data_size = ET.SubElement(layer, 'total_data_size_element')
            # total_data_size.tail = str(layer_specification.total_data_size)
            total_data_size.tail = str(total_data_size_number)
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
            # minimum_cost.tail = str(round(cost_model_output.total_cost, 1))
            minimum_cost.tail = str(total_cost)
            energy_breakdown = ET.SubElement(energy, 'energy_breakdown')
            cost_model_output.operand_cost = energy_clean(cost_model_output.operand_cost)
            energy_breakdown_W = ET.SubElement(energy_breakdown, 'mem_energy_W')
            # energy_breakdown_W.tail = str(cost_model_output.operand_cost['W'])
            energy_breakdown_W.tail = str(operand_cost['W'])
            energy_breakdown_I = ET.SubElement(energy_breakdown, 'mem_energy_I')
            # energy_breakdown_I.tail = str(cost_model_output.operand_cost['I'])
            energy_breakdown_I.tail = str(operand_cost['I'])
            energy_breakdown_O = ET.SubElement(energy_breakdown, 'mem_energy_O')
            # energy_breakdown_O.tail = str(cost_model_output.operand_cost['O'])
            energy_breakdown_O.tail = str(operand_cost['O'])

            mac_cost = ET.SubElement(energy, 'mac_energy')
            mac_cost.tail = 'active: ' + str(mac_cost_active) + ', idle: ' + str(mac_cost_idle)

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
            total_area = ET.SubElement(area, 'total_area')
            total_area.tail = str(round(cost_model_output.area[0],1))
            active_area = ET.SubElement(area, 'active_area')
            active_area.tail = str(round(cost_model_output.area[1],1))
            dark_silicon = ET.SubElement(area, 'dark_silicon_percentage')
            dark_silicon.tail = str(100-round(cost_model_output.area[1]/cost_model_output.area[0]*100, 1)) + ' %'

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


def yaml_compatible(argument: Any) -> Any:
    """
    Returns a YAML compatible representation of the argument. This function
    will behave differently depending on the type of the argument.

    Arguments
    =========
     - argument: The python object that should be written to the YAML file.

    Returns
    =======
    A YAML compatible representation of the argument.

    Exceptions
    ==========
    If there is no known YAML compatible representation for the argument type,
    a TypeError is raised.
    """
    # If the argument is a numpy array, we turn it into a list, recursively.
    if isinstance(argument, tuple) or isinstance(argument, list):
        # The argument is some sort of iterable (list, tuple or numpy array),
        # we return its list representation, recursively making its content yaml
        # compatible.
        return [yaml_compatible(element) for element in argument]
    elif isinstance(argument, bool):
        # Bool is YAML compatible.
        return argument
    elif isinstance(argument, float) or isinstance(argument, np.float64):
        # Float is YAML compatible, but not np.float64
        return float(argument)
    elif isinstance(argument, int) or isinstance(argument, np.int64):
        # Int is YAML compatible, but not np.int64
        return int(argument)
    elif isinstance(argument, str):
        # str is YAML compatible.
        return argument
    elif isinstance(argument, dict):
        # Returning the dict with YAML compatible keys and values.
        return {
            yaml_compatible(key): yaml_compatible(value)
            for key, value in argument.items()
        }
    else:
        # Raising an exception because we don't know what to do with the
        # argument.
        raise TypeError(
            "The argument {} is of type {} ".format(argument, type(argument))
            + "which has no known YAML compatible representation."
        )


def print_yaml(
    results_filename: str,
    layer_specification: Layer,
    mem_scheme: MemoryScheme,
    cost_model_output: "CostModelOutput",
    common_settings: "CommonSettings",
    hw_pool_sizes: int,
    elapsed_time: float,
    result_print_mode: str,
):
    """
    Saves the output of ZigZag to the YAML format.

    Arguments
    =========
     - results_filename: The basename used to create the output file of
        ZigZag.
     - layer_specification: A description of the input layer that ZigZag has
        optimized.
     - mem_scheme: A description of the memory layout given to or found by
        ZigZag.
     - cost_model_output: The cost computed by ZigZag for running the given
        layer on the hardware.
     - common_settings: ??
     - hw_pool_sizes: The number of hardware evaluations performed by ZigZag.
     - elapsed_time: The time ZigZag took to run the optimization.
     - result_print_mode: "complete" to see the full output, and some other
        value otherwise

    Exceptions
    ==========
    If the cost model is empty, a ValueError will be raised.

    Side-effects
    ============
    The expected YAML file will be created.
    """
    # NOTE
    # The types given in strings are defined within this very file. This has to
    # do with the order in which python resolves annotations.

    # First, we get the directory in which the output should be created. If the
    # directory doesn't already exist, we create it.
    result_directory_path = os.path.dirname(results_filename)
    # We create the directory and all its parents or do nothing if they already
    # exist.
    os.makedirs(result_directory_path, exist_ok=True)

    # NOTE
    # Unlike the previous XML implementation, if the output file already exists,
    # it will be overwritten.
    #
    # We are going to build a dictionary which holds all our values, then we
    # will dump it into our yaml file.
    yaml_dictionary: Dict[str, Any] = dict()

    # ABREVIATIONS
    # To make the implementation shorter, we are going to give shorted names to
    # some recurrent values:
    #
    # The yaml_compatible function will be referred to as "c"
    c = yaml_compatible
    # A boolean telling us if we should print optional outputs.
    verbose = result_print_mode == "complete"

    # Correcting the outputed values to account for the grouped convolution.
    (
        size_list_output_print,
        total_MAC_op_number,
        total_data_size_number,
        mem_access_elem,
        total_cost,
        operand_cost,
        mac_cost_active,
        mac_cost_idle,
        latency_tot_number,
        latency_no_load_number,
        total_cycles_number,
        cc_load_tot_number,
        cc_load_number,
        cc_load_comb_number,
        cc_mem_stall_tot_number,
        stall_cc_number,
        stall_cc_mem_share_number,
    ) = handle_grouped_convolutions(layer_specification, cost_model_output)


    # Creating the "simulation" section.
    simulation: Dict[str, Any] = dict()
    # Adding the section to our YAML dictionary.
    yaml_dictionary["simulation"] = simulation

    # We add the time of run to the output.
    simulation["result_generation_time"] = c(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    # Handle ZigZag failure.
    if cost_model_output in [None, [], {}]:
        raise ValueError(
            "The cost model returned by ZigZag is empty, perhaps the run did "
            "not succeed."
        )

    # Else we print the output normally.

    # LAYER SECTION
    layer: Dict[str, Any] = dict()
    simulation["layer"] = layer

    if verbose:
        layer["layer_index"] = c(common_settings.layer_index)

    layer["layer_specification"] = c(size_list_output_print)
    layer["total_MAC_operation"] = c(total_MAC_op_number)
    layer["total_data_size_element"] = c(layer_specification.total_data_size)

    if verbose:
        layer["total_data_reuse"] = c(layer_specification.total_data_reuse)
    # END OF LAYER SECTION

    ########################### OPTIONAL OUTPUT HERE ###########################

    if verbose:
        # SEARCH ENGINE SECTION
        search_engine: Dict[str, Any] = dict()
        simulation["search_engine"] = search_engine

        # MEMORY HIERARCHY SEARCH ENGINE SECTION
        # Short for "memory_hierarchy_search_engine"
        mhse: Dict[str, Any] = dict()
        search_engine["memory_hierarchy_search_engine"] = mhse

        mhse["mode"] = c(common_settings.search_mode["Mem"])

        if not common_settings.search_mode["Mem"] == "fixed":
            # If the memory hierarchy was not fixed, we have some more
            # values to output.
            mhse["area_constraint"] = c(
                common_settings.area_constraint["max_area"]
            )
            # Threshold is reached when we are greater than or equal to this
            # value.
            mhse["area_utilization_threshold"] = c(
                common_settings.area_constraint["area_th"]
            )
            mhse["memory_pool"] = c(common_settings.mem_pool)
            # Our value should be greater than or equal to this ratio.
            mhse["consecutive_memory_level_size_ratio"] = c(
                common_settings.memory_hierarchy_ratio
            )
            mhse["maximum_inner_PE_memory_size_bit"] = c(
                common_settings.max_inner_PE_mem_size
            )
            mhse["maximum_inner_PE_memory_level"] = c(
                common_settings.max_inner_PE_mem_level
            )
            mhse["maximum_outer_PE_memory_level"] = c(
                common_settings.max_outer_PE_mem_level
            )

        mhse["memory_scheme_index"] = c(common_settings.mem_scheme_count)
        # END OF MEMORY HIERARCHY SEARCH ENGINE SECTION

        # SPATIAL MAPPING SEARCH SECTION
        # Short for "spatial_mapping_search_engine"
        smse: Dict[str, Any] = dict()
        search_engine["spatial_mapping_search"] = smse

        smse["mode"] = c(common_settings.search_mode["Spatial"])

        if not common_settings.search_mode["Spatial"] == "fixed":
            # If the the spatial mapping was explored.
            if common_settings.spatial_unrolling_mode != 3:
                # ?? I have no idea what this condition.
                smse["spatial_utilization_threshold"] = c(
                    common_settings.SU_threshold
                )
            else:
                smse["spatial_mapping_hint_list"] = c(
                    common_settings.spatial_mapping_hint_list
                )

        smse["unrolling_scheme_index"] = c(common_settings.spatial_count)
        # END OF SPATIAL MAPPING SEARCH ENGINE SECTION

        # TEMPORAL MAPPING SEARCH ENGINE SECTION
        # Short for "temporal_mapping_search_engine"
        tmse: Dict[str, Any] = dict()
        search_engine["temporal_mapping_search"] = tmse

        tmse["mode"] = c(common_settings.search_mode["Temporal"])

        if not common_settings.search_mode["Temporal"] == "fixed":
            # If the temporal mapping was explored.
            tmse["memory_utilization_hint"] = c(
                common_settings.mem_utilization_rate
            )
            tmse["valid_temporal_mapping_found"] = c(hw_pool_sizes)
        # END OF TEMPORAL MAPPING SEARCH ENGINE SECTION
        # END OF SEARCH ENGINE SECTION

    ####################### END OF OPTIONAL OUTPUT HERE ########################

    # HARDWARE SPECIFICATION SECTION
    hardware_specification: Dict[str, Any] = dict()
    simulation["hardware_specification"] = hardware_specification

    # PE ARRAY SECTION
    PE_array: Dict[str, Any] = dict()
    hardware_specification["PE_array"] = PE_array

    PE_array["precision_bit"] = c(common_settings.precision)
    PE_array["array_size"] = c(common_settings.array_size)
    # END OF PE ARRAY SECTION

    # MEMORY HIERARCHY SECTION
    memory_hierarchy: Dict[str, Any] = dict()
    hardware_specification["memory_hierarchy"] = memory_hierarchy

    # MEMORY NAME SECTION
    memory_name: Dict[str, Any] = dict()
    memory_hierarchy["memory_name_in_the_hierarchy"] = memory_name

    memory_name["W"] = c(mem_scheme.mem_name["W"])
    memory_name["I"] = c(mem_scheme.mem_name["I"])
    memory_name["O"] = c(mem_scheme.mem_name["O"])
    # END OF MEMORY NAME SECTION

    memory_hierarchy["memory_size_bit"] = c(mem_scheme.mem_size)

    # TODO
    # Why is there a deepcopy here? We should only be reading data in
    # this function.
    mem_access_cost = deepcopy(common_settings.mem_access_cost)
    # And what is this next block? Not only are we indeed changing the
    # argument data for some non-obvious reason, we are also not
    # printing it, so why is that here? I don't understand...
    if type(mem_scheme.mem_bw["W"][0][0]) in [list, tuple]:
        mem_scheme.mem_bw = iterative_data_format_clean(mem_scheme.mem_bw)
    if type(common_settings.mem_access_cost["W"][0][0]) in [list, tuple]:
        mem_access_cost = iterative_data_format_clean(mem_access_cost)
    if type(mem_scheme.mem_area["W"][0]) in [list, tuple]:
        mem_scheme.mem_area = iterative_data_format_clean(mem_scheme.mem_area)

    # NOTE
    # The original name of this field was:
    # 'mem_bw_bit_per_cycle_or_mem_wordlength', but I don't really
    # understand it so I am going to pick the simpler name.
    memory_hierarchy["memory_word_length"] = c(mem_scheme.mem_bw)
    memory_hierarchy["memory_access_energy_per_word"] = c(mem_access_cost)

    if verbose:
        # TODO
        # pun_factor? What does that mean?
        memory_hierarchy["pun_factor"] = c(
            cost_model_output.utilization.pun_factor
        )

    memory_hierarchy["memory_type"] = c(add_mem_type_name(mem_scheme.mem_type))
    memory_hierarchy["memory_share"] = c(
        mem_share_reformat(mem_scheme.mem_share, mem_scheme.mem_name)
    )
    memory_hierarchy["memory_area_single_module"] = c(mem_scheme.mem_area)
    memory_hierarchy["memory_unrolling"] = c(
        mem_unroll_format(cost_model_output.spatial_loop.unit_count)
    )
    # END OF MEMORY HIERARCHY SECTION
    # END OF HARDWARE SPECIFICATION SECTION

    # RESULTS SECTION
    results: Dict[str, Any] = dict()
    simulation["results"] = results

    # BASIC INFORMATION SECTION
    basic_information: Dict[str, Any] = dict()
    results["basic_information"] = basic_information

    # SPATIAL MAPPING SECTION
    spatial_mapping: Dict[str, Any] = dict()
    basic_information["spatial_unrolling"] = spatial_mapping

    # TODO
    # Why is this try-except here for? What could go wrong?
    # Also except what? Which specific exception would be raised here?
    try:
        spatial_mapping["W"] = c(
            s_loop_name_transfer(cost_model_output.spatial_scheme["W"])
        )
        spatial_mapping["I"] = c(
            s_loop_name_transfer(cost_model_output.spatial_scheme["I"])
        )
        spatial_mapping["O"] = c(
            s_loop_name_transfer(cost_model_output.spatial_scheme["O"])
        )
    except:
        cost_model_output.spatial_scheme = spatial_loop_same_term_merge(
            cost_model_output.spatial_scheme, cost_model_output.flooring
        )
        spatial_mapping["W"] = c(
            s_loop_name_transfer(cost_model_output.spatial_scheme["W"])
        )
        spatial_mapping["I"] = c(
            s_loop_name_transfer(cost_model_output.spatial_scheme["I"])
        )
        spatial_mapping["O"] = c(
            s_loop_name_transfer(cost_model_output.spatial_scheme["O"])
        )
    # END OF SPATIAL MAPPING SECTION

    # TEMPORAL MAPPING SECTION
    temporal_mapping: Dict[str, Any] = dict()
    basic_information["temporal_mapping"] = temporal_mapping

    temporal_mapping["W"] = c(
        t_loop_name_transfer(cost_model_output.temporal_scheme["W"])
    )
    temporal_mapping["I"] = c(
        t_loop_name_transfer(cost_model_output.temporal_scheme["I"])
    )
    temporal_mapping["O"] = c(
        t_loop_name_transfer(cost_model_output.temporal_scheme["O"])
    )
    # END OF TEMPORAL MAPPING SECTION

    ########################### OPTIONAL OUTPUT HERE ###########################

    if verbose:
        # TODO
        # Why are we doing this? This is changing the argument data without
        # printing anything. I guess this has something to do with the
        # digit_truncate call later on, but I have very little clue here.
        try:
            del cost_model_output.loop.data_reuse["I_base"]
            del cost_model_output.loop.data_reuse["I_zigzag"]
        except:
            pass
        basic_information["data_reuse"] = c(
            digit_truncate(cost_model_output.loop.data_reuse, 2)
        )
        # The original name of this fields was
        # I_pr_diagonally_broadcast_or_fifo_effect, but since I don't
        # understand what it means I only kept the simpler name.
        #
        # Also, the content of this field was preceded by a "'I': ", but
        # since that wouldn't be machine readable I am removing it.
        basic_information["fifo_effect"] = c(cost_model_output.loop.I_fifo)
        basic_information["used_memory_size_bit"] = c(
            elem2bit(
                cost_model_output.loop.req_mem_size, common_settings.precision
            )
        )
        basic_information["actual_individual_memory_utilization"] = c(
            digit_truncate(cost_model_output.utilization.mem_utilize, 2)
        )
        basic_information["actual_memory_utilization_shared"] = c(
            digit_truncate(cost_model_output.utilization.mem_utilize_shared, 2)
        )
        basic_information["effective_memory_size"] = c(
            elem2bit(
                digit_truncate(cost_model_output.loop.effective_mem_size, 0),
                common_settings.precision,
            )
        )
        basic_information["total_unit_count"] = c(
            cost_model_output.spatial_loop.unit_count
        )
        basic_information["unique_unit_count"] = c(
            cost_model_output.spatial_loop.unit_unique
        )
        basic_information["duplicate_unit_count"] = c(
            cost_model_output.spatial_loop.unit_duplicate
        )

        # MEMORY ACCESS COUNT ELEMENT SECTION
        access_count: Dict[str, Any] = dict()
        basic_information["access_count"] = access_count

        access_count["W"] = c(mem_access_elem["W"])
        access_count["I"] = c(mem_access_elem["I"])
        access_count["O"] = c(mem_access_elem["O"])
        access_count["O_partial"] = c(mem_access_elem["O_partial"])
        access_count["O_final"] = c(mem_access_elem["O_final"])
        # END OF ACCESS COUNT SECTION

        basic_information["array_element_distance"] = c(
            cost_model_output.loop.array_wire_distance
        )
        # END OF BASIC INFORMATION SECTION

    ####################### END OF OPTIONAL OUTPUT HERE ########################

    # ENERGY SECTION
    energy: Dict[str, Any] = dict()
    results["energy"] = energy

    energy["total_energy"] = c(round(total_cost, 1))

    # ENERGY BREAKDOWN SECTION
    energy_breakdown: Dict[str, Any] = dict()
    energy["energy_breakdown"] = energy_breakdown

    # We clean the output for some reason.
    cleaned_energy = energy_clean(operand_cost)

    energy_breakdown["W"] = c(cleaned_energy["W"])
    energy_breakdown["I"] = c(cleaned_energy["I"])
    energy_breakdown["O"] = c(cleaned_energy["O"])
    # END OF ENERGY BREAKDOWN SECTION

    energy["mac_energy"] = {
        "active": c(mac_cost_active),
        "idle": c(mac_cost_idle)
    }

    # END OF ENERGY SECTION

    # PERFORMANCE SECTION
    performance: Dict[str, Any] = dict()
    results["performance"] = performance

    if verbose:
        # UTILIZATION SECTION
        utilization: Dict[str, Any] = dict()
        performance["mac_array_utilization"] = utilization

        utilization["utilization_with_data_loading"] = c(
            round(cost_model_output.utilization.mac_utilize, 4)
        )
        utilization["utilization_without_data_loading"] = c(
            round(cost_model_output.utilization.mac_utilize_no_load, 4)
        )
        utilization["utilization_spatial"] = c(
            round(cost_model_output.utilization.mac_utilize_spatial, 4)
        )
        utilization["utilization_temporal_with_data_loading"] = c(
            round(cost_model_output.utilization.mac_utilize_temporal, 4)
        )
        utilization["utilization_temporal_without_data_loading"] = c(
            round(cost_model_output.utilization.mac_utilize_temporal_no_load, 4)
        )
        # END OF UTILIZATION SECTION
    else:
        # Here the meaning of "mac_array_utilization" in the output will be
        # different, which I am not really fond of.
        performance["mac_array_utilization"] = c(
            round(cost_model_output.utilization.mac_utilize_no_load, 4)
        )

    if verbose:
        # LATENCY SECTION
        latency: Dict[str, Any] = dict()
        performance["latency"] = latency

        latency["latency_cycle_with_data_loading"] = c(
           latency_tot_number
        )
        latency["latency_cycle_without_data_loading"] = c(
            latency_no_load_number
        )
        latency["ideal_computing_cycle"] = c(
            cost_model_output.temporal_loop.total_cycles
        )

        # DATA LOADING SECTION
        data_loading: Dict[str, Any] = dict()
        latency["data_loading"] = data_loading

        data_loading["load_cycle_total"] = c(
            cc_load_tot_number
        )
        data_loading["load_cycle_individual"] = c(
            cc_load_number
        )
        data_loading["load_cycle_combined"] = c(
            cc_load_comb_number
        )
        # END OF DATA LODAING SECTION

        # MEMORY STALLING SECTION
        memory_stalling: Dict[str, Any] = dict()
        latency["memory_stalling"] = memory_stalling

        memory_stalling["memory_stalling_cycle_count"] = c(
            cc_mem_stall_tot_number
        )
        memory_stalling["memory_stalling_cycle_individual"] = c(
            stall_cc_number
        )
        memory_stalling["memory_stalling_cycle_shared"] = c(
            stall_cc_mem_share_number
        )
        memory_stalling["required_memory_bandwidth_per_cycle_individual"] = c(
            digit_truncate(cost_model_output.utilization.req_mem_bw_bit, 1)
        )
        memory_stalling["required_memory_bandwidth_per_cycle_shared"] = c(
            digit_truncate(cost_model_output.utilization.req_sh_mem_bw_bit, 1)
        )
        memory_stalling["memory_bandwidth_requirement_meet"] = c(
            mem_bw_req_meet_check(
                cost_model_output.utilization.req_sh_mem_bw_bit,
                mem_scheme.mem_bw,
            )
        )
        # END OF MEMORY STALLING SECTION
        # END OF LATENCY SECTION
    else:
        # Here also, the latency field will have a different meaning in the
        # concise output.
        performance["latency"] = c(
            cost_model_output.utilization.latency_no_load
        )
    # END OF PERFORMANCE LATENCY SECTION

    results["total_area"] = c(round(cost_model_output.area[0], 1))
    results["active_area"] = c(round(cost_model_output.area[1], 1))
    results["dark_silicon_percentage"] = c(100 - round(cost_model_output.area[1] / cost_model_output.area[0] * 100, 1))
    # END OF RESULTS SECTION

    simulation["elapsed_time_second"] = c(round(elapsed_time, 3))
    # END OF SIMULATION SECTION
    # END OF DOCUMENT

    # Creating the YAML file.
    with open(results_filename + ".yaml", "w") as yaml_file:
        # We serialize the YAML dictionary directly to the targeted file. We use
        # the default flow style because the output file will not be human
        # readable anyway.
        #
        # NOTE
        # Because YAML is a superset of json, we can write our output in the
        # json format, which is simpler to read and sufficient here, and we
        # still have a valid YAML output.
        if verbose:
            json.dump(yaml_dictionary, yaml_file)
        else:
            # If the output is meant to be concise, we try to make it human
            # readable (at least a little) by indenting it correctly.
            json.dump(yaml_dictionary, yaml_file, indent=4)

    if cost_model_output in [None, [], {}]:
        # L means layer index, M means memory scheme count, SU means spatial
        # unrolling count and cost model output is explicit.
        print(
            "L: ",
            common_settings.layer_index,
            " M: ",
            common_settings.mem_scheme_count,
            " SU: ",
            common_settings.spatial_count,
            " cost model output",
            cost_model_output,
        )
    else:
        print_good_su_format(
            cost_model_output.spatial_scheme,
            mem_scheme.mem_name,
            results_filename + ".mapping",
        )
        print_good_tm_format(
            cost_model_output.temporal_scheme,
            mem_scheme.mem_name,
            results_filename + ".mapping",
        )


def print_helper(input_settings, layers, layers_saved, multi_manager):

    # Use this for other print types (such as yaml) in the future
    # print_type = 'xml'
    print_type = input_settings.result_print_type

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
    for j, layer_index in enumerate(input_settings.layer_number):

        layer_idx_str = 'L_%d' % layer_index
        layer = layers[j]

        if save_all_arch or input_settings.mem_hierarchy_single_simulation:
            for mem_scheme_index in range(multi_manager.mem_scheme_count):

                mem_scheme_str = 'M_%d' % (mem_scheme_index + 1)
                msc = multi_manager.mem_scheme_sim[mem_scheme_index]

                su_count = list_su_count[mem_scheme_str][layer_idx_str]
                if su_count is None:
                    # su count is None for duplicate layers, get su count from parent
                    parent_str = 'L_%d' % layers[j].parent
                    su_count = list_su_count[mem_scheme_str][parent_str]

                if save_all_su:
                    # Save all the SU + best TM combinations

                    for i in range(1, su_count + 1):
                        mem_scheme_su_str = '%s_SU_%d_%d' % (mem_scheme_str, su_count, i)

                        if mem_scheme_su_str not in list_min_en_output[mem_scheme_str][layer_idx_str]['best_tm_each_su']:
                            # This means there was no temporal mapping found for this spatial unrolling, so skip
                            continue

                        best_output_energy = list_min_en_output[mem_scheme_str][layer_idx_str]['best_tm_each_su'][
                            mem_scheme_su_str]
                        tm_count_en = list_tm_count_en[mem_scheme_str][layer_idx_str]['best_tm_each_su'][
                            mem_scheme_su_str]
                        sim_time = list_sim_time[mem_scheme_str][layer_idx_str]['best_tm_each_su'][mem_scheme_su_str]

                        best_output_utilization = list_max_ut_output[mem_scheme_str][layer_idx_str]['best_tm_each_su'][
                            mem_scheme_su_str]
                        tm_count_ut = list_tm_count_ut[mem_scheme_str][layer_idx_str]['best_tm_each_su'][
                            mem_scheme_su_str]

                        mem_scheme_count_str = '%d/%d' % (mem_scheme_index + 1, multi_manager.mem_scheme_count)
                        spatial_unrolling_count = str(i) + '/' + str(su_count)
                        common_settings = CommonSetting(input_settings, layer_index, mem_scheme_count_str,
                                                        spatial_unrolling_count, msc)

                        mem_scheme_su_save_str = '_M%d_SU%d' % (mem_scheme_index + 1, i)

                        sub_path = '/all_su_best_tm/'

                        rf_en = (rf_base % sub_path) + '_L' + str(layer_index) + mem_scheme_su_save_str + rf_ending_en
                        rf_ut = (rf_base % sub_path) + '_L' + str(layer_index) + mem_scheme_su_save_str + rf_ending_ut

                        if input_settings.im2col_enable_pw and (input_settings.spatial_unrolling_mode not in [4, 5]) \
                                and (input_settings.fixed_temporal_mapping is False):
                            if multi_manager.pw_im2col_flag[j]:
                                best_output_energy.spatial_scheme, \
                                best_output_energy.flooring, \
                                best_output_energy.temporal_scheme = \
                                    pw_layer_col2im(best_output_energy.spatial_scheme,
                                                    best_output_energy.flooring,
                                                    best_output_energy.temporal_scheme, layers_saved[j].size_list)
                                best_output_utilization.spatial_scheme, \
                                best_output_utilization.flooring, \
                                best_output_utilization.temporal_scheme = \
                                    pw_layer_col2im(best_output_utilization.spatial_scheme,
                                                    best_output_utilization.flooring,
                                                    best_output_utilization.temporal_scheme, layers_saved[j].size_list)

                        if print_type == "xml":
                            print_xml(rf_en, layer, msc, best_output_energy, common_settings, tm_count_en, sim_time, input_settings.result_print_mode)
                            print_xml(rf_ut, layer, msc, best_output_utilization, common_settings, tm_count_ut, sim_time, input_settings.result_print_mode)
                        else:
                            print_yaml(rf_en, layer, msc, best_output_energy, common_settings, tm_count_en, sim_time, input_settings.result_print_mode)
                            print_yaml(rf_ut, layer, msc, best_output_utilization, common_settings, tm_count_ut, sim_time, input_settings.result_print_mode)


                # Save the best SU + TM combination
                if su_count != 0:
                    try:
                        [[mem_scheme_su_str_en, best_output_energy]] = list_min_en_output[mem_scheme_str][layer_idx_str][
                            'best_su_each_mem'].items()
                        [[mem_scheme_su_str_en, tm_count_en]] = list_tm_count_en[mem_scheme_str][layer_idx_str][
                            'best_su_each_mem'].items()
                        sim_time_en = list_sim_time[mem_scheme_str][layer_idx_str]['best_su_each_mem'][mem_scheme_su_str_en]

                        [[mem_scheme_su_str_ut, best_output_utilization]] = list_max_ut_output[mem_scheme_str][layer_idx_str][
                            'best_su_each_mem'].items()
                        [[mem_scheme_su_str_ut, tm_count_ut]] = list_tm_count_ut[mem_scheme_str][layer_idx_str][
                            'best_su_each_mem'].items()
                        sim_time_ut = list_sim_time[mem_scheme_str][layer_idx_str]['best_su_each_mem'][mem_scheme_su_str_ut]
                    except:
                        # For all the mem_schemes, there was no TM found (for this layer)
                        continue

                    mem_scheme_count_str = '%d/%d' % (mem_scheme_index + 1, multi_manager.mem_scheme_count)
                    spatial_unrolling_count_en = str(mem_scheme_su_str_en.split('_')[-1]) + '/' + str(
                        mem_scheme_su_str_en.split('_')[-2])
                    spatial_unrolling_count_ut = str(mem_scheme_su_str_ut.split('_')[-1]) + '/' + str(
                        mem_scheme_su_str_ut.split('_')[-2])
                    common_settings_en = CommonSetting(input_settings, layer_index, mem_scheme_count_str,
                                                    spatial_unrolling_count_en, msc)
                    common_settings_ut = CommonSetting(input_settings, layer_index, mem_scheme_count_str,
                                                    spatial_unrolling_count_ut, msc)

                    mem_scheme_su_save_str_en = '_M%d_SU%s' % (
                    mem_scheme_index + 1, str(mem_scheme_su_str_en.split('_')[-1]))
                    mem_scheme_su_save_str_ut = '_M%d_SU%s' % (
                    mem_scheme_index + 1, str(mem_scheme_su_str_ut.split('_')[-1]))

                    sub_path = '/best_su_best_tm/'

                    rf_en = (rf_base % sub_path) + '_L' + str(layer_index) + mem_scheme_su_save_str_en + rf_ending_en
                    rf_ut = (rf_base % sub_path) + '_L' + str(layer_index) + mem_scheme_su_save_str_ut + rf_ending_ut

                    if input_settings.im2col_enable_pw and (input_settings.spatial_unrolling_mode not in [4, 5]) \
                            and (input_settings.fixed_temporal_mapping is False):
                        if multi_manager.pw_im2col_flag[j]:
                            best_output_energy.spatial_scheme, \
                            best_output_energy.flooring, \
                            best_output_energy.temporal_scheme = \
                                pw_layer_col2im(best_output_energy.spatial_scheme,
                                                best_output_energy.flooring,
                                                best_output_energy.temporal_scheme, layers_saved[j].size_list)
                            best_output_utilization.spatial_scheme, \
                            best_output_utilization.flooring, \
                            best_output_utilization.temporal_scheme = \
                                pw_layer_col2im(best_output_utilization.spatial_scheme,
                                                best_output_utilization.flooring,
                                                best_output_utilization.temporal_scheme, layers_saved[j].size_list)

                    if print_type == "xml":
                        print_xml(rf_en, layer, msc, best_output_energy, common_settings_en, tm_count_en, sim_time_en, input_settings.result_print_mode)
                        print_xml(rf_ut, layer, msc, best_output_utilization, common_settings_ut, tm_count_ut, sim_time_ut, input_settings.result_print_mode)
                    else:
                        print_yaml(rf_en, layer, msc, best_output_energy, common_settings_en, tm_count_en, sim_time_en, input_settings.result_print_mode)
                        print_yaml(rf_ut, layer, msc, best_output_utilization, common_settings_ut, tm_count_ut, sim_time_ut, input_settings.result_print_mode)

        else:
            # Only save the best memory + su + tm combination
            [[mem_scheme_su_str_en, best_output_energy]] = list_min_en_output['best_mem_each_layer'][
                layer_idx_str].items()
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

            spatial_unrolling_count_en = str(mem_scheme_su_str_en.split('_')[-1]) + '/' + str(
                mem_scheme_su_str_en.split('_')[-2])
            spatial_unrolling_count_ut = str(mem_scheme_su_str_ut.split('_')[-1]) + '/' + str(
                mem_scheme_su_str_ut.split('_')[-2])
            common_settings_en = CommonSetting(input_settings, layer_index, mem_scheme_count_str_en,
                                               spatial_unrolling_count_en, msc_en)
            common_settings_ut = CommonSetting(input_settings, layer_index, mem_scheme_count_str_ut,
                                               spatial_unrolling_count_ut, msc_ut)

            mem_scheme_su_save_str_en = '_M%d_SU%s' % (
            mem_scheme_index_en + 1, str(mem_scheme_su_str_en.split('_')[-1]))
            mem_scheme_su_save_str_ut = '_M%d_SU%s' % (
            mem_scheme_index_ut + 1, str(mem_scheme_su_str_ut.split('_')[-1]))

            sub_path = '/best_mem_each_layer/'

            rf_en = (rf_base % sub_path) + '_L' + str(layer_index) + mem_scheme_su_save_str_en + rf_ending_en
            rf_ut = (rf_base % sub_path) + '_L' + str(layer_index) + mem_scheme_su_save_str_ut + rf_ending_ut

            if input_settings.im2col_enable_pw and (input_settings.spatial_unrolling_mode not in [4, 5]) \
                    and (input_settings.fixed_temporal_mapping is False):
                if multi_manager.pw_im2col_flag[j]:
                    best_output_energy.spatial_scheme, \
                    best_output_energy.flooring, \
                    best_output_energy.temporal_scheme = \
                        pw_layer_col2im(best_output_energy.spatial_scheme,
                                        best_output_energy.flooring,
                                        best_output_energy.temporal_scheme, layers_saved[j].size_list)
                    best_output_utilization.spatial_scheme, \
                    best_output_utilization.flooring, \
                    best_output_utilization.temporal_scheme = \
                        pw_layer_col2im(best_output_utilization.spatial_scheme,
                                        best_output_utilization.flooring,
                                        best_output_utilization.temporal_scheme, layers_saved[j].size_list)

            if print_type == "xml":
                print_xml(rf_en, layer, msc_en, best_output_energy, common_settings_en, tm_count_en, sim_time_en, input_settings.result_print_mode)
                print_xml(rf_ut, layer, msc_ut, best_output_utilization, common_settings_ut, tm_count_ut, sim_time_ut, input_settings.result_print_mode)
            else:
                print_yaml(rf_en, layer, msc_en, best_output_energy, common_settings_en, tm_count_en, sim_time_en, input_settings.result_print_mode)
                print_yaml(rf_ut, layer, msc_ut, best_output_utilization, common_settings_ut, tm_count_ut, sim_time_ut, input_settings.result_print_mode)
    
        if (len(input_settings.layer_number) > 1) and (multi_manager.mem_scheme_count > 1):
            # Save the best memory hierarchy for all layers in separate folder

            best_mem_scheme_idx_en = multi_manager.best_mem_scheme_index_en
            best_mem_scheme_idx_ut = multi_manager.best_mem_scheme_index_ut
            best_mem_scheme_str_en = 'M_%d' % (best_mem_scheme_idx_en + 1)
            best_mem_scheme_str_ut = 'M_%d' % (best_mem_scheme_idx_ut + 1)

            [[mem_scheme_su_str_en, best_output_energy]] = list_min_en_output[best_mem_scheme_str_en][layer_idx_str][
                'best_su_each_mem'].items()
            [[mem_scheme_su_str_en, tm_count_en]] = list_tm_count_en[best_mem_scheme_str_en][layer_idx_str][
                'best_su_each_mem'].items()
            sim_time_en = list_sim_time[best_mem_scheme_str_en][layer_idx_str]['best_su_each_mem'][mem_scheme_su_str_en]

            [[mem_scheme_su_str_ut, best_output_utilization]] = \
            list_max_ut_output[best_mem_scheme_str_ut][layer_idx_str]['best_su_each_mem'].items()
            [[mem_scheme_su_str_ut, tm_count_ut]] = list_tm_count_ut[best_mem_scheme_str_ut][layer_idx_str][
                'best_su_each_mem'].items()
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

            if input_settings.im2col_enable_pw and (input_settings.spatial_unrolling_mode not in [4, 5]) \
                    and (input_settings.fixed_temporal_mapping is False):
                if multi_manager.pw_im2col_flag[j]:
                    best_output_energy.spatial_scheme, \
                    best_output_energy.flooring, \
                    best_output_energy.temporal_scheme = \
                        pw_layer_col2im(best_output_energy.spatial_scheme,
                                        best_output_energy.flooring,
                                        best_output_energy.temporal_scheme, layers_saved[j].size_list)
                    best_output_utilization.spatial_scheme, \
                    best_output_utilization.flooring, \
                    best_output_utilization.temporal_scheme = \
                        pw_layer_col2im(best_output_utilization.spatial_scheme,
                                        best_output_utilization.flooring,
                                        best_output_utilization.temporal_scheme, layers_saved[j].size_list)

            if print_type == "xml":
                print_xml(rf_en, layer, msc_en, best_output_energy, common_settings_en, tm_count_en, sim_time_en, input_settings.result_print_mode)
                print_xml(rf_ut, layer, msc_ut, best_output_utilization, common_settings_ut, tm_count_ut, sim_time_ut, input_settings.result_print_mode)
            else:
                print_yaml(rf_en, layer, msc_en, best_output_energy, common_settings_en, tm_count_en, sim_time_en, input_settings.result_print_mode)
                print_yaml(rf_ut, layer, msc_ut, best_output_utilization, common_settings_ut, tm_count_ut, sim_time_ut, input_settings.result_print_mode)


class CostModelOutput:

    def __init__(self, total_cost, operand_cost, mac_cost, temporal_scheme, spatial_scheme, flooring, loop,
                 spatial_loop, greedy_mapping_flag, footer_info, temporal_loop, area, utilization, shared_bw):
        self.operand_cost = operand_cost
        self.total_cost = total_cost
        self.temporal_scheme = temporal_scheme
        self.spatial_scheme = spatial_scheme
        self.flooring = flooring
        self.loop = loop
        self.mac_cost = mac_cost
        self.spatial_loop = spatial_loop
        self.greedy_mapping_flag = greedy_mapping_flag
        self.footer_info = footer_info
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
            else 'hint-driven search' if input_settings.spatial_unrolling_mode == 3
            else 'greedy mapping with hint' if input_settings.spatial_unrolling_mode == 4
            else 'greedy mapping without hint',

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
        self.im2col_enable = input_settings.im2col_enable_all

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

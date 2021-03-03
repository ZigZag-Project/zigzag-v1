import os
import xml.etree.cElementTree as ET
import ast
from PyQt5 import QtCore
Qt = QtCore.Qt

def xml_info_extraction():
    file_path_and_name = 'PATH/TO/YOUR/RESULTS/***.xml'
    if os.path.isfile(file_path_and_name):
        mem_size_collect = []
        area_collect = []
        utilize_collect = []
        latency_collect = []
        energy_collect = []
        tree = ET.parse(file_path_and_name)
        root = tree.getroot()
        for sim in root:
            mem_size = ast.literal_eval(sim.find("hw_spec/memory_hierarchy/mem_size_bit").tail)
            area = ast.literal_eval(sim.find("results/area").tail)
            utilize = ast.literal_eval(sim.find("results/performance/mac_array_utilization/utilization_with_data_loading").tail)
            latency = ast.literal_eval(sim.find("results/performance/latency/latency_cycle_with_data_loading").tail)
            energy = ast.literal_eval(sim.find("results/energy/total_energy").tail)
            mem_size_collect.append(mem_size)
            area_collect.append(area)
            utilize_collect.append(utilize)
            latency_collect.append(latency)
            energy_collect.append(energy)
            break
    return mem_size_collect, area_collect, utilize_collect, latency_collect, energy_collect
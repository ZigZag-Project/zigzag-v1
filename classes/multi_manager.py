from multiprocessing import Manager, Value
import sys, time

class Multimanager(object):

    def __init__(self, mem_scheme_sim, layer_spec):

        self.start_time = time.time()

        self.mem_scheme_sim = mem_scheme_sim
        self.mem_scheme_count = len(mem_scheme_sim)

        self.min_cost_mem_scheme = sys.float_info.max
        self.min_cost_mem_scheme_index = None
        self.list_min_cost_mem_scheme = [self.min_cost_mem_scheme for i in range(self.mem_scheme_count)]
        self.list_min_output_mem_scheme = [self.min_cost_mem_scheme for i in range(self.mem_scheme_count)]

        self.layer_spec = layer_spec



        ''' Multiprocessing. Variables used for collecting output. '''
        manager = Manager()

        self.list_tm_count_en = manager.dict()
        self.list_tm_count_ut = manager.dict()
        self.list_min_energy = manager.dict()
        self.list_min_en_output = manager.dict()
        self.list_max_utilization = manager.dict()
        self.list_max_ut_output = manager.dict()
        self.list_sim_time = manager.dict()

        self.list_tm_count_en['best_mem_each_layer'] = []
        self.list_tm_count_ut['best_mem_each_layer'] = []
        self.list_min_energy['best_mem_each_layer'] = []
        self.list_min_en_output['best_mem_each_layer'] = []
        self.list_max_utilization['best_mem_each_layer'] = []
        self.list_max_ut_output['best_mem_each_layer'] = []
        self.list_sim_time['best_mem_each_layer'] = []

        for jj in range(self.mem_scheme_count):
            self.list_tm_count_en['M_%d' % (jj + 1)] = {'best_su_each_mem': [], 'best_tm_each_su': []}
            self.list_tm_count_ut['M_%d' % (jj + 1)] = {'best_su_each_mem': [], 'best_tm_each_su': []}
            self.list_min_energy['M_%d' % (jj + 1)] = {'best_su_each_mem': [], 'best_tm_each_su': []}
            self.list_min_en_output['M_%d' % (jj + 1)] = {'best_su_each_mem': [], 'best_tm_each_su': []}
            self.list_max_utilization['M_%d' % (jj + 1)] = {'best_su_each_mem': [], 'best_tm_each_su': []}
            self.list_max_ut_output['M_%d' % (jj + 1)] = {'best_su_each_mem': [], 'best_tm_each_su': []}
            self.list_sim_time['M_%d' % (jj + 1)] = {'best_su_each_mem': [], 'best_tm_each_su': []}


        self.new_best_mem_scheme_index = Value('i', 0)
        self.min_cost_mem_scheme = Value('d', float('inf'))


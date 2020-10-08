from multiprocessing import Manager, Value
import sys, time

class Multimanager(object):

    def __init__(self, input_settings, mem_scheme_sim, layer_spec):

        self.start_time = time.time()

        self.mem_scheme_sim = mem_scheme_sim
        self.mem_scheme_count = len(mem_scheme_sim)

        self.min_cost_mem_scheme = sys.float_info.max
        self.min_cost_mem_scheme_index = None
        self.list_min_cost_mem_scheme = [self.min_cost_mem_scheme for i in range(self.mem_scheme_count)]
        self.list_min_output_mem_scheme = [self.min_cost_mem_scheme for i in range(self.mem_scheme_count)]

        self.layer_spec = layer_spec

        self.best_mem_scheme_index_en = None
        self.best_mem_scheme_index_ut = None

        ''' Multiprocessing. Variables used for collecting output. '''
        manager = Manager()

        self.list_tm_count_en = manager.dict()
        self.list_tm_count_ut = manager.dict()
        self.list_min_energy = manager.dict()
        self.list_min_en_output = manager.dict()
        self.list_max_utilization = manager.dict()
        self.list_max_ut_output = manager.dict()
        self.list_sim_time = manager.dict()
        self.list_su_count = manager.dict()
        self.list_sim_time_en = manager.dict()
        self.list_sim_time_ut = manager.dict()

        self.list_tm_count_en['best_mem_each_layer'] = {}
        self.list_tm_count_ut['best_mem_each_layer'] = {}
        self.list_min_energy['best_mem_each_layer'] = {}
        self.list_min_en_output['best_mem_each_layer'] = {}
        self.list_max_utilization['best_mem_each_layer'] = {}
        self.list_max_ut_output['best_mem_each_layer'] = {}
        self.list_sim_time_en['best_mem_each_layer'] = {}
        self.list_sim_time_ut['best_mem_each_layer'] = {}

        for jj in range(self.mem_scheme_count):

            mem_str = 'M_%d' % (jj + 1)

            a = self.list_tm_count_en[mem_str] = {}
            b = self.list_tm_count_ut[mem_str] = {}
            c = self.list_min_energy[mem_str] = {}
            d = self.list_min_en_output[mem_str] = {}
            e = self.list_max_utilization[mem_str] = {}
            f = self.list_max_ut_output[mem_str] = {}
            g = self.list_sim_time[mem_str] = {}
            h = self.list_su_count[mem_str] = {}

            for kk in input_settings.layer_number:

                layer_str = 'L_%d' % kk

                a[layer_str] = {'best_su_each_mem': {}, 'best_tm_each_su': {}}
                b[layer_str] = {'best_su_each_mem': {}, 'best_tm_each_su': {}}
                c[layer_str] = {'best_su_each_mem': {}, 'best_tm_each_su': {}}
                d[layer_str] = {'best_su_each_mem': {}, 'best_tm_each_su': {}}
                e[layer_str] = {'best_su_each_mem': {}, 'best_tm_each_su': {}}
                f[layer_str] = {'best_su_each_mem': {}, 'best_tm_each_su': {}}
                g[layer_str] = {'best_su_each_mem': {}, 'best_tm_each_su': {}}

            self.list_tm_count_en[mem_str] = a
            self.list_tm_count_ut[mem_str] = b
            self.list_min_energy[mem_str] = c
            self.list_min_en_output[mem_str] = d
            self.list_max_utilization[mem_str] = e
            self.list_max_ut_output[mem_str] = f
            self.list_sim_time[mem_str] = g
            self.list_su_count[mem_str] = h

        self.new_best_mem_scheme_index = Value('i', 0)
        self.min_cost_mem_scheme = Value('d', float('inf'))


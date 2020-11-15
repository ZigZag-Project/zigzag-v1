from multiprocessing import Manager, Value
import sys, time

class MultiManager(object):

    def __init__(self, input_settings, mem_scheme_sim, layer_spec, layers, layer_info_im2col, layers_im2col,
                 pw_im2col_flag):

        self.start_time = time.time()

        self.mem_scheme_sim = mem_scheme_sim
        self.mem_scheme_count = len(mem_scheme_sim)

        self.layer_spec = layer_spec
        self.layer_info_im2col = layer_info_im2col
        self.layers_im2col = layers_im2col
        self.pw_im2col_flag = pw_im2col_flag
        self.greedy_mapping_flag = []
        self.footer_info = []

        self.best_mem_scheme_index_en = None
        self.best_mem_scheme_index_ut = None

        ''' Multiprocessing. Variables used for collecting output. '''
        manager = Manager()

        # Create the nested dictionary structure.
        # Use manager.dict() object instead of regular dict
        # to avoid synchronization issues across multiprocessing

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

        self.list_tm_count_en['best_mem_each_layer'] = manager.dict()
        self.list_tm_count_ut['best_mem_each_layer'] = manager.dict()
        self.list_min_energy['best_mem_each_layer'] = manager.dict()
        self.list_min_en_output['best_mem_each_layer'] = manager.dict()
        self.list_max_utilization['best_mem_each_layer'] = manager.dict()
        self.list_max_ut_output['best_mem_each_layer'] = manager.dict()
        self.list_sim_time_en['best_mem_each_layer'] = manager.dict()
        self.list_sim_time_ut['best_mem_each_layer'] = manager.dict()

        for i in range(self.mem_scheme_count):

            mem_str = 'M_%d' % (i + 1)

            self.list_tm_count_en[mem_str] = manager.dict()
            self.list_tm_count_ut[mem_str] = manager.dict()
            self.list_min_energy[mem_str] = manager.dict()
            self.list_min_en_output[mem_str] = manager.dict()
            self.list_max_utilization[mem_str] = manager.dict()
            self.list_max_ut_output[mem_str] = manager.dict()
            self.list_sim_time[mem_str] = manager.dict()

            self.list_su_count[mem_str] = manager.dict()

            for layer_idx, layer_number in enumerate(input_settings.layer_number):

                layer_str = 'L_%d' % layer_number

                # If the layer is a duplicate, track variables of parent layer
                if layers[layer_idx].is_duplicate:
                    parent_number = layers[layer_idx].parent
                    parent_str = 'L_%d' % parent_number

                    self.list_tm_count_en[mem_str][layer_str] = self.list_tm_count_en[mem_str][parent_str]
                    self.list_tm_count_ut[mem_str][layer_str] = self.list_tm_count_ut[mem_str][parent_str]
                    self.list_min_energy[mem_str][layer_str] = self.list_min_energy[mem_str][parent_str]
                    self.list_min_en_output[mem_str][layer_str] = self.list_min_en_output[mem_str][parent_str]
                    self.list_max_utilization[mem_str][layer_str] = self.list_max_utilization[mem_str][parent_str]
                    self.list_max_ut_output[mem_str][layer_str] = self.list_max_ut_output[mem_str][parent_str]
                    self.list_sim_time[mem_str][layer_str] = self.list_sim_time[mem_str][parent_str]
                    self.list_su_count[mem_str][layer_str] = self.list_su_count[mem_str][parent_str]
                else:
                    self.list_tm_count_en[mem_str][layer_str] = manager.dict(
                        [('best_su_each_mem', manager.dict()), 
                        ('best_tm_each_su', manager.dict())]
                    )
                    self.list_tm_count_ut[mem_str][layer_str] = manager.dict(
                        [('best_su_each_mem', manager.dict()), 
                        ('best_tm_each_su', manager.dict())]
                    )
                    self.list_min_energy[mem_str][layer_str] = manager.dict(
                        [('best_su_each_mem', manager.dict()), 
                        ('best_tm_each_su', manager.dict())]
                    )
                    self.list_min_en_output[mem_str][layer_str] = manager.dict(
                        [('best_su_each_mem', manager.dict()), 
                        ('best_tm_each_su', manager.dict())]
                    )
                    self.list_max_utilization[mem_str][layer_str] = manager.dict(
                        [('best_su_each_mem', manager.dict()), 
                        ('best_tm_each_su', manager.dict())]
                    )
                    self.list_max_ut_output[mem_str][layer_str] = manager.dict(
                        [('best_su_each_mem', manager.dict()), 
                        ('best_tm_each_su', manager.dict())]
                    )
                    self.list_sim_time[mem_str][layer_str] = manager.dict(
                        [('best_su_each_mem', manager.dict()), 
                        ('best_tm_each_su', manager.dict())]
                    )
                    self.list_su_count[mem_str][layer_str] = None



"""
Layer size rounding regarding array dimension, to support greedy mapping.
"""
import copy


class LayerRound(object):

    def __init__(self, layer_spec_raw, array_size, unrolling_scheme_list):
        ll = {1: 'FX', 2: 'FY', 3: 'OX', 4: 'OY', 5: 'C', 6: 'K', 7: 'B'}
        roun_layer_info = []
        footer_info = []
        fraction_spatial_unrolling = []
        for i in range(len(unrolling_scheme_list)):
            roun_layer_info.append(copy.deepcopy(layer_spec_raw))
            su_list_flatten = [item for sublist in unrolling_scheme_list[i] for item in sublist]
            footer_info.append({'B': 0, 'K': 0, 'C': 0, 'OY': 0, 'OX': 0, 'FY': 0, 'FX': 0})
            fraction_spatial_unrolling.append([])
            for su_dim, su_type in enumerate(su_list_flatten):
                footer = layer_spec_raw[ll[su_type]] % array_size[su_dim]

                if footer != 0 and layer_spec_raw[ll[su_type]] > array_size[su_dim]:
                    roun_layer_info[-1][ll[su_type]] += array_size[su_dim] - footer

                footer_info[-1][ll[su_type]] = footer
                mapping_time = max(1,roun_layer_info[-1][ll[su_type]] // array_size[su_dim])
                fraction = (array_size[su_dim] * (mapping_time - 1) + footer) / mapping_time
                fraction_spatial_unrolling[-1].append([su_type, fraction])

        self.roun_layer_info = roun_layer_info
        self.footer_info = footer_info
        self.fraction_su = fraction_spatial_unrolling

    @classmethod
    def layer_rounding(cls, layer_spec_raw, array_size, unrolling_scheme_list):
        return cls(layer_spec_raw, array_size, unrolling_scheme_list)

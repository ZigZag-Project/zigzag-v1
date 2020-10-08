"""
Layer size rounding regarding array dimension.
"""
import copy


class LayerRound(object):

    def __init__(self, layer_spec_raw, array_size, unrolling_scheme_list):
        ll = {1: 'FX', 2: 'FY', 3: 'OX', 4: 'OY', 5: 'C', 6: 'K', 7: 'B'}
        new_layer_info = copy.deepcopy(layer_spec_raw.layer_info)
        footer_info = {}
        su_list_flatten = [item for sublist in unrolling_scheme_list[0] for item in sublist]
        for idx, layer in layer_spec_raw.layer_info.items():
            footer_info[idx] = {'B': 0, 'K': 0, 'C': 0, 'OY': 0, 'OX': 0, 'FY': 0, 'FX': 0}
            for su_dim, su_type in enumerate(su_list_flatten):
                footer = layer[ll[su_type]]%array_size[su_dim]
                footer_info[idx][ll[su_type]] = footer
                if footer != 0:
                    new_layer_info[idx][ll[su_type]] += array_size[su_dim] - footer

        self.layer_info = new_layer_info
        self.footer_info = footer_info

    @classmethod
    def layer_rounding(cls, layer_spec_raw, array_size, unrolling_scheme_list):
        return cls(layer_spec_raw, array_size, unrolling_scheme_list)

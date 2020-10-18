import copy


def im2col_layer_transform(layer_info):
    im2col_layer_info = {}
    for layer_index, layer in layer_info.items():

        # TODO support stride under im2col mode
        im2col_layer_info[layer_index] = {'B': 1, 'K': 1, 'C': 1, 'OY': 1, 'OX': 1, 'FY': 1, 'FX': 1, 'SY': 1, 'SX': 1,
                                          'SFY': 1, 'SFX': 1, 'PY': 0, 'PX': 0}
        im2col_layer_info[layer_index]['B'] = layer['B']*layer['OY']*layer['OX']
        im2col_layer_info[layer_index]['K'] = layer['K']
        im2col_layer_info[layer_index]['C'] = layer['C']*layer['FY']*layer['FX']

    return im2col_layer_info

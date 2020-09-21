"""
Layer specification.
"""


class Layer(object):
    """
    NN layer parameters.

 (7)B:   Batch size
 (6)K:   Filter kernels / Output channels
 (5)C:   Filter channels / Input channels
 (4)OY:  Y dimension of output feature map
 (3)OX:  X dimension of output feature map
 (2)FY:  Y dimension of filters
 (1)FX:  X dimension of filters
    SY:  Stride on input feature map, Y dimension
    SX:  Stride on input feature map, X dimension
    SFY: Stride on filters, Y dimension
    SFX: Stride on filters, X dimension
    PY:  Padding on input feature map, Y dimension
    PX:  Padding on input feature map, X dimension

    """

    def __init__(self, B, K, C, OY, OX, FY, FX, SY=1, SX=1, SFY=1, SFX=1, PY=0, PX=0):
        self.B = B
        self.K = K
        self.C = C
        self.OY = OY
        self.OX = OX
        self.IY = SY * (OY - 1) + SFY * (FY - 1) + 1
        self.IX = SX * (OX - 1) + SFX * (FX - 1) + 1
        self.FY = FY
        self.FX = FX
        self.SY = SY
        self.SX = SX
        self.SFY = SFY
        self.SFX = SFX
        self.PY = PY
        self.PX = PX
        self.total_MAC_op = B * K * C * OY * OX * FY * FX

        self.total_data_size = {'W': K * C * FY * FX,
                                'I': B * C * self.IY * self.IX,
                                'O': B * K * OY * OX}

        '''
        total_data_reuse: the total data reuse possibility for each element in W/I/O.

        Note that for 'I', each element can has different maximum data reuse possibility, 
        thus it is represented by the average value, i.e. total fetch / total input element. 
        '''

        self.total_data_reuse = {'W': B * OY * OX,
                                 'I': self.total_MAC_op / self.total_data_size['I'],
                                 'O': C * FY * FX}

        self.size_list = [[SY, SX, SFY, SFX, PY, PX], FX, FY, OX, OY, C, K, B]
        self.size_list_output_print = {'B': B,
                                       'K': K,
                                       'C': C,
                                       'OY': OY,
                                       'OX': OX,
                                       'IY': self.IY,
                                       'IX': self.IX,
                                       'FY': FY,
                                       'FX': FX,
                                       'SY': SY,
                                       'SX': SX,
                                       'SFY': SFY,
                                       'SFX': SFX}

    @classmethod
    def extract_layer_info(cls, info):
        return cls(info["B"],
                   info["K"], info["C"],
                   info["OY"], info["OX"],
                   info["FY"], info["FX"])

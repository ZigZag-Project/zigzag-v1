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
    G:   Number of groups for grouped convolution

    For multiple groups, the K and C of this object is for one group.
    When printing to the output file, it will be scaled up accordingly.

    """

    def __init__(self, B, K, C, OY, OX, FY, FX, SY=1, SX=1, SFY=1, SFX=1, PY=0, PX=0, G=1):
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
        self.G = G

        # Use individual group K and C (in case of G != 1)
        self.total_MAC_op = B * K * C * OY * OX * FY * FX

        # Use provided (total) K and C in case of G != 1
        self.total_data_size = {'W': K * C * FY * FX,
                                'I': B * C * self.IY * self.IX,
                                'O': B * K * OY * OX}

        '''
        total_data_reuse: the total data reuse possibility for each element in W/I/O.

        Note that for 'I', each element can have different maximum data reuse possibility, 
        thus it is represented by the average value, i.e. total fetch / total input element. 
        '''

        # Not entirely sure if this should be changed according to group size bc total values
        self.total_data_reuse = {'W': B * OY * OX,
                                 'I': self.total_MAC_op / self.total_data_size['I'],
                                 'O': C * FY * FX}

        self.size_list = [[SY, SX, SFY, SFX, PY, PX, G], FX, FY, OX, OY, C, K, B]

        # Only used for printing to xml, so keep as total dimensions (for grouped convolution)
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
                                       'SFX': SFX,
                                       'G': G}

        # Initialize is_duplicate bool to False
        self.is_duplicate = False
        # First occurring identical layer will be refered to as parent
        self.parent = None

    def __eq__(self, other):
        "Override the object-based equality operator"

        if isinstance(other, Layer):
            return self.size_list == other.size_list
        return NotImplemented

    def set_duplicate(self, other_layer_number):
        """
        Set the layer as a duplicate layer.
        Set the layer's 'parent', i.e. the earliest identical layer number
        """
        self.is_duplicate = True
        self.parent = other_layer_number

    @classmethod
    def extract_layer_info(cls, info):
        return cls(info["B"],
                   info["K"], info["C"],
                   info["OY"], info["OX"],
                   info["FY"], info["FX"],
                   info["SY"], info["SX"],
                   info["SFY"], info["SFX"],
                   info["PY"], info["PX"],
                   G=info.get("G", 1))

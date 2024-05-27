from regex import I
import torch
import torch.nn as nn



CLIPMIN = 1e-5



def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x



class MixUniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        per_channel_axes=[],
        metric="minmax",
        dynamic=False,
        dynamic_method="per_channel",
        group_size=None,
        shape=None,
        lwc=False,
        disable_zero_point=False,
        block_precision=None,
    ):
        """
        support cluster quantize
        dynamic_method support per_token and per_cluster
        """
        super().__init__()
        self.symmetric = symmetric
        self.disable_zero_point = disable_zero_point
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        self.block_precision = block_precision  

        self.qmins = [-1, 0, 0, 0, 0, 0, 0, 0]
        self.qmaxs = [1, 3, 7, 15, 31, 63, 127, 255]

        # tmp_list = self.block_precision
        # self.block_precision  = [x + (self.n_bits-2) for x in tmp_list]
        # self.block_precision = [x + (self.n_bits-2) for x in tmp_list]
        print(self.block_precision)
        self.per_channel_axes = per_channel_axes
        self.metric = metric
        self.cluster_counts = None
        self.cluster_dim = None

        self.scale = None
        self.zero_point = None
        self.round_zero_point = None

        self.cached_xmin = None
        self.cached_xmax = None
        self.dynamic = dynamic
        self.dynamic_method = dynamic_method
        self.deficiency = 0
        self.lwc = lwc
        
        init_value = 4.             # inti value of learnable weight clipping
        if lwc:
            self.upbound_factor = nn.Parameter(torch.ones((shape[0],len(block_precision)))*init_value)
            self.lowbound_factor = nn.Parameter(torch.ones((shape[0],len(block_precision)))*init_value)
        self.sigmoid = nn.Sigmoid()

        self.enable = True
        self.group_size = group_size

    def fake_quant(self, x, scale, round_zero_point):
        _, dim2 = x.shape

        x_dequant = torch.zeros_like(x)
        for blocki, col_st in enumerate(range(0, dim2, self.group_size)):
            col_ed = min(col_st + self.group_size, dim2)
            st = col_st
            ed = col_ed
            block_precision = self.block_precision[blocki]
  
            weight_block = x[:,st:ed]  
            scale_block = scale[:, blocki].unsqueeze(1) 
            zero_block = round_zero_point[:, blocki].unsqueeze(1) if round_zero_point is not None else None
            q_min = self.qmins[block_precision-1]
            q_max = self.qmaxs[block_precision-1]
            
            if block_precision == 1:
                weight_block = weight_block.sub(zero_block)
                x_int = torch.sign(weight_block / scale_block)
            else:
                x_int = round_ste(weight_block / scale_block)
                x_int = x_int.add(zero_block)
            x_int = x_int.clamp(q_min, q_max)


            if block_precision != 1:
                x_int = x_int.sub(zero_block)
                x_dequant[:, st:ed] = x_int.mul(scale_block)
            else: 
                x_dequant[:, st:ed] = x_int.mul(scale_block)
                x_int = x_int.add(zero_block)

        return x_dequant
    

    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16 or not self.enable:
            return x
        if self.dynamic_method == "per_token" or self.dynamic_method == "per_channel":
            self.per_token_dynamic_calibration(x)
        else:
            raise NotImplementedError()   

        x_dequant = self.fake_quant(x, self.mixed_scales, self.round_zero_points)
        return x_dequant

    def per_token_dynamic_calibration(self, x):

        self.mixed_scales = torch.zeros_like(self.upbound_factor)
        self.round_zero_points = torch.zeros_like(self.upbound_factor)
        
        for blocki, col_st in enumerate(range(0,  x.shape[1], self.group_size)):
            col_ed = min(col_st + self.group_size, x.shape[1])
            st = col_st
            ed = col_ed
            weight_block = x[:,st:ed]
            x_min_block = weight_block.amin(-1, keepdim=True)
            x_max_block = weight_block.amax(-1, keepdim=True)
            block_precision = self.block_precision[blocki]
            
            if self.lwc:
                xmax = self.sigmoid(self.upbound_factor[:, blocki])*x_max_block[:,0]
                xmin = self.sigmoid(self.lowbound_factor[:, blocki])*x_min_block[:,0]

            if self.symmetric:
                abs_max = torch.max(xmax.abs(),xmin.abs())
                scale = abs_max / (2**(block_precision-1)-1)
                scale = scale.clamp(min=CLIPMIN, max=1e4)
                zero_point = (2**(block_precision-1)-1)*torch.ones_like(scale)
            else:
                range_ = xmax - xmin
                if block_precision == 1:
                    abs_mean = torch.mean(torch.abs(weight_block), dim=1)
                    scale = abs_mean * (self.sigmoid(self.upbound_factor[:, blocki]) + 0.5)                                
                    zero_point = torch.mean(weight_block, dim=1)
                  

                else:
                    scale = range_ / (2**block_precision-1)
                    zero_point = -(xmin) / (scale)

                scale = scale.clamp(min=CLIPMIN, max=1e4)
            if self.disable_zero_point:
                round_zero_point = None
            else:
                round_zero_point = zero_point.clamp(min=-1e4, max=1e4)
                if block_precision > 1:
                    round_zero_point = round_zero_point.round()
            self.mixed_scales[:, blocki] = scale
            self.round_zero_points[:, blocki] = round_zero_point


    def register_scales_and_zeros(self):
        self.register_buffer('scales', self.mixed_scales)
        self.register_buffer('zeros', self.round_zero_points)
        del self.mixed_scales
        del self.round_zero_points

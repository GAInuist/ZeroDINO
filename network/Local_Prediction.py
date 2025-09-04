from .Visual_Guided_Encoder import *
from .Semantic_Guided_Decoder import *
from .utils import *



class DINOMulti(nn.Module):
    def __init__(
        self,
        attr_num: int = 312
    ):
        super().__init__()
        
        self.attr_num = attr_num
        self.dim = 768
        self.VGE = VGE()
        self.SGD = SGD(self.attr_num, self.dim)
        self.fine_predictor = nn.Linear(self.dim, self.attr_num)
        self.x_fine_predictor = nn.Linear(self.dim, self.attr_num)
        self.x_macro_predictor = nn.Linear(self.dim, self.attr_num)


    def forward(self, x, w2v):
        x_macro, x_fine, x_global, pool_fine, pool_macro = self.VGE(x)
        x_attr, x_attr_fine, x_attr_macro, fine_weights, macro_weights, alpha, beta = self.SGD(x_macro, x_fine, w2v)
        
        fine_result = torch.einsum('bae, ae -> ba', x_attr, self.fine_predictor.weight)
        x_fine_result = torch.einsum('bae, ae -> ba', x_attr_fine, self.x_fine_predictor.weight)
        x_macro_result = torch.einsum('bae, ae -> ba', x_attr_macro, self.x_macro_predictor.weight)


        output = {
            'pool_fine' : pool_fine,
            'pool_macro' : pool_macro,
            'fine_result' : fine_result,
            'x_fine_result' : x_fine_result,
            'x_macro_result' : x_macro_result,
            'fine_weights' : fine_weights,
            'macro_weights' : macro_weights,
            'alpha' : alpha,
            'beta' : beta,
        }
        return output
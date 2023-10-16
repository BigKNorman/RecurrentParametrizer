import torch.nn as nn

class TransformerParametrizer(nn.Module): 
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(TransformerParametrizer, self).__init__()


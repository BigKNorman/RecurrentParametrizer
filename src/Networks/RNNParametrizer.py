import torch.nn as nn

class RNNParametrizer(nn.Module): 
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.layers = []

        self.num_layers = num_layers 
            
        inp_lay = input_size + hidden_size
        out_lay = hidden_size
        for i in range(num_layers):
            self.layers.append(nn.Linear(inp_lay, out_lay))
            inp_lay = input_size + hidden_size
            if i == num_layers - 2:
                inp_lay = hidden_size
                out_lay = output_size
            else:
                out_lay = hidden_size
                

    def forward(self, input, hidden): 
        combined = hidden
        for i in range(self.num_layers): 
            layer = self.layers[i]
            if i == self.num_layers - 1: 
                output = layer(combined)
                output = self.softmax(output)
            else: 
                combined = torch.cat((input, combined), 1)
                combined = layer(combined)

        return output, combined

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)



# RecurrentParametrizer
Use RNN-like structure (Transformer could work here too) to parametrize a potentially infinite-width one-layer neural network



Actually, further idea: using a network to parametrize another network. What should be the structure of the target network? 

This idea adds another layer between data and prediction, which is interesting. That reminds me of correlation vs. causation, because neural networks are correlational machines generally speaking.

Since we're adding a layer anyway, what if it was causal? What if this approach to stuff was intended to bridge the jump between neural network correlation and causation? Maybe it could populate the parameters of a graph neural net, or a causal graph. 

But if I was going to use a causal graph, how would I decide the structure? I feel like the whole point of this was that having a one-layer neural net is the simplest structure, and adding width has to be able to structurally encode information in some manner, right? Otherwise, the RNN or other generative model would just keep going, just keep predicting parameters and extending the neural network horizontally. 

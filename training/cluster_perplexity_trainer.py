"""
The trainer class for training the clustering phase.
goal is to just call trainer.train() to perform training.

The trainer should be managing:
- config (lr, schedule, hyperparam)
- model and tokenizer, whose name should be included in the config, and instantiate here.
- optimizer.
- instantiate an agent cluster.
- get the loss and backprop logic. 
- using callback and logging to enhance modularity and logic.
"""

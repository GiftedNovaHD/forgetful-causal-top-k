# Forgetful Causal top-$k$ Attention Implementation


## Features
- Top-$k$ attention implementation - for global attention: 
  - For each query, mask out all but its $k$ largest dot product with the keys, so that in each row of $QK^{\intercal}$, we only keep the $k$ largest values and mask out the rest.
- Compatible with Approximate Nearest Neighbour (ANN) algorithms 
  - Uses forgetful masking at training time to account for imperfect recall of ANN algorithms
- Use Deepseek V2/3 Multi-Latent Attention (MLA)
  - Underlying ANN algorithm at inference time is LoRANN
  - We store the latent KV vectors in LoRANN
- Dynamically select $k$ to reduce approximation error below a certain bound
- Increase dimension of $Q$ vectors as done in Sigma
- Select which Key-Value expert pairs to use (maybe also have query experts)
- Integrate persistent tokens as learnable attention sinks and sliding window attention (with special kernels)
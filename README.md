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

## To-Do 
- [ ] Forgetful Top-$k$ Attention Mechanism 
  - [ ] Static top-$k$ Selection  
       *(Not yet fully implemented)*
  - [ ] Dynamic top-$k$ Selection for training 
  - [x] MLA Forward Pass 
  - [x] MLA Backward Pass
- [ ] Sliding Window Attention 
- [ ] Persistent Tokens
- [ ] Replace the exact top-$k$ attention with an ANN approximation like LoRANN during inference
- [ ] Add dynamic $k$-selection logic based on approximation error bounds
- [ ] Implement expert selection using router networks and multiple KV projections
- [ ] Add support for multiple attention layers and model parallelism
- [ ] Implement DiffQKV Attention with setting to (more) aggressively compress $K$ compared to $V$ components, whilst also increasing the head-dimension of $Q$.


## To-Check 
- Combining top-$k$ makes batching hard
- LoRANN operates on latent $K$ and $V$ vectors, but training uses full-dimensional keys for exact top-$k$, which creates a projection mismatch
  - Might need to jointly train the latent projection layer with forgetful masking to align the latent and full space similarity
- <s>Latent projections of MLA need to preserve rank order of attention scores as a poorly trained projection layer would degrade ANN recall</s>
  - <s>Consider contrastive loss</s>
  - Fixed: LoRANN rank reduction objective should fix 
- Dynamic $k$ selection to adjust $k$ to bound to an approximation error is kind of hard for long sequences because $\lvert A_\text{full} - A_\text{top-k} \rvert$ is intractable for larger sequences so something like entropy or gradient variance is needed? 
- MoE routing for KVs makes ANN operations complicated. Note further that ANN must index expert-specific latent vectors so this adds memory and compute cost. 
  - Also indexing expert-specific keys in ANNs may require $O(E \times N)$ memory for $E$ experts and $N$ tokens which makes it kind of prohibitive imo
- Might need separate attention head for persistent tokens (global top-$k$) and regular tokens (sliding window) to efficiently mix them into one layer. 

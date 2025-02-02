# Pseudocode for FCTK

There are 3 main components to FCTK:
  - The forgetful top-$k$ attention mechanism
  - Persistent tokens
  - Sliding window attention


We pretend that we are doing normal causal attention thus normal algorithms like FA-2 can be used
We can do this because we can then mask out tokens that we don't want contributing to the top-k attention
This is because the union of the masks for persistent, sliding window, and top-k is equivalent to the mask for causal attention

To implement the dynamic top-k attention in training:
  - We already have the attention scores $QK^{\intercal}$ for all relevant tokens that would be in the top-k
  - We first compute the partition function $Z$ for sliding window and persistent tokens for each row
  - Sort attention scores that should be in top-k in descending order <- Do math to figure out what the indices should be
  - Keep adding tokens to the top-k until the softmaxxed attention score for the ($k+1$)th token is less than the threshold, where the threshold is user defined
  - Mask tokens in the top-k with a probability defined by the user

To implement the dynamic top-k attention in inference:
  - Keep persistent tokens and sliding window tokens in VRAM
  - Store the top-k tokens in LoRANN in DRAM
  - Use CPU to find centroids
  - Directly copy vectors from DRAM to VRAM
  - Compute attention scores for ANN vectors
  - Perform a linear search if any attention scores after softmax are below the threshold
  - If there are no attention scores below the threshold, find neighbouring centroid and load into VRAM and repeat
  - If there are attention scores below the threshold, update softmax values for persistent tokens and sliding window tokens
  - Return appropriate attention scores with values computed since the vectors in ANN create the value vectors cos MLA


# LoRA Implementation in Pytorch

Just playing around with LoRA (Low-Rank Adaptation) Basically, we're making big neural nets more efficient by adding these tiny rank decomposition matrices.

## What's LoRA?

Instead of training ALL the parameters (which is like a lot), LoRA says " what if we just add a small update matrix?" The math looks like this:

$$h = W_0x + ∆Wx$$

where `∆W` is our clever little low-rank decomposition:

$$∆W = BA$$

Here, `B` is a `d × r` matrix and `A` is an `r × k` matrix, where `r` is wayyy smaller than both `d` and `k`. Pretty neat, right?

## Why is this cool? 

1. Way fewer parameters to train (like, seriously way fewer)
2. Original weights stay untouched (plug-and-play!)
3. Can target specific parts of the model that need help

## The Process 

1. Train the full model first (but not too much)
2. Find which parts are struggling
3. Add LoRA just to fix those parts
4. Original weights don't change at all!

## Quick Math Note 

The actual update in training looks like:

$$W = W_0 + \frac{\alpha}{r}BA$$

where:
- $W_0$ is the frozen original weights
- $\alpha$ is a scaling factor
- $r$ is our low rank (usually like 1-16)
- $B$ and $A$ are our small trainable matrices


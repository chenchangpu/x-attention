# Reference - XAttention: Block Sparse Attention with Antidiagonal Scoring

## Changes
- Modify the selection method of sparse blocks to make it faster.
- The block sparse attention operator has been re-optimized using Triton and TileLang (the Triton version has been optimized  for Hopper, and its performance surpasses the original Block Sparse Attention kernel optimized based on Flash Attention 2).

## usage
- Xattention_prefill(q, k, v, threshold=threshold, use_pooling=True).
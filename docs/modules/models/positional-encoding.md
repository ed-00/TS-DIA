
# Overview

Positional Encoding (PE) is a technique that modifies a sequence of inputs in such a way to create systematic pattern which is relates each section of the inputs to others denoting their relative positioning.

PE is widely used in conjunction with self attention in the transformer architecture since the attention module is position indifferent. Unlike RNNs which encodes the positioning by virtue of each recursion step.

Positional encoding is presented in the literature in two main form:
      - Relative
      - Absolute

## Formal Definition
let $\mathbb{S} \in \{w_i\}^{N}_{i=1}$ be a sequence of $N$ input tokens with $w_i$ being the $i^{th}$ element. 


## Absolute positional embedding



```bibtex
@misc{su2023roformerenhancedtransformerrotary,
      title={RoFormer: Enhanced Transformer with Rotary Position Embedding}, 
      author={Jianlin Su and Yu Lu and Shengfeng Pan and Ahmed Murtadha and Bo Wen and Yunfeng Liu},
      year={2023},
      eprint={2104.09864},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2104.09864}, 
}
```
# Make_qkx3_quants
[Link to the method](https://github.com/ggerganov/llama.cpp/blob/30f80ca0bcee58669ada7a94244eeccc8c4807cc/ggml/src/ggml-quants.c#L2008)

The same method as [make_qkx2_quants](make_qkx2_quants.md) with small adjustement.

# Problem Statement

Minimize the square difference between quantized weights and the original values:

```math
F(s) = \sum_i w_i (s q_i - x_i)^2
```

But in this method $w_i$ is optional, and in cases when $w_i$ is not provided the following assumtion is made:

```math
w_i = x_i^2
```

This assumption gives more importance to the values (model weights) with higher magnitude. And in that case [imatrix](importance_matrix.md) is optinal.

But in case of presended imatrix, this method performes similarly to `make_qkx2_quants`.
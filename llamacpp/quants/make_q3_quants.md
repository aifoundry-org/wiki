# Make_q3_quants
[Link to the method](https://github.com/ggerganov/llama.cpp/blob/30f80ca0bcee58669ada7a94244eeccc8c4807cc/ggml/src/ggml-quants.c#L1708)

# Problem Statement

$$
F(s) = \sum_i x_i^2 (s (q_i - n_{max}) - x_i) \quad (1)
$$

where:
- $s$ is the quantization scale
- $q_i$ are the quantized weights
- $x_i$ are the original weights
- $n_{max}$ is a non-learnable parameter
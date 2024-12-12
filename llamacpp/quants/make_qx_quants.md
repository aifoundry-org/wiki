# Make_qx_quants
[Link to the method](https://github.com/ggerganov/llama.cpp/blob/30f80ca0bcee58669ada7a94244eeccc8c4807cc/ggml/src/ggml-quants.c#L1639)

# Problem Statement

The idea is to formulate the quantization problem as follows and solve it using Least Squares Method:

Minimize the square difference between quantized weights and the original values:

```math
F(s) = \sum_i w_i (s q_i - x_i)^2 \quad (1)
```

where:
- $w_i$ is the weight importance *(taken from the importance matrix in the previous step)*,
- $s$ is the quantization scale,
- $q_i$ are the quantized weights,
- $x_i$ are the original weights.

The equation can be rewritten as:

$$
F = \sum_i w_i (s q_i - x_i)^2 = 
\sum_i (w_i s^2 q_i^2 - 2 w_i s q_i x_i + w_i x_i^2)
$$

Further simplification gives:

$$
F = s^2 \sum_i w_i q_i^2 - 2s \sum_i w_i q_i x_i + \sum_i w_i x_i^2
$$

### Solving for $s$

Now, let's take the derivative of this function with respect to $s$ and solve for $s$ to determine the optimal scale.

$$
\frac{\partial F}{\partial s} = 2s \sum_i w_i q_i^2 - 2 \sum_i w_i q_i x_i
$$

Setting the derivative equal to zero:

$$
\frac{\partial F}{\partial s} = 0 \implies 2s \sum_i w_i q_i^2 - 2 \sum_i w_i q_i x_i = 0
$$

Simplifying this:

$$
s \sum_i w_i q_i^2 = \sum_i w_i q_i x_i
$$

Solving for $s$:

$$
s = \frac{\sum_i w_i q_i x_i}{\sum_i w_i q_i^2}
$$

This is exactly what is implemented [here](https://github.com/ggerganov/llama.cpp/blob/30f80ca0bcee58669ada7a94244eeccc8c4807cc/ggml/src/ggml-quants.c#L1681).

### The Actual Problem

Now that the scale $s$ can be computed for the given $q_i$, the actual problem is to find such $q_i$ and $s$ that minimize the function $(1)$. This is a mixed integer problem, which is very hard to solve in general.

### Approximation Approach

The approach is to iterate over some permutations of $q_i$ to achieve lower perplexity (ppl) of the target model rather than directly minimizing function (1), as this function only measures the similarity between quantized weights and original weights, not the model's quality itself.

### Quantization loop


1. **Choose an initial scale $s$**:

$$
s_m = \frac{(\sum_i w_i q_i x_i)_m}{(\sum_i w_i q_i^2)_m}
$$

#### For each step $is$ from $-9$ to $9$

*$(-9, 9)$ range is a heuristic*


2. **Choose the $iscale$**:

$$
iscale = \frac{-(nmax + 0.1 \cdot is)}{max}
$$

where:
* $nmax$ is a parameter
* $max$ is observed max weights value

3. **Define the initial alignment between quantized values and original ones**:

$$
best_m = \frac{(\sum_i w_i q_i x_i)^2_m}{(\sum_i w_i q_i^2)_m}
$$

4. **Calculate the quantized weights based on the chosen $iscale$**:

$$
q = \text{clip}(\text{round}(iscale \cdot x), i, j)
$$

5. **Calculate new intermediate sums based on the newly quantized weights**:

$$
(\sum_i w_i q_i x_i)_{m+1}
$$

$$
(\sum_i w_i q_i^2)_{m+1}
$$

6. **Check if the current alignment is better than the previous one, and update the scale and alignment**:

```math
s_{m+1} = 
\begin{cases}
\frac{(\sum_i w_i q_i x_i)_{m+1}}{(\sum_i w_i q_i^2)_{m+1}} & \text{if} & (\sum_i w_i q_i x_i)^2 > best_{m} \cdot \sum_i w_i q_i^2 \\
s_m & \text{otherwise}
\end{cases}
```

$$
best_{m+1} = 
\begin{cases} 
s_{m+1} \cdot \sum_i w_i q_i x_i & \text{if} & (\sum_i w_i q_i x_i)^2 > best_m \cdot \sum_i w_i q_i^2 \\
best_m & \text{otherwise}
\end{cases}
$$

7. **Repeat from step $2$**.
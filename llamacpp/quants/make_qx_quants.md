# Make_qx_quants
[Link to the method](https://github.com/ggerganov/llama.cpp/blob/30f80ca0bcee58669ada7a94244eeccc8c4807cc/ggml/src/ggml-quants.c#L1639)

## Problem statement
Idea is to formulate the quantization problem as follows. 

Minimize the square difference between quantized weights and the original values.
1. $$F = \sum_i w_i (s q_i - x_i)^2 $$

In this equation:
- $w_i$ - weights importance *(taken from improtance matrix on the previous step)*
- $s$   - quantization scale
- $q_i$ - quantized weights
- $x_i$ - original weights

Equation can be rewritten:

$$F = \sum_i w_i (s q_i - x_i)^2 = 
\sum_i (w_i s^2q_i^2 - 2 w_i s q_i x_i + w_i x_i^2)$$

And clarified:

$$F = s^2\sum_i w_i q_i^2 - 2s \sum_i w_i q_i x_i + \sum_i w_i x_i^2$$

Let's take derivative of that function and solver for $s$ in order to determine scale.

$$\frac{dF}{ds} = 2s \sum_i w_i q_i^2 -2 \sum_i w_i q_i x_i$$
$$\frac{dF}{ds} = 0 \implies 2s \sum_i w_i q_i^2 -2 \sum_i w_i q_i x_i = 0 $$
$$s\sum_iw_iq_i^2 = \sum w_i q_i x_i$$

Solving for s:
$$s = \frac{\sum_i w_i q_i x_i}{\sum_i w_i q_i^2}$$

Which is exactly what is written here https://github.com/ggerganov/llama.cpp/blob/30f80ca0bcee58669ada7a94244eeccc8c4807cc/ggml/src/ggml-quants.c#L1681

Now scale $s$ can be computer for the given $q_i$. But the actual problem is to find such $q_i$ and $s$ that would minimize function 1. That is formulated as mixed integer problem that is very hard to solve in general.
___

So, the approach is to iterate over some permutations of $q$ in order to achieve lower ppl of target model rather than minimize function 1. directly, because this function is a measure of similarity of quantizedand not the measure of quality of model itself.

The algorithm is folows:
1. Choose initial scale $s$
$$s_m = \frac{(\sum_i w_i q_i x_i)_m}{(\sum_i w_i q_i^2)_m}$$
2. Choose the initial $iscale$

3. Choose initial definition of allignment of quantized values and the original ones as:
$$best_m = \frac{(\sum_i (w_i q_i x_i)^2)_m}{(\sum_i w_i q_i^2)_m} $$

4. Calculate the quantized weight based on chosen $iscale$:
$$q = clip(round(iscale * x), i,j)$$

5. Calculate new intermediate sums based on newly quantized weights:
$$(\sum_i w_i q_i x_i)_{m+1}$$
$$(\sum_i w_i q_i^2)_{m+1}$$

6. Check if current allignment better than the previous one and update scale and allignment:
$$
s_{m+1} = 
\begin{cases}
\frac{(\sum_i w_i q_i x_i)_{m+1}}{(\sum_i w_i q_i^2)_{m+1}} & if & \sum_i (w_i q_i x_i)^2 > best_{m} \cdot  \sum_i w_i q_i^2\\
s_m & else
\end{cases}
$$
\[x=y\]
$$
best_{m+1} = 
\begin{cases} 
    s_{m+1} \cdot \sum_i w_i q_i x_i & if &  \sum_i (w_i q_i x_i)^2 > best_{m} \cdot  \sum_i w_i q_i^2\\
    best_{m} & else
\end{cases}
$$

7. Variate $iscale$, try again from the step 4

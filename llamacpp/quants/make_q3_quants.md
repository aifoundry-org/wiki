# Make_q3_quants
[Link to the method](https://github.com/ggerganov/llama.cpp/blob/30f80ca0bcee58669ada7a94244eeccc8c4807cc/ggml/src/ggml-quants.c#L1708)

# Problem Statement
The idea is to formulate the quantization problem as follows and solve it using Least Squares Method:

Minimize the square difference between quantized weights and the original values:

$$
F(s) = \sum_i w_i (s q_i - x_i)^2 \quad (1)
$$

where:
- $s$ is the quantization scale
- $q_i$ are the quantized weights
- $x_i$ are the original weights
- $w_i$ is equals to $x_i^2$ giving more improtance to the weights with higher magnitude

### Solving for $s$

Solution exactly the same as in [make_qx_quants](make_qx_quants.md#solving-for) method. Which gaves an equation to define scale $s$:

$$
s = \frac{\sum_i w_i q_i x_i}{\sum_i w_i q_i^2} \quad(2)
$$

### Quantization loop
In this method approach similar to **Coordinate decent** is performed.
The overall idea is to minimize overall function $(1)$ iteratively along every presented axis and check whether the target function is minimized.

1. Obtain initial quantized values:
    
    1. Initialize scale $s$ as:
        $$
        s = \frac{-nmax}{max(x)}
        $$ 
        where:
        * $nmax$ is a parameter defining range of quantization levels.
    
    2. Quantize weights:
        $$
        q =  {clip}( {round}(s \cdot x), -nmax, nmax-1)
        $$
2. Definine initial terms to express scale $s$:
```math
sumlx = \sum_i w_i q_i x_i
```
```math
suml2 = \sum_i w_i q_i^2
```

3. Fix values along dimension $j$ and extract it from the calculated sums:
```math
slx = sumlx - w x_j q_j = \sum_i w_i q_i x_i - w q_j x_j
```
```math
sl2 = suml2 - w q_i^2 = \sum_i w_i q_i^2 - w q_j^2
```

4. Define new scale $s$:
```math
s_{new} = \frac{\sum_i w_i q_i^2 - w q_i^2}{\sum_i w_i q_i x_i - w q_j x_j}
```

5. Quantize values with new scale $s_{new}$ like in step 1.2
    $$
    q_{new} =  {clip}( {round}(s_{new} \cdot x), -nmax, nmax-1)
    $$
6. Add removed on step 3 terms to $slx$ and $sl2$ sums but with new quantized values:
```math
slx = slx + w x_j q_{new}
```

```math
sl2 = sl2 + w q_{new}^2
```
7. Check if new quantization improves target minimization goal.

    1. To deduce that target function was improved for new quantization valules and therefore quantization error was decreased $\Delta F$ should be calculated:


    ```math
    \Delta F = F_{new} - F_{old} = \\(\sum_i w_i x_i^2 - \frac{(\sum_i w_i x_i q_i)_{new}^2}{\sum_i (w_i q_i^2)_{new}}) - (\sum_i w_i x_i^2 - \frac{(\sum_i w_i x_i q_i)_{old}^2}{\sum_i (w_i q_i^2)_{old}}) = \\ (\frac{(\sum_i w_i x_i q_i)_{old}^2}{(\sum_i w_i q_i^2)_{old}}) - (\frac{(\sum_i w_i x_i q_i)_{new}^2}{(\sum_i w_i q_i^2)_{new}})
    ```
    2. Condition $\Delta F < 0$ should be met in order for quantization error to decrease. Therefore:
    ```math
    \frac{(\sum_i w_i x_i q_i)_{new}^2}{(\sum_i w_i q_i^2)_{new}} > \frac{(\sum_i w_i x_i q_i)_{old}^2}{(\sum_i w_i q_i^2)_{old}}
    ```
    3. Rearrange the inequality to isolate the terms involving the new quantization values:

    ```math
    (\sum_i w_i x_i q_i)_{new}^2 \cdot (\sum_i w_i q_i^2)_{old} > (\sum_i w_i x_i q_i)_{old}^2 \cdot (\sum_i w_i q_i^2)_{new}
    ```

    4. Substitute the definitions of $sumlx$ and $suml2$:

    ```math
    ( {sumlx}_{new})^2 \cdot  {suml2}_{old} > ( {sumlx}_{old})^2 \cdot  {suml2}_{new} \quad (3)
    ```

    5. If condition $(3)$ is met, then accept new quantization value along this dimensino $j$ and set sums $$sumlx_{old} = sumlx_{new}$$ $$suml2_{old} = suml2_{new} $$

8. Proceed to the next dimension at stage 3.
9. Iterate over all dimensions 5 times *(heuristic i guess)*
10. Shift all quantization values by $nmax$ into right making them positive.
11. Calculate final quantization scale and return it
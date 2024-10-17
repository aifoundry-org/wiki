# Importance matrix

[Link to pr](https://github.com/ggerganov/llama.cpp/pull/4861)

The common approach for computing an importance matrix is to get the weights gradients from a training run on a given set of training tokens. When the gradient of a given model weight is small, it means that a large change in the model weight will result in a small change in model performance, and, vice versa, a large gradient implies a large change in model performance from a small change in model weight. 

The goal is to minimize the impact of the quantization. 

Function to minimize states as follows :

$$F = [\sum_j(q_j - w_j)a_j]^2$$

$q_j$ - quantized weights tensor row\
$w_j$ - weights tensor row\
$a_j$ - activation column\

To solve this problem second-order optimization approach is used.
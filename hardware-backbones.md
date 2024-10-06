## Backbones

- ### CPU
    - **Overview**  
        CPUs are the most popular and affordable type of hardware for neural network inference. They can be equipped with large amounts of memory, allowing inference on models like **7B**, **11B**, and others without needing quantization.
    - **Performance**
        - **Advantages**
            - Large memory capacity per CPU, up to **2TB**.
            - Easily scalable:
                - Multiple CPUs per node is a common solution.
                - Many interfaces for communication between nodes.
            - Highly flexible: Supports future architecture modifications.
        - **Disadvantages**
            - Significantly slower compared to other devices. [CPU benchmark link](https://github.com/Mozilla-Ocho/llamafile/discussions/450) shows typical performance is **≤100 tokens/second** on modern consumer CPUs.
            - Specific instruction sets (like **AVX-512**) can provide a performance boost but are rarely available.
            - Worse energy efficiency compared to other hardware.
    - **Experience**
        - **Advantages**
            - Relatively cheap and widely available.
            - Easy to set up—no need for specialized drivers or software.
        - **Disadvantages**
            - Limited performance: While every model can theoretically run on a CPU, larger models often become impractical, even with quantization applied.
- ### GPU
    - #### NVIDIA
        - **Overview**  
            NVIDIA dominates the GPU market, and its solutions are considered optimal for AI/ML applications.
        - **Performance**
            - **Advantages**
                - **High throughput**: GPUs are optimized for parallelism with a large number of computational cores.
                - **FP16/TF32/INT8 precision**: Tensor Cores support lower precision operations, leading to significant performance boosts. [NVIDIA A100 link](https://www.nvidia.com/en-us/data-center/a100/)
                - Better energy efficiency than CPUs. [Energy efficiency comparison](https://developer.nvidia.com/blog/inference-next-step-gpu-accelerated-deep-learning/)
            - **Disadvantages**
                - Specific hardware is required to fully utilize the GPU's potential.
        - **Experience**
            - **Advantages**
                - Well-established software stack, including **CUDA** and **cuDNN**, with extensive community and developer support.
                - Widely available on cloud platforms, allowing access without purchasing hardware.
                - A variety of models cater to different performance needs.
            - **Disadvantages**
                - **Cost**: High-end models like the **A100** and **H200** are expensive to purchase and operate, which can be prohibitive for smaller organizations.
                - Setting up environments (drivers, **CUDA** dependencies) can be challenging for non-expert users.
                - **Driver compatibility**: Server-grade hardware has many variables, and GPU operation may not be guaranteed unless certified by **NVIDIA**.
    - #### AMD
		- **Overview**  
				AMD has gained ground in the AI/ML space with its **Radeon Instinct** and **MI200/MI300 series** GPUs, focusing on energy efficiency and price-to-performance.
		- **Performance**  
				To properly compare **AMD** to **NVIDIA**, let's consider **MI300X** vs **H200**.  
				[AMD MI300X link](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)  
				[NVIDIA H200 link](https://www.nvidia.com/en-us/data-center/h200/)
			- **Advantages**  
				-  **Higher memory capacity**: 192 GB vs 141 GB.
				- **Higher memory bandwidth**: 5.2 TB/s vs 5 TB/s.
				- **Better FP32 performance**: 163.4 TFLOPS vs 67 TFLOPS.
			- **Disadvantages**  
				- **Higher TDP**: 750W vs 700W.
                - **Lower TF32 performance** 653.7 TFLOPS vs 989 TFLOPS
				- **Lower FP16 performance**: 1,300 TFLOPS vs 1,979 TFLOPS.
				- **Lower FP8 performance**: 2,610 TFLOPS vs 3,958 TFLOPS
				- **Lower INT8 performance**: 2,600 TOPS vs 3,958 TOPS.
		- **Experience**
			- **Advantages**  
				- **ROCm** (open software stack) vs **CUDA** (proprietary).
				- The **MI300X** is estimated to be cheaper than **H200**: ~$15,000 vs ~$28,000.
			- **Disadvantages**  
				- Less mature software stack; **CUDA** and **cuDNN** are still more widely supported.
				- Fewer software tools and optimizations compared to NVIDIA’s ecosystem.
	- #### Moore Threads
		- **Overview**  
			Moore Threads is a new player in the GPU market, focused on general-purpose computing and AI inference, primarily targeting the Chinese market.
		- **Performance**  
			At this stage, Moore Threads lacks competitive performance compared to AMD and NVIDIA.
		- **Experience**
			- **Advantages**  
				- A promising future player in the GPU market.
				- Supported by the **llama.cpp** community.
			- **Disadvantages**  
				- Limited availability and software support.
				- Not yet a serious competitor in the global market, especially against **NVIDIA** and **AMD**.
- ### Specilized hardware
    - #### Google TPU
    - #### Graphcore IPU
    - #### Amazon Trainium
    - #### SambaNova SN Series
    - #### Cerebras CS-2
    - #### Groq
    - #### D-Matrix Corsair
    - #### Cambricon Siyuan
    - #### Intel Gaudi
    - #### Etched Sohu
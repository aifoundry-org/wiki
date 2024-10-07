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
            - Ease of prototyping
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
        - **Overview**\
            Tensor Processing Units (TPUs) are Google's custom-developed application-specific integrated circuits (ASICs) used to accelerate machine learning workloads. For more detailed information about TPU hardware, see System Architecture. Cloud TPU is a web service that makes TPUs available as scalable computing resources on Google Cloud. [Google TPU intro](https://cloud.google.com/tpu/docs/intro-to-tpu)
        - **Performance**
            - **Advantages**
                - Massive amount of ALU's which results in more parallelism then GPU
            - **Disadvantages**
        - **Experience**
            - **Advantages**
            - **Disadvantages**
    - #### Graphcore IPU
        - **Overview**\
        Intelligency Processing Units (IPUs) are the ASICs by Graphcore. They provide it as a hardware to be used on-premisis solution, as well as cloud solution to be connected through Polar SDK. [Graphcore IPU](https://www.graphcore.ai/products/ipu)
    - #### Amazon Trainium
        - **Overview**\
        AWS Trainium is the machine learning (ML) chip that AWS purpose built for deep learning (DL) training of 100B+ parameter models. Each Amazon Elastic Compute Cloud (Amazon EC2) Trn1 instance deploys up to 16 Trainium accelerators to deliver a high-performance, low-cost solution for DL training in the cloud. [Amazon Trainium](https://aws.amazon.com/machine-learning/trainium/)
    - #### SambaNova SN Series
        - **Overview**\
        SambaNova offers the SN40L chip, a high-performance, energy-efficient solution optimized for AI inference tasks, especially for large language models like Llama 3.1. It provides superior throughput according to their [site](https://sambanova.ai/blog/sn40l-chip-best-inference-solution) , fewer chips required, and a smaller datacenter footprint compared to competitors. Its reconfigurable dataflow architecture allows for faster, more efficient handling of massive models, making it ideal for cost-sensitive, high-performance AI workloads in datacenters.

    - #### Cerebras CS-3
        - **Overview**\
        The Cerebras CS-3 is a third-generation wafer-scale AI accelerator designed for training large AI models. It features over 4 trillion transistors, making it 57 times larger than the biggest GPU, and it is twice as fast as its predecessor. The CS-3 can support clusters of up to 2048 systems and train models with up to 24 trillion parameters. [Site](https://cerebras.ai/blog/cerebras-cs3)
    - #### Groq
        - **Overview**\
        Groq offers AI inference solutions focusing on high scalability and low latency:\
            **GroqCloud**: Cloud-based inference platform with both public and private instances for scalable AI workloads.\
            **GroqRack**: On-premises cluster for enterprise data centers, ideal for scaling up AI compute needs.\
            **GroqCard** Accelerator: A plug-and-play PCIe card with Groq's technology for easy integration into servers, featuring deterministic performance and large bandwidth.\
            All these products are built around Groq's Logical Processing Unit (LPU) technology.
            [Site](https://groq.com/)
    - #### D-Matrix Corsair
        - **Overview**\
        The D-Matrix Corsair is an advanced AI inference solution that uses a Digital In-Memory Compute (DIMC) architecture. This architecture integrates compute and memory directly, which drastically improves efficiency for inference tasks on large language models (LLMs). Corsair offers up to 20x better throughput, 20x lower latency, and up to 30x better total cost of ownership (TCO) compared to traditional GPU-based solutions.
        [Site](https://www.d-matrix.ai/product/)
    - #### Intel Gaudi
        - **Overview**\
        Intel Gaudi is an AI accelerator series designed to enhance the efficiency of deep learning workloads, particularly focusing on generative AI and large language models (LLMs). Developed by Habana Labs, a subsidiary of Intel, the Gaudi accelerators offer a more cost-effective alternative to traditional GPU-based architectures like NVIDIA for large-scale AI tasks. [Site](https://www.intel.com/content/www/us/en/products/details/processors/ai-accelerators/gaudi-overview.html)
    - #### Etched Sohu
        - **Overview**\
        Etched Sohu is an AI chip developed by the startup Etched, founded by a group of Harvard dropouts. The chip is specifically designed for AI inference tasks, particularly for **transformer-based** models like LLMs. It is described as an ASIC (Application-Specific Integrated Circuit), which means it is tailored to run transformer models efficiently compared to general-purpose GPUs.
        [Site](https://www.etched.com/announcing-etched)

### Usefull links
- https://artificialanalysis.ai/models/llama-3-1-instruct-70b/providers
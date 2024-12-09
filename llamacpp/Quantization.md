# Quantization Documentation
This document provides explanations for quantization methods from [llamacpp](https://github.com/ggerganov/llama.cpp), though they have not yet been verified by the original authors.

### To have a brief understanding of what quantization is of tu fresh up you knowledge consider this take a look at this resources:
- https://www.youtube.com/watch?v=tFmQj7W4qlk - great video with good visualizations
- https://arxiv.org/abs/2106.08295 - pretty paper with decent basic information
- https://arxiv.org/abs/1712.05877 - another decent paper on quantization
- https://huggingface.co/docs/optimum/concept_guides/quantization - brief overview on quantization from huggingface
- https://www.youtube.com/watch?v=zpOSA503DAs - video of very questionable quality, with a bit less questionable content.

## General Overview
All methods discussed fall under post-training quantization (PTQ). Additionally, the document presents interesting concepts for calculating importance matrices and different approaches to quantization granularity.

## Implementation

In this section each quantization method is presented as well as the links to it's implementation.

| **type** | **source** | **description** |
| --- | --- | --- |
| F64 | [**Wikipedia**](https://en.wikipedia.org/wiki/Double-precision_floating-point_format) | 64-bit standard IEEE 754 double-precision floating-point number. |
| I64 | [**GH**](https://github.com/ggerganov/llama.cpp/pull/6062) | 64-bit fixed-width integer number. |
| F32 | [**Wikipedia**](https://en.wikipedia.org/wiki/Single-precision_floating-point_format) | 32-bit standard IEEE 754 single-precision floating-point number. |
| I32 | [**GH**](https://github.com/ggerganov/llama.cpp/pull/6045) | 32-bit fixed-width integer number. |
| F16 | [**Wikipedia**](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) | 16-bit standard IEEE 754 half-precision floating-point number. |
| BF16 | [**Wikipedia**](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) | 16-bit shortened version of the 32-bit IEEE 754 single-precision floating-point number. |
| I16 | [**GH**](https://github.com/ggerganov/llama.cpp/pull/6045) | 16-bit fixed-width integer number. |
| Q8_0 | [**GH**](https://github.com/huggingface/huggingface.js/pull/615#discussion_r1557654249) | 8-bit round-to-nearest quantization (`q`). Each block has 32 weights. Weight formula: `w = q * block_scale`. Legacy quantization method (not used widely as of today). |
| Q8_1 | [**GH**](https://github.com/huggingface/huggingface.js/pull/615#discussion_r1557682290) | 8-bit round-to-nearest quantization (`q`). Each block has 32 weights. Weight formula: `w = q * block_scale + block_minimum`. Legacy quantization method (not used widely as of today) |
| Q8_K | [**GH**](https://github.com/ggerganov/llama.cpp/pull/1684#issue-1739619305) | 8-bit quantization (`q`). Each block has 256 weights. Only used for quantizing intermediate results. All 2-6 bit dot products are implemented for this quantization type. Weight formula: `w = q * block_scale`. |
| I8 | [**GH**](https://github.com/ggerganov/llama.cpp/pull/6045) | 8-bit fixed-width integer number. |
| Q6_K | [**GH**](https://github.com/ggerganov/llama.cpp/pull/1684#issue-1739619305) | 6-bit quantization (`q`). Super-blocks with 16 blocks, each block has 16 weights. Weight formula: `w = q * block_scale(8-bit)`, resulting in 6.5625 bits-per-weight. |
| Q5_0 | [**GH**](https://github.com/huggingface/huggingface.js/pull/615#discussion_r1557654249) | 5-bit round-to-nearest quantization (`q`). Each block has 32 weights. Weight formula: `w = q * block_scale`. Legacy quantization method (not used widely as of today). |
| Q5_1 | [**GH**](https://github.com/huggingface/huggingface.js/pull/615#discussion_r1557682290) | 5-bit round-to-nearest quantization (`q`). Each block has 32 weights. Weight formula: `w = q * block_scale + block_minimum`. Legacy quantization method (not used widely as of today). |
| Q5_K | [**GH**](https://github.com/ggerganov/llama.cpp/pull/1684#issue-1739619305) | 5-bit quantization (`q`). Super-blocks with 8 blocks, each block has 32 weights. Weight formula: `w = q * block_scale(6-bit) + block_min(6-bit)`, resulting in 5.5 bits-per-weight. |
| Q4_0 | [**GH**](https://github.com/huggingface/huggingface.js/pull/615#discussion_r1557654249) | 4-bit round-to-nearest quantization (`q`). Each block has 32 weights. Weight formula: `w = q * block_scale`. Legacy quantization method (not used widely as of today). |
| Q4_1 | [**GH**](https://github.com/huggingface/huggingface.js/pull/615#discussion_r1557682290) | 4-bit round-to-nearest quantization (`q`). Each block has 32 weights. Weight formula: `w = q * block_scale + block_minimum`. Legacy quantization method (not used widely as of today). |
| Q4_K | [**GH**](https://github.com/ggerganov/llama.cpp/pull/1684#issue-1739619305) | 4-bit quantization (`q`). Super-blocks with 8 blocks, each block has 32 weights. Weight formula: `w = q * block_scale(6-bit) + block_min(6-bit)`, resulting in 4.5 bits-per-weight. |
| Q3_K | [**GH**](https://github.com/ggerganov/llama.cpp/pull/1684#issue-1739619305) | 3-bit quantization (`q`). Super-blocks with 16 blocks, each block has 16 weights. Weight formula: `w = q * block_scale(6-bit)`, resulting. 3.4375 bits-per-weight. |
| Q2_K | [**GH**](https://github.com/ggerganov/llama.cpp/pull/1684#issue-1739619305) | 2-bit quantization (`q`). Super-blocks with 16 blocks, each block has 16 weight. Weight formula: `w = q * block_scale(4-bit) + block_min(4-bit)`, resulting in 2.5625 bits-per-weight. |
| IQ4_NL | [**GH**](https://github.com/ggerganov/llama.cpp/pull/5590) | 4-bit quantization (`q`). Super-blocks with 256 weights. Weight `w` is obtained using `super_block_scale` & `importance matrix`. |
| IQ4_XS | [**HF**](https://huggingface.co/CISCai/OpenCodeInterpreter-DS-6.7B-SOTA-GGUF/blob/main/README.md?code=true#L59-L70) | 4-bit quantization (`q`). Super-blocks with 256 weights. Weight `w` is obtained using `super_block_scale` & `importance matrix`, resulting in 4.25 bits-per-weight. |
| IQ3_S | [**HF**](https://huggingface.co/CISCai/OpenCodeInterpreter-DS-6.7B-SOTA-GGUF/blob/main/README.md?code=true#L59-L70) | 3-bit quantization (`q`). Super-blocks with 256 weights. Weight `w` is obtained using `super_block_scale` & `importance matrix`, resulting in 3.44 bits-per-weight. |
| IQ3_XXS | [**HF**](https://huggingface.co/CISCai/OpenCodeInterpreter-DS-6.7B-SOTA-GGUF/blob/main/README.md?code=true#L59-L70) | 3-bit quantization (`q`). Super-blocks with 256 weights. Weight `w` is obtained using `super_block_scale` & `importance matrix`, resulting in 3.06 bits-per-weight. |
| IQ2_XXS | [**HF**](https://huggingface.co/CISCai/OpenCodeInterpreter-DS-6.7B-SOTA-GGUF/blob/main/README.md?code=true#L59-L70) | 2-bit quantization (`q`). Super-blocks with 256 weights. Weight `w` is obtained using `super_block_scale` & `importance matrix`, resulting in 2.06 bits-per-weight. |
| IQ2_S | [**HF**](https://huggingface.co/CISCai/OpenCodeInterpreter-DS-6.7B-SOTA-GGUF/blob/main/README.md?code=true#L59-L70) | 2-bit quantization (`q`). Super-blocks with 256 weights. Weight `w` is obtained using `super_block_scale` & `importance matrix`, resulting in 2.5 bits-per-weight. |
| IQ2_XS | [**HF**](https://huggingface.co/CISCai/OpenCodeInterpreter-DS-6.7B-SOTA-GGUF/blob/main/README.md?code=true#L59-L70) | 2-bit quantization (`q`). Super-blocks with 256 weights. Weight `w` is obtained using `super_block_scale` & `importance matrix`, resulting in 2.31 bits-per-weight. |
| IQ1_S | [**HF**](https://huggingface.co/CISCai/OpenCodeInterpreter-DS-6.7B-SOTA-GGUF/blob/main/README.md?code=true#L59-L70) | 1-bit quantization (`q`). Super-blocks with 256 weights. Weight `w` is obtained using `super_block_scale` & `importance matrix`, resulting in 1.56 bits-per-weight. |
| IQ1_M | [**GH**](https://github.com/ggerganov/llama.cpp/pull/6302) | 1-bit quantization (`q`). Super-blocks with 256 weights. Weight `w` is obtained using `super_block_scale` & `importance matrix`, resulting in 1.75 bits-per-weight. |


## Documentation

This section provides links to overviews of various quantization methods.

The descriptions below link to explanations of specific quantization methods and their optimal quantization configurations.

| Quantization Method                            | Bit Configurations                 |
|----------------------------------------------- | ---------------------------------- |
| [quantize_row_iq1_m_impl](quants/quantize_row_iq1_m_impl.md)| IQ1_M                 |
| [make_qkx1_quants](quants/make_qkx1_quants.md) | Not used                           |
| [make_qkx2_quants](quants/make_qkx2_quants.md) | q2_K_ref, q4_K_ref, q5_K_ref       |
| [make_qkx3_quants](quants/make_qkx3_quants.md) | q2_K_impl, q4_K_impl, q5_1_impl    |
| [make_q3_quants](quants/make_q3_quants.md)     | q3_K_ref                           |
| [make_qx_quants](quants/make_qx_quants.md)     | q5_0_impl, q3_K_impl, q6_K_ref, q6_K_impl, q4_0_impl |

---
### Importance matrix calculation

[**importance matrix**](quants/importance_matrix.md)

---



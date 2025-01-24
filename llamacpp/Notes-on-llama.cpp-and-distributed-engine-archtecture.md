# Repository structure

There are two repositories that are of interest to anyone starting to hack on llama.cpp:
   * https://github.com/ggerganov/llama.cpp
   * https://github.com/ggerganov/ggml

Note that llama.cpp repo [has a copy of ggml](https://github.com/ggerganov/llama.cpp/tree/master/ggml) embedded into it. This can be considered as an "upstream" of the standalone ggml. This is especially true for hardware people who are adding support for their backends. They would typically do end-to-end enablement of LLM inference for their acceleratoers in llama.cpp (mostly by just changing code under embedded ggml folder) and then @ggerganov backports these changes in bulk into ggml. E.g. here's how enablement of Huawei's CANN (Compute Architecture of Neural Networks) went from [llama.cpp](https://github.com/ggerganov/llama.cpp/pull/6035) to [gglm](https://github.com/ggerganov/ggml/commit/a06c68343e9976fdfc80917a958b903a0d7c8cc6). Note that these kinds of backports also happen between [whisper.cpp](https://github.com/ggerganov/whisper.cpp) and ggml but at a much slower rate. There appears to be a GH CI job that facilitates these kinds of back-ports (see GitHub workflows below).

# Build system

ggml is an unapologetically CMake managed project. At the same time, llama.cpp seems to have an old school make-based build system that ignores CMake *AND* it has some level of CMake structure that does a lot of the same work that make does. It is certainly possible to build llama.cpp without CMake (and that's what [llamafile](https://github.com/Mozilla-Ocho/llamafile/blob/main/llama.cpp/BUILD.mk) does for example). However, CMake seems to be maintained for the benefit of Windows users (and the like). It would be nice to dig deeper into the relationship between two build systems (especially when/if we start to "librarify" the project more). *NOTE*: that this double build system confuses the heck out of IDEs hence if you're planning to use things like VSCode some level of manual tweaking is going to be required.

# GitHub workflows

TODO: at some point it maybe useful to document this

# Logical architecture

llama.cpp codebase doesn't have a lot of abstraction layers (which is a great thing for the project at this stage of churn!) and most of the functionality is implemented in the top-level [llama.cpp file](https://github.com/ggerganov/llama.cpp/blob/master/src/llama.cpp). The file is big (21k LOC and counting) but pretty well organized. The file itself is mostly a library although there's not really much of an effort to librarify it (either as a .so or .a). Instead, the entry points for various use-cases get statically linked with llama.cpp (and the rest of dependencies). All of the entry-points are located under [examples](https://github.com/ggerganov/llama.cpp/blob/master/examples) folder. The key ones are:
   * [main](https://github.com/ggerganov/llama.cpp/blob/master/examples/main) which is a well-known llama-cli
   * [embedding](https://github.com/ggerganov/llama.cpp/blob/master/examples/embedding) cli for generating embedding vectors with models like BERT

Note that all the entry-points share some common logic (and thus command-line flags) but they also have their own specific ones. It also goes without saying that they expect models specific to their task (e.g. a full-fledged LLM model like llama v3 will be used with llama-cli where an embedding-only model like BERT or NOMIC will be used with llama-embedding (as a quick aside -- since nothing is ever black-n-white in the magical world of AI -- models can be repurposed for various non-standard uses e.g. https://towardsdatascience.com/turn-llama-3-into-an-embedding-model-with-llm2vec-8448005f99aa)

## Backends, aka hardware

llama.cpp defines backends, which implement the ability to execute models. For example, an Nvidia H100 GPU is a backend,
a Huawei Ascend is a backend, etc.

Each backend requires a "driver" - not the official term - i.e. the software to implement the connection between
llama.cpp and the backend.

There are several main steps to implementing a backend:

1. Create the "driver" or backend. This is primarily in GGML.
1. Make the common files of ggml aware of the backend, so it can be registered and used.
1. Make the llama.cpp aware of the backend, so it can be used.

If that sounds like a lot, it is. There are plenty of places to make mistakes.

As referenced above, [here](https://github.com/ggerganov/llama.cpp/pull/6035) is the PR that added the Huawei Ascend (CANN) backend to llama.cpp. The major changes are:

* Additions:
  * Under [ggml](https://github.com/ggerganov/llama.cpp/tree/master/ggml)
    * Created directory for the backend [ggml/src/ggml-cann](https://github.com/ggerganov/llama.cpp/tree/master/ggml/src/ggml-cann)
    * Created primary entrypoint for the driver [ggml/src/ggml-cann.cpp](https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-cann.cpp)
    * Created header file for the driver [ggml/include/ggml-cann.h](https://github.com/ggerganov/llama.cpp/blob/master/ggml/include/ggml-cann.h)
* Modifications:
  * Under [ggml](https://github.com/ggerganov/llama.cpp/tree/master/ggml)
    * Modified [ggml/include/ggml.h](https://github.com/ggerganov/llama.cpp/blob/master/ggml/include/ggml.h) to include `ggml_cpu_has_can()`, which parallels similar hardware-specific capabilities functions, like `ggml_cpu_has_cuda()` and `ggml_cpu_has_vulkan()`
    * Modified [ggml/src/ggml-backend.c](https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-backend.c) to include an `#ifdef GGML_USE_CAN` to register the CANN backend when compiled with `-DGGML_USE_CAN`, as well as [ggml/src/ggml.c](https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml.c) to implement the `ggml_cpu_has_can()` function
  * Under [src](https://github.com/ggerganov/llama.cpp/blob/master/src):
    * Modified [src/llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/src/llama.cpp) to include functions that use the CANN backend for common functions, like:
      * `llama_get_device_count()`
      * `llama_default_buffer_type_offload()`
      * `llama_get_device_memory()`
      * `llama_max_devices()`

The actual registration of devices process is called in [ggml/src/ggml-backend.c](https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-backend.c), where an `#ifdef GGML_USE_<BACKEND>` is used to register the backend. For example, some of the backends:

```c
#ifdef GGML_USE_SYCL
    extern void ggml_backend_sycl_reg_devices(void);
    ggml_backend_sycl_reg_devices();
#endif
#ifdef GGML_USE_METAL
    extern GGML_CALL ggml_backend_t ggml_backend_reg_metal_init(const char * params, void * user_data);
    extern GGML_CALL ggml_backend_buffer_type_t ggml_backend_metal_buffer_type(void);
    ggml_backend_register("Metal", ggml_backend_reg_metal_init, ggml_backend_metal_buffer_type(), NULL);
#endif
#ifdef GGML_USE_VULKAN
    extern GGML_CALL int ggml_backend_vk_reg_devices(void);
    ggml_backend_vk_reg_devices();
#endif
#ifdef GGML_USE_KOMPUTE
    extern GGML_CALL void ggml_backend_kompute_reg_devices(void);
    ggml_backend_kompute_reg_devices();
#endif

#ifdef GGML_USE_CANN
    extern GGML_CALL int ggml_backend_cann_reg_devices(void);
    ggml_backend_cann_reg_devices();
#endif
```

Similar code exists for others. The actual function call is expected to be in the backend-specific file, e.g.

* `ggml_backend_cann_reg_devices` in [ggml/src/ggml-cann.cpp](https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-cann.cpp)
* `ggml_backend_sycl_reg_devices` in [ggml/src/ggml-sycl.cpp](https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-sycl.cpp)
* etc.

The actual registration process involves registering the backend with the backend handler.
For example, the CANN registration function from [ggml/src/ggml-cann.cpp](https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-cann.cpp):

```c
GGML_CALL int ggml_backend_cann_reg_devices() {
    uint32_t device_count = ggml_backend_cann_get_device_count();
    // initialization
    for (uint32_t i = 0; i < device_count; i++) {
        char name[128];
        snprintf(name, sizeof(name), "CANN%d", i);
        ggml_backend_register(name, ggml_backend_reg_cann_init,
                              ggml_backend_cann_buffer_type(i),
                              (void*)(intptr_t)i);
    }
    return device_count;
}
```

Note that since late September 2024, there is a process in place to replace some of this with a common registration
process, involving replaced `ggml-backend.c` with `ggml-backend.cpp`. While this is in place, it appears that the
main branch is missing device support.

Some things become clear from this process.

First, drivers are static. Even though they use a registration process, only one really is enabled at a time,
compiled in. So an executable version of `llama.cpp` really means, "llama.cpp for this OS, architecture and specific
backend." You cannot take llama.cpp compiled for Linux amd64 with a CANN backend and run it on Linux amd64 with a CUDA backend. You would need to recompile it.

This is even more obvious from the "frontend" changes in [src/llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/src/llama.cpp), such as `llama_get_device_count()`. This function is a big `#ifdef` block that checks for the backend and then calls the appropriate function, meaning only one is enabled.

```c
#ifdef GGML_USE_CUDA
#  include "ggml-cuda.h"
#elif defined(GGML_USE_VULKAN)
#  include "ggml-vulkan.h"
#elif defined(GGML_USE_SYCL)
#  include "ggml-sycl.h"
#elif defined(GGML_USE_KOMPUTE)
#   include "ggml-kompute.h"
#elif defined(GGML_USE_CANN)
#   include "ggml-cann.h"
#endif
```

It usually is possible to _disable_ a specific backend if it was compiled with it, i.e. fall back to the CPU instead
of the compiled-in backend.

Second, the process is not simple. It requires changes in the `llama.cpp` frontend, the ggml integration with the backend,
and the backend itself.

# Debugging/control flow

In order to appreciate the very basic control flow/setup your debugging session it is recommended that one starts with a very basic and small model. [BERT for embeddings](https://github.com/aifoundry-org/.github/tree/main/bits-n-bytes) happens to be one of those. Currently most of us are using VSCode so the following [launch.json](https://github.com/aifoundry-org/.github/blob/main/bits-n-bytes/launch.json) template could provide a good start to your debugging sessions. Note that because of the complexity of hooking up VSCode to the llama.cpp build system we're currently stubbing-out the build portion in launch.json (IOW you will have to build the project in your terminal or something by doing `make LLAMA_DEBUG=1`). To not be annoyed by VSCode's attempts to CMake-configure your project only to fail the following diff maybe useful (it will also allow you to have a much nicer interaction with a VSCode debugger targets later on):
```
diff --git a/CMakePresets.json b/CMakePresets.json
index d22ffa49..ba2bfc17 100644
--- a/CMakePresets.json
+++ b/CMakePresets.json
@@ -63,6 +63,7 @@
     { "name": "x64-windows-sycl-debug"  , "inherits": [ "sycl-base", "debug"   ] },
     { "name": "x64-windows-sycl-debug-f16", "inherits": [ "sycl-base", "debug", "sycl_f16" ] },
     { "name": "x64-windows-sycl-release", "inherits": [ "sycl-base", "release" ] },
-    { "name": "x64-windows-sycl-release-f16", "inherits": [ "sycl-base", "release", "sycl_f16" ] }
+    { "name": "x64-windows-sycl-release-f16", "inherits": [ "sycl-base", "release", "sycl_f16" ] },
+    { "name": "default", "inherits": [ "base", "debug" ] }
   ]
 }
```

The basic control flow goes through the following routines (most of them defined in top-level llama.cpp source):
   * `llama_model_load()` (initial entry point for model loading)
       * `llm_load_arch()` -- this loads up a crucial bit of metadata from gguf -- model's architecture. A lot of things from this point on will `case/#ifdef` on that value
       * `llm_load_hparams()` -- loads up the rest of the model's metadata from gguf
       * `llm_load_vocab()` -- this business is kind of specific to LLMs -- TODO: review it more
       * `llm_load_tensors()` -- this is where the actual action and bulk data loading begin, the bulk of this functions figures out where to route the data (what to push to GPUs/CPUs, how to do mmap's etc.) and then it does the actual loading with the one below
           * `load_all_data()` -- study this very carefully to understand how actual marshaling and SerDe of tensor data happens

At this point the model is considered to be loaded in whatever compute/memory units were specified and the computational graph has been constructed at the ggml level (TODO: verify this). This is, however, only half of the battle since depending on the kind of a model and an entry point a lot of the data preparation has to happen before the model can be asked to inference and produce the result. The shortest path for that is probably in the [llama-embeddings](https://github.com/ggerganov/llama.cpp/blob/master/examples/embedding/embedding.cpp#L112) and you can follow that logic.

# Librarification/single node daemon

Ideally llama.cpp could be librarified into .so that can then be embedded into any kind of a control process on a single node. The idea is that a control process will take care of care and feeding of the core inference/fine-tuning engine and will be able to communicate with that engine through some kind of an RPC mechanism. We really hope that the control plane can be outside of the llama.cpp engine itself (and perhaps implemented in Go and hooked to K8s later on). We do expect all of the data plane (including marshaling of the tesnor layers) to reside within llama.cpp though. This division of labor would allow us to not have a high-bandwith connection between the two components (since only control messages will be exchanged through that RPC). In fact, one of the simplest implementation of this architecture could even skip librarification of llama.cpp and go straight to the examples-styles standalone server. In fact, there are already two examples of these kinds of standalone servers in examples:
   * An [RPC server](https://github.com/ggerganov/llama.cpp/blob/master/examples/rpc/README.md)
   * An [OpenAI-compatible inference server](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md)

Extending OpenAI-compatible inference server for control plane doesn't feel right -- since it sits at a different layer in the system, but it may provide a good starting point for implementing a RESTful server for control plane. We should probably consider full-fledged gRPC/protobuf approach here is we decide to go this route (unless [things like Cap'n Proto](https://medium.com/@learnwithshobhit/comparing-capn-proto-and-grpc-in-rust-a-performance-and-feature-analysis-61d2da815d18) are still alive ;-)).

# Distributed engine

Ultimately we would like individual, standalone llama.cpp daemons to be interconnected into a mesh that would allow dynamic resource balancing, sharing of layers and inference/finetuning execution on the cluster. [The diagram](https://github.com/ggerganov/llama.cpp/blob/master/examples/rpc/README.md) in the RPC section of llama.cpp is one tiny step in that direction but much more careful design and engineering of the entire system will have to done to come up with the right APIs and implementation for all of this. However, there's a feeling that the overall approach is going to hinge around layers (likely referenced in a CAS-like fashion) the same way that [Apache Spark](https://www.analyticsvidhya.com/blog/2021/08/understanding-the-basics-of-apache-spark-rdd/) was hinged around RDDs and building a distributed graph of operations over them.

# Prior art in llama.cpp clusterization

To be fair, there are various attempts at different levels of layering to cluster llama.cpp engines together. Most of this is done for specifically LLM inference purposes only and it doesn't really get to the level of making different engines share their internal state in any meaningful form. Still, the following implementations may serve as useful sources of inspiration (or even some code re-use):
   
   * Distantmagic folks have a [stateful load balancer custom-tailored for llama.cpp](https://github.com/distantmagic/paddler) in general their approach to clustering is an interesting one -- and [their project cookbook](https://llmops-handbook.distantmagic.com/general-concepts/load-balancing/index.html) is a really good read.
   * olama could be considered a [forward proxy](https://llmops-handbook.distantmagic.com/general-concepts/load-balancing/forward-proxy.html) and a [supervisor](https://llmops-handbook.distantmagic.com/general-concepts/load-balancing/supervisor.html) for llama.cpp. It doesn't really cluster though, but see Distantmagic approach to that.
   * [kubeai](https://www.kubeai.org/) is attempting some level of clustering on K8s and tries to be agnostic enough to run llama.cpp alongside with vllm and others
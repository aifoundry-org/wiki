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

# Debugging/control flow

In order to appreciate the very basic control flow/setup your debugging session it is recommended that one starts with a very basic and small model. [BERT for embeddings](bits-n-bytes) happens to be one of those. Currently most of us are using VSCode so the following [launch.json](bits-n-bytes/launch.json) template could provide a good start to your debugging sessions. Note that because of the complexity of hooking up VSCode to the llama.cpp build system we're currently stubbing-out the build portion in launch.json (IOW you will have to build the project in your terminal or something by doing `make LLAMA_DEBUG=1`). To not be annoyed by VSCode's attempts to CMake-configure your project only to fail the following diff maybe useful (it will also allow you to have a much nicer interaction with a VSCode debugger targets later on):
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


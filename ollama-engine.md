# ollama Engine

How does the ollama engine work? What happens when you enter `ollama run`?

## Overview

ollama is essentially a client-server ease-of-use and ease-of-launch wrapper around
[`llama.cpp`](https://github.com/ggerganov/llama.cpp).

```
ollama client ----> ollama server ----> llama.cpp server
```

The ollama server exposes a REST API, which can be used directly, or via its convenient client.
The REST API is documented [here](https://github.com/ollama/ollama/blob/defbf9425af8228f3420d567e9eeaa29d8ac87e3/docs/api.md).

While `llama.cpp` must be given a local model and configuration to run, ollama handles making
models available, providing a more comprehensive REST API, managing multiple calls to the same
model, handling multiple models in parallel, including adapter layers, etc.

When a REST API request is received to prompt a model, or even just ensure that it is
loaded into memory, the server:

1. Makes sure the requested model is available locally
1. Locates the model in local storage and validates it
1. Finds an existing instance of the model running; if none available, schedules a runner for the model
1. The runner finds the appropriate configuration, including GPU devices
1. The runner extracts a ready-to-run binary for `llama.cpp` and launches it to listen on a provided port

## Flow Detail

When you run `ollama run <model>`, the command is handled via the [RunHandler](https://github.com/ollama/ollama/blob/defbf9425af8228f3420d567e9eeaa29d8ac87e3/cmd/cmd.go#L382), which sets up
the remote connection and makes calls to `/api/generate` to send requests, or `/api/chat` for an
ongoing chat.

Note that the model can be loaded or unloaded into memory by sending a request to `/api/generate`
with an empty body. A keep-alive of `0` causes the model to unload.

The options and formats for the `/api/generate` and `/api/chat` can be read in
[the API docs](https://github.com/ollama/ollama/blob/defbf9425af8228f3420d567e9eeaa29d8ac87e3/docs/api.md).
The rest of this document focuses on the server-side implementation, specifically the engine for
running models.

Upon receiving a request, the server:

1. Schedules a runner for the model
1. If there is a request, processes the request

Scheduling a runner is handled by [Server.scheduleRunner](https://github.com/ollama/ollama/blob/defbf9425af8228f3420d567e9eeaa29d8ac87e3/server/routes.go#L79-L109), which is a setup wrapper around [Scheduler.GetRunner](https://github.com/ollama/ollama/blob/defbf9425af8228f3420d567e9eeaa29d8ac87e3/server/sched.go#L80-L101).

ollama server keeps one copy of each model type loaded in memory. This is done via a map of
model to runner. When a new generate request arrives, the request, along with the model type,
is [passed to a `pending` channel](https://github.com/ollama/ollama/blob/defbf9425af8228f3420d567e9eeaa29d8ac87e3/server/sched.go#L96).
The [`processPending` infinite loop](https://github.com/ollama/ollama/blob/defbf9425af8228f3420d567e9eeaa29d8ac87e3/server/sched.go#L115-L311) sees the request,
and [looks for an existing runner for that model](https://github.com/ollama/ollama/blob/defbf9425af8228f3420d567e9eeaa29d8ac87e3/server/sched.go#L143).

If it finds one, it returns that runner. If it doesn't, it
[gets a reference to the model from its local storage](https://github.com/ollama/ollama/blob/defbf9425af8228f3420d567e9eeaa29d8ac87e3/server/sched.go#L190). It then uses the reference to the
model and the request to check capabilities for parallel models, GPUs and CPUs, and clears existing
models to make room, if needed. Finally, it [loads the model into memory](https://github.com/ollama/ollama/blob/defbf9425af8228f3420d567e9eeaa29d8ac87e3/server/sched.go#L223-L233) and returns the runner.

Model references are accessed via [LoadModel](https://github.com/ollama/ollama/blob/defbf9425af8228f3420d567e9eeaa29d8ac87e3/llm/server.go#L71-L84). `LoadModel` _only_ works with GGML
files; its function to understand the model actually is called [`DecodeGGML()`](https://github.com/ollama/ollama/blob/defbf9425af8228f3420d567e9eeaa29d8ac87e3/llm/ggml.go#L307-L348). This does _not_
load it into memory. Rather, it decodes the file, and returns a [`GGML` struct](https://github.com/ollama/ollama/blob/defbf9425af8228f3420d567e9eeaa29d8ac87e3/llm/ggml.go#L13),
which contains references to a [`model` interface](https://github.com/ollama/ollama/blob/defbf9425af8228f3420d567e9eeaa29d8ac87e3/llm/ggml.go#L18-L21) and a [`container` interface](https://github.com/ollama/ollama/blob/defbf9425af8228f3420d567e9eeaa29d8ac87e3/llm/ggml.go#L264-L267). These are used later to actually load the model into memory.
`LoadModel` solely validates that it is recognizable as a GGML file and gets a usable handle
to it for later loading.

The load into memory is handled via [`Schedule.load()`](https://github.com/ollama/ollama/blob/defbf9425af8228f3420d567e9eeaa29d8ac87e3/server/sched.go#L412-L471), which in turn
calls [`NewLlamaServer`](https://github.com/ollama/ollama/blob/defbf9425af8228f3420d567e9eeaa29d8ac87e3/llm/server.go#L86-L443) to create a
[`LlamaServer`](https://github.com/ollama/ollama/blob/defbf9425af8228f3420d567e9eeaa29d8ac87e3/llm/server.go#L34-L45).

A `LlamaServer` is a running instance of an LLM that can receive prompts and respond,
and is essentially a wrapper to `llama.cpp`. Its
interface is [defined](https://github.com/ollama/ollama/blob/defbf9425af8228f3420d567e9eeaa29d8ac87e3/llm/server.go#L34-L45):

```go
type LlamaServer interface {
	Ping(ctx context.Context) error
	WaitUntilRunning(ctx context.Context) error
	Completion(ctx context.Context, req CompletionRequest, fn func(CompletionResponse)) error
	Embedding(ctx context.Context, input string) ([]float32, error)
	Tokenize(ctx context.Context, content string) ([]int, error)
	Detokenize(ctx context.Context, tokens []int) (string, error)
	Close() error
	EstimatedVRAM() uint64 // Total VRAM across all GPUs
	EstimatedTotal() uint64
	EstimatedVRAMByGPU(gpuID string) uint64
}
```

The `llama.cpp` binary is built as part of building `ollama` and embedded into the `ollama`
binary. When a runner is launched, the binary is extracted from the embedded filesystem, placed in
a local temporary directory, and run, listening on a port.

For example, a running binary:

```
TMPDIR/runners/metal/ollama_llama_server --model ~/.ollama/models/blobs/sha256-6a0746a1ec1aef3e7ec53868f220ff6e389f6f8ef87a01d77c96807de94ca2aa --ctx-size 8192 --batch-size 512 --embedding --log-disable --n-gpu-layers 33 --verbose --parallel 4 --port 56457
```

When the server `GenerateHandler` has a handle to a runner with the model loaded,
it calls [`r.Completion()`](https://github.com/ollama/ollama/blob/defbf9425af8228f3420d567e9eeaa29d8ac87e3/server/routes.go#L246), passing it the prompt, format,
images and options as parameters, and a handler function for the response.

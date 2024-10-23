# ramalama

[ramalama](https://github.com/containers/ramalama) is a wrapper tool to:

1. Pull models down from multiple registry/repository types to local
1. Run models locally inside a container

## Registries & Transport

ramalama supports 3 registry protocols:

* HuggingFace Hub - git
* Ollama Hub - almost OCI compliant
* OCI - any OCI-compliant registry

By default, ramalama uses the ollama protocol for transport, i.e. assumes the image is being
pulled from ollama using their protocol. The default protocol can be changed via the
`RAMALAMA_TRANSPORTS` environment variable, and transport for an individual pull can be
set via a protocol, e.g. `huggingface://model`.

## Local Storage

All models are stored in a local cache directory. By default, that directory is:

* regular user: `~/.local/share/ramalama/`
* root user: `/var/lib/ramalama/`

In the storage directory, models are installed into their own directory.
For example:

* `oci://docker.io/foo/bar:1.25` -> `~/.local/share/ramalama/repos/oci/docker.io/foo/bar/1.25`
* `huggingface://julien-c/EsperBERTo-small` -> `~/.local/share/ramalama/repos/huggingface/julien-c/EsperBERTo-small`
* `ollama://llama3.1:7B` -> `~/.local/share/ramalama/repos/ollama/v2/llama3.1/7B`

The actual downloads are performed by exec'ing locally installed tools or a built-in library:

* OCI: depends on [omlmd](https://containers.github.io/omlmd/)
* HuggingFace: depends on [huggingface-cli](https://huggingface.co/docs/huggingface_hub/guides/cli)
* Ollama: depends on has [native code](https://github.com/containers/ramalama/blob/main/ramalama/ollama.py)

Note that there is no unified storage, and no deduplication. Even OCI is local to each directory.

## Runtime

ramalama supports running models using both `llama.cpp` (default) and `vllm`. Specific one can
be specified using `ramalama run --runtime <runtime>`.

The actual runtime is executed in one of three ways, in the following descending order of
priority. A specific option can :

1. Podman inside a container, id podman is available
1. Docker inside a container, if docker is available
1. Locally installed tools

When running inside a container, it uses `quay.io/ramalama/ramalama:latest` by default. It can be
overridden by the environment variable `RAMALAMA_IMAGE`.

## Build and dependencies

ramalama is a Python tool. It depends on:

1. Python tools installed
1. Python environment and dependencies, although basic Python tools like pip and pyenv can handle it
1. Local tools for OCI and HuggingFace
1. Docker, podman or locally installed llama.cpp or vLLM

## Shortcomings and issues

1. Python dependency. In general, for engineer-time work, this is sufficient. For production, it is a lot. A natively compiled single-binary with fewer dependencies is preferred.
1. It depends on externally installed tools, beyond a common set like a container runtime.
1. It stores images locally in the native format for each transport, with no content-addressability or deduplication.
1. It relies on a container image using `latest`, which is mutable.

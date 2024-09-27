# Ollama Storage Format

ollama is a single binary distribution, built to manage downloading, running and interfacing with multiple LLM models. It is distributed as a single binary.

Ollama is explicitly Docker-derived, in format, style, and distribution, and started by ex-Docker people.

## Model Locator (URI)

Ollama uses a Docker-style reference (like URI) for models. For example,
`ollama.com/library/llama3:8B` is a model reference. There also is a short
format of `llama3:8B`, which is equivalent to `ollama.com/library/llama3:8B`.

Based on this, it should be possible to reference models elsewhere, like
`myserver.com/models/smartmodel:1.0`. Initial checks have not shown it yet.

## Distribution Hub

Ollama primarily deals with models in their own registry, https://ollama.com/library/. There are
multiple models there. Just like Docker `alpine:3.20` translates as a default into
`docker.io/library/alpine:3.20`, so `llama3:8B` translates into `ollama.com/library/llama3:8B`.

There is code inside ollama that looks like it can override the registry, but it is not clear yet.

We should be able to pull down a model from almost any OCI-compliant registry, except that its
distributon spec - the network protocol - appears to be different. Initial experiments show failure
but there is more to be done.

Ideally, you would be able to do `docker pull ollama.com/library/llama3:8B` and pull it down,
and equally `ollama pull docker.io/library/alpine:3.20` and pull it down. It would fail to run,
as the manifest is missing container media-type elements, but that is a run-time, not a
distribution-time, issue.

## Distribution Spec

Still being determined. It appears to be similar to Docker, or more specifically,
[OCI Distribution Spec](https://github.com/opencontainers/distribution-spec/blob/main/spec.md),
but not 100% compatible.

## Local Storage

Local model storage after downloading (`pull`) is very similar to Docker, stored
in a local directory. The location is OS-dependent,  for example, in macOS it
is `~/.ollama/models/`.

The structure is under the directory is:

* `manifests/` which contains basic json manifests, which appear identical to OCI manifests
* `blobs/` which contain sha256 digested blobs

```sh
$ tree models
models
├── blobs
│   ├── sha256-3f8eb4da87fa7a3c9da615036b0dc418d31fef2a30b115ff33562588b32c691d
│   ├── sha256-4fa551d4f938f68b8c1e6afa9d28befb70e3f33f75d0753248d530364aeea40f
│   ├── sha256-577073ffcc6ce95b9981eacc77d1039568639e5638e83044994560d9ef82ce1b
│   ├── sha256-6a0746a1ec1aef3e7ec53868f220ff6e389f6f8ef87a01d77c96807de94ca2aa
│   └── sha256-8ab4849b038cf0abc5b1c9b8ee1443dca6b93a045c2272180d985126eb40bf6f
└── manifests
    └── registry.ollama.ai
        └── library
            └── llama3
                └── latest
```

And the file types:

```sh
$ file models/blobs/*
models/blobs/sha256-3f8eb4da87fa7a3c9da615036b0dc418d31fef2a30b115ff33562588b32c691d: JSON data
models/blobs/sha256-4fa551d4f938f68b8c1e6afa9d28befb70e3f33f75d0753248d530364aeea40f: Unicode text, UTF-8 text, with very long lines (711)
models/blobs/sha256-577073ffcc6ce95b9981eacc77d1039568639e5638e83044994560d9ef82ce1b: JSON data
models/blobs/sha256-6a0746a1ec1aef3e7ec53868f220ff6e389f6f8ef87a01d77c96807de94ca2aa: data
models/blobs/sha256-8ab4849b038cf0abc5b1c9b8ee1443dca6b93a045c2272180d985126eb40bf6f: ASCII text
```

### Manifests

The `manifests` directory contains JSON manifests for finding a model by referenced name.

Unlike OCI, rather than an `index.json`, it uses directories. So
`~/.ollama/models/manifests/registry.ollama.ai/library/llama3/latest` is the equivalent of `registry.ollama.ai/library/llama3:latest`.

The contents of a manifest actually are a `application/vnd.docker.distribution.manifest.v2+json`
manifest. Interestingly, it is a Docker v2 rather than OCI v1. They effectively are identical,
but the choice to go with `vnd.docker` is interesting.

For example, downloading llama3.1:8B has the following manifest at `~/.ollama/models/manifests/registry.ollama.ai/library/llama3/latest`:

```json
{
  "schemaVersion": 2,
  "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
  "config": {
    "mediaType": "application/vnd.docker.container.image.v1+json",
    "digest": "sha256:3f8eb4da87fa7a3c9da615036b0dc418d31fef2a30b115ff33562588b32c691d",
    "size": 485
  },
  "layers": [
    {
      "mediaType": "application/vnd.ollama.image.model",
      "digest": "sha256:6a0746a1ec1aef3e7ec53868f220ff6e389f6f8ef87a01d77c96807de94ca2aa",
      "size": 4661211424
    },
    {
      "mediaType": "application/vnd.ollama.image.license",
      "digest": "sha256:4fa551d4f938f68b8c1e6afa9d28befb70e3f33f75d0753248d530364aeea40f",
      "size": 12403
    },
    {
      "mediaType": "application/vnd.ollama.image.template",
      "digest": "sha256:8ab4849b038cf0abc5b1c9b8ee1443dca6b93a045c2272180d985126eb40bf6f",
      "size": 254
    },
    {
      "mediaType": "application/vnd.ollama.image.params",
      "digest": "sha256:577073ffcc6ce95b9981eacc77d1039568639e5638e83044994560d9ef82ce1b",
      "size": 110
    }
  ]
}
```

Note the main keys:

* `schemaVersion`
* `mediaType`, which actually is `application/vnd.docker.distribution.manifest.v2+json`
* `config`, which is a reference to the configuration file, and whose media type actually is a `application/vnd.docker.container.image.v1+json`, i.e. a docker image configuration.
* `layers`, which is a list of layers, each with a mediaType, digest, and size.

For example, here is the OCI manifest for `golang:1.23.1` for architecture `arm64`:

```json
{
	"schemaVersion": 2,
	"mediaType": "application/vnd.oci.image.manifest.v1+json",
	"config": {
		"mediaType": "application/vnd.oci.image.config.v1+json",
		"digest": "sha256:2737b855c4589a4564aba00d16fc897fa4b8489ed918ec317ff872dc988fbbfe",
		"size": 2856
	},
	"layers": [
		{
			"mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
			"digest": "sha256:56c9b9253ff98351db158cb6789848656b8d54f411c0037347bf2358efb18f39",
			"size": 49585623
		},
		{
			"mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
			"digest": "sha256:364d19f59f69474a80c53fc78da91f85553e16e8ba6a28063cbebf259821119e",
			"size": 23594279
		},
		{
			"mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
			"digest": "sha256:843b1d8321825bc8302752ae003026f13bd15c6eef2efe032f3ca1520c5bbc07",
			"size": 63997467
		},
		{
			"mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
			"digest": "sha256:ecb27c98d5b9e78892d876693427ae0a01e3113b36989718360a5aa9e319fd80",
			"size": 86293965
		},
		{
			"mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
			"digest": "sha256:a355a3cac949bed5cda9c62103ceb0f004727cedcd2a17d7c9836aea1a452fda",
			"size": 70624628
		},
		{
			"mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
			"digest": "sha256:83f1399aa9166438efeea4696812a3fa3f3397ff114492577a919f9b09f3c1ea",
			"size": 126
		},
		{
			"mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
			"digest": "sha256:4f4fb700ef54461cfa02571ae0db9a0dc1e0cdb5577484a6d75e68dc38e8acc1",
			"size": 32
		}
	]
}
```

#### Layers

The layers in the ollama manifest are not typical layers in the OCI sense, where they usually
represent actual "layers", all of the same type, to be layered one on top of the lower one.

In the golang OCI image referenced above, the layers are
`application/vnd.oci.image.layer.v1.tar+gzip`.

In ollama, each layer is a different media type, representing a different part of the model:

* Model
* Adapter
* License
* Template
* Params

This structure is similar to OCI Artifacts. The original [OCI Artifacts spec](https://github.com/opencontainers/artifacts) was a proposal, and was mostly adopted into the
[OCI distribution spec](https://github.com/opencontainers/distribution-spec/blob/main/spec.md#enabling-the-referrers-api)
and [OCI Image Spec](https://github.com/opencontainers/image-spec/blob/main/manifest.md#guidelines-for-artifact-usage).

However, it does not quite hew to the OCI Artifacts spec, which requires the usage of actual
media-types and an `artifactType` root-level key.

##### Model and Adapter Layers

The `Model` and `Adapter` layers are the actual model and adapter files, respectively. There
can be 0, 1 or more adapter layers, but should be only one model layer.

For example:

```json
  "layers": [
    {
      "mediaType": "application/vnd.ollama.image.model",
      "digest": "sha256:6a0746a1ec1aef3e7ec53868f220ff6e389f6f8ef87a01d77c96807de94ca2aa",
      "size": 4661211424
    },
    {
      "mediaType": "application/vnd.ollama.image.adapter",
      "digest": "sha256:0427fbda54677091b24fe62a2e1e4d2c475930075a0ef26fbf85609ac7c70539",
      "size": 319876032
    },
```

#### Config

The configuration file, although referenced in the manifest as a `vnd.docker.container.image.v1+json` media type, actually does not match the spec. For example:

```json
{
  "model_format": "gguf",
  "model_family": "llama",
  "model_families": [
    "llama"
  ],
  "model_type": "8.0B",
  "file_type": "Q4_0",
  "architecture": "amd64",
  "os": "linux",
  "rootfs": {
    "type": "layers",
    "diff_ids": [
      "sha256:6a0746a1ec1aef3e7ec53868f220ff6e389f6f8ef87a01d77c96807de94ca2aa",
      "sha256:4fa551d4f938f68b8c1e6afa9d28befb70e3f33f75d0753248d530364aeea40f",
      "sha256:8ab4849b038cf0abc5b1c9b8ee1443dca6b93a045c2272180d985126eb40bf6f",
      "sha256:577073ffcc6ce95b9981eacc77d1039568639e5638e83044994560d9ef82ce1b"
    ]
  }
}
```

### Blobs

Just like OCI, each of the layers and config - in the above example, 4 layers and 1 config -
has a corresponding blob in `~/.ollama/models/blobs/`. The actual filename and content digest is
provided in the `digest` field, which includes the digest algorithm, like `sha256`, and actual
digest.

This is nearly identical to docker and more specifically
[OCI image layout v1](https://github.com/opencontainers/image-spec/blob/main/image-layout.md).

The differences are:

* storage of the manifests in `manifests` rather than in the `index.json` file.
* usage of filenames as `blobs/sha256-<digest>` rather than `blobs/sha256/<digest>`

## Model Creation

Ollama supports creating new models using `Modelfile` in a directory and then running
`ollama create`.

For example:

```Dockerfile
FROM basemodel

# make some changes
```

Run `ollama create`.

This creates the new model, and then saves it with its layers
and config, in the local ollama server. It then can be pushed to a registry with `ollama push`.
This is very similar to docker.

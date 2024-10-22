# Huggingface Storage Formats

NOTE: HuggingFace storage format mostly is documented reasonably well in
the [huggingface hub docs](https://huggingface.co/docs/huggingface_hub/).
This is intended to provide easy access and focus on the parts that matter to Ainekko,
which are secondary to people interacting with huggingface, most of whom care about the APIs
for interacting with models.

Of particular importance are the following docs:


* [Downloading Files](https://huggingface.co/docs/huggingface_hub/v0.26.0/en/package_reference/file_download#huggingface_hub.hf_hub_download) - in particular, the section about local storage.
* [HuggingFace API HfApi library](https://huggingface.co/docs/huggingface_hub/v0.26.0/en/package_reference/hf_api#huggingface_hub.HfApi)
* [Git vs HTTP Paradigm](https://huggingface.co/docs/huggingface_hub/v0.26.0/en/concepts/git_vs_http) - this is important to understand the difference between the two ways of accessing models.

## Hub Storage

HuggingFace Hub ("Hub") storage is primarily git, with heavy usage of [GitLFS](https://git-lfs.com)
to manage the very large files required for models.

Since git works with file digests, this is storage that can track any individual file changes.
Internally on the server, and when cloning, it uses digests to recognize that files are identical
and efficiently copy them.

## API

There are two interfaces to the storage, the git interface and the HTTP interface.
The git interface functions exactly as git normally would. Each model is stored locally
as a git repository.

The HTTP interface is newer, and is intended to replace the git API. It still uses git
for server-side storage, but the client does not require the challenges of cloning,
and the issues related to loss of sync.

## Local Storage

If the git API is used, then local storage is just git. As described above, this is the older
API, with HTTP being encouraged for usage.

If the HTTP API is used, the huggingface library stores models in a local cache directory.
By default, that directory is `~/.cache/huggingface/`, but it can be changed via parameter.

Within the cache directory, the structure is based upon repository. The following is a typical
sample:

```
[  96]  .
└── [ 160]  models--julien-c--EsperBERTo-small
    ├── [ 160]  blobs
    │   ├── [321M]  403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd
    │   ├── [ 398]  7cb18dc9bafbfcf74629a4b760af1b160957a83e
    │   └── [1.4K]  d7edf6bd2a681fb0175f7735299831ee1b22b812
    ├── [  96]  refs
    │   └── [  40]  main
    └── [ 128]  snapshots
        ├── [ 128]  2439f60ef33a0d46d85da5001d52aeda5b00ce9f
        │   ├── [  52]  README.md -> ../../blobs/d7edf6bd2a681fb0175f7735299831ee1b22b812
        │   └── [  76]  pytorch_model.bin -> ../../blobs/403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd
        └── [ 128]  bbc77c8132af1cc5cf678da3f1ddf2de43606d48
            ├── [  52]  README.md -> ../../blobs/7cb18dc9bafbfcf74629a4b760af1b160957a83e
            └── [  76]  pytorch_model.bin -> ../../blobs/403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd
```

A root directory exists for each model, with subdirectories for blobs, refs, and snapshots.
This is very similar to the structure of a bare, or server-side, git repository:

* `blobs` contain the actual content
* `refs` contain the references to the content
* `snapshots` contain the actual files, with links to the actual blobs for each file

## Comparing with OCI

git uses digested content to optimize storage and transfer. However, it has a few challenges:

* It assumes the need to access files by traditional name and path. This makes it necessary to keep alternate file names.
* It is not optimized for large files, and can be slow to clone.

OCI, on the other hand, is excellent for large files, served from any http server.
It works very well for anything that has a root - manifest - and can be addressed structurally,
as layers or blobs, as opposed to arbitrary files.

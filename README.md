# wiki
Place for all our docs and whitepapers

```mermaid
flowchart LR
wiki[Wiki]
arch[Architecture]
hw[Hardware]
sw[Software]
wp[Wrappers]
bb[Backbones]
llcp[LLama.cpp]
olm[Olama]
hf[Huggingface]
rlm[Ramalama]

wiki --> hw
wiki --> sw
wiki --> arch
sw   --> bb
sw   --> wp
bb   --> llcp
bb   --> olm
bb   --> hf
wp   --> rlm

click hw "https://github.com/aifoundry-org/wiki/tree/refactor/hardware"
click llcp "https://github.com/aifoundry-org/wiki/tree/refactor/llamacpp"
click rlm "https://github.com/aifoundry-org/wiki/blob/refactor/ramalama.md"
click olm "https://github.com/aifoundry-org/wiki/tree/refactor/olama"
click hf "https://github.com/aifoundry-org/wiki/tree/refactor/huggingface"
click arch "https://github.com/aifoundry-org/wiki/tree/refactor/architecture"
```

# OptShapeGen shape encoder adaptation

Upstream: https://github.com/NeuralCarver/Michelangelo
Fork: https://github.com/YangYunjia/Michelangelo
Branch: `optshapegen-shape-encoder`.
Base: `6d83b0bacef92715dd5179d45647ed9a3d39bc95`.

The original upstream source tree and Git history are retained. This branch adds
a standalone encoder-only entry in the repository root: `AlignedShapeEncoder`
from `__init__.py`, backed by `encoder.py`, `embedder.py`, `transformer_blocks.py`
and `checkpoint.py`. It avoids importing the optional full application or
constructing the unused VAE and geometry decoders.

The encoder implementation was extracted from the Michelangelo copy in the
MeshAnything v2 vendor at revision `461d3b6ed750ab3443281b2e4a0e30e8ee98097e`.
Its local changes are relative imports, concise docstrings and the lightweight
encoder-only wrapper. Existing encoder parameter names and numerical operations
are retained. The accompanying parent-vendor license notice is preserved in
`LICENSE.meshanything.txt`; the original upstream `LICENSE` remains authoritative
for its source tree. Exact extraction hashes are in `OPTSHAPEGEN_SOURCE.json`.

Input: normalized xyz/normal clouds `(B,N,6)`.
Output: global shape token followed by 256 local tokens `(B,257,768)`.
Only PyTorch and einops are required by this standalone entry.

OptShapeGen loads the released `model.shape_model.encoder.*` checkpoint branch
through `experiments/common/shape_encoder.py`, using strict key loading, frozen
parameters and evaluation mode. No target/frame/sampling policy lives here.

The released-weight comparison used 4096-point inputs and produced bit-identical
fp32, bfloat16 and saved float16 features before/after extraction. Git commits on
this branch, rather than a copied source archive in the main project, track the
adaptation. Runtime imports from `third_party/michelangelo` remain unchanged.

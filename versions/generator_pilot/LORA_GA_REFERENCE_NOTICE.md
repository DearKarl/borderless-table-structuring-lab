# LoRA-GA implementation reference

The gradient factor initialization in `large_adapters.py` follows
[Outsider565/LoRA-GA](https://github.com/Outsider565/LoRA-GA), commit
`c4cd5372c75b290924214b348008891f744512ef`, specifically ArB2r direction and stable
scaling in `peft/src/peft/tuners/lora/layer.py`.

The upstream Apache 2.0 license is preserved in `LORA_GA_REFERENCE_LICENSE.txt`.
The explicit low-rank compensation wrapper, input mapping and experiment harness
are local implementations. Their numerical execution differs from storing rounded
residual BF16 base weights; see the campaign methods for this distinction.

Reference: Wang et al., *LoRA-GA: Low-Rank Adaptation with Gradient Approximation*,
[arXiv:2407.05000](https://arxiv.org/abs/2407.05000).

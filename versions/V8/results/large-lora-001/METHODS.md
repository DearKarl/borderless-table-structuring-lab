# V8.1–V8.3 methods

V8.1 trains ordinary rank-8 attention LoRA and the visual merger. V8.2 uses
LoRA-GA gradient initialization with the same trainable components. V8.3 adds
a training-only cell-location head to the V8.2 method.

All methods use 20,000 tables from 11,217 TRAIN sources, 256 DEV sources and
512 confirmation sources. Inputs are predicted native table crops at PDF 200 DPI.
Each run completes 3,750 updates and 60,000 exposures; DEV TEDS selects its checkpoint.

The attention adapters have 2,293,760 parameters with rank 8, alpha 16 and dropout
0.05. The visual merger has 31,464,704 parameters. V8.3's 4,100-parameter cell head
adds normalized box L1 with weight 0.1 to assistant-token generation cross entropy.
The remaining pretrained parameters remain frozen. The head is unused in generation.

LoRA-GA uses a shared 256-example TRAIN gradient calibration, ArB2r initialization
and initial-factor compensation. [Original implementation](https://github.com/Outsider565/LoRA-GA)
and [paper](https://arxiv.org/abs/2407.05000) provide attribution.

[Parameters and source identities](EXECUTION_SPECIFICATION.json) and
[checkpoint/package identities](MODEL_PACKAGES.json) accompany the reproduction
[source guide](../../source/generator_pilot/LARGE_CAMPAIGN_RUNTIME.md).
Runtime bindings, input payloads and original pretrained assets are separate prerequisites.

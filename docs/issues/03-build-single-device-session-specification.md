# 03: Build a Single-Device Session Specification
**Status:** ready-for-agent
**User stories covered:** 10-18
## What to build
From validated toolchain and model inputs, detect the system topology. CPU-only mode is permitted on any host, including one with accelerators present. If acceleration is chosen and multiple accelerators are present, exactly one accelerator must be selected explicitly. No implicit accelerator selection and no multi-GPU split are allowed. Build and display a Session Specification containing the selected device, the target total n_ctx decomposition, a Resource Budget, and a context advisory showing a likely-fit range and an uncertain maximum based on model metadata, training context, KV cache cost, main-model footprint, selected hardware, and conservative headroom. Require explicit user acknowledgement when the requested context exceeds the known training context or when the training context is unknown or internally conflicted. On discrete-GPU systems separate RAM and VRAM budgets; on unified-memory systems produce one non-double-counted budget. Default budgets are visible, configurable, fingerprinted, and include headroom. The advisory must state clearly that it is an estimate, not a tested guarantee. Integrate through CLI setup with review and confirmation. Tests must cover CPU-only on a multi-GPU host, CPU-only without accelerator, one GPU selected from multiple, multiple GPU selection error, unified memory, and invalid context composition rejection.
## Acceptance criteria
- [ ] System topology is detected: CPU, memory, accelerator count and type, unified vs discrete memory
- [ ] CPU-only mode is allowed on any host, including a multi-GPU host, when explicitly requested
- [ ] If acceleration is chosen and multiple accelerators are present, exactly one must be selected explicitly; selecting none or more than one produces an error
- [ ] The Session Specification includes the selected device, target n_ctx with prompt/generation/depth summing to the requested total, Resource Budget, and context advisory
- [ ] The context advisory shows a likely-fit range and an uncertain maximum, sourced from model metadata, KV cache cost estimates, model footprints, and hardware topology
- [ ] Advisory text states it is an estimate and not a tested guarantee
- [ ] Requesting context above the model's training context requires a warning and explicit acknowledgement
- [ ] When the model's training context is unknown or internally conflicted, the context is labelled unknown and the same acknowledgement is required
- [ ] Discrete-GPU systems track system RAM and selected-device VRAM as separate budgets
- [ ] Unified-memory systems track a single shared budget
- [ ] Default budgets include headroom, are visible before execution, are configurable, and are fingerprinted in the journal
- [ ] Tests cover CPU-only on a multi-GPU host, CPU-only without accelerator, one GPU selected from multiple, multiple GPU selection error, unified-memory topology, invalid n_ctx composition, and advisory range estimates
- [ ] README documents the Session Specification, resource budget defaults, and context advisory meaning
## Blocked by
- [02-validate-local-toolchain-and-model.md](02-validate-local-toolchain-and-model.md)

# Perception engine

Baseline ladder, in order. Each evaluated on the same held-out split
before the next is built. No self-supervised pretraining until the
supervised ladder has real comparable numbers.

1. Baseline 1 — 2D CNN on representative slices
2. Baseline 2 — 2.5D model (stacked adjacent slices)
3. Baseline 3 — 3D CNN (full volumetric context)
4. Baseline 4 — 3D Vision Transformer
5. Advanced — Self-supervised pretrained 3D foundation model, fine-tuned

Do not skip steps. The ablation table is the point, not the final number.

# 전체 실험의 혼동행렬 비교

동일 Gold58 OOF, seed42, 임계값 0.5. AUC는 3개 seed 평균. 총 판정수 696 = 58검사 × 12소견.

[두 실험 나란히 비교하는 HTML](confusion_comparison.html) · [소견별 부모 대비 변화](parent_category_deltas.csv)

| 실험 | AUC | TN | FP | FN | TP | micro 재현율 | micro 정밀도 |
|---|---:|---:|---:|---:|---:|---:|---:|
| [e00_resnet18_3](confusion_e00_resnet18_3.png) | 0.5907 | 248 | 208 | 87 | 153 | 63.7% | 42.4% |
| [e01_resnet18_5](confusion_e01_resnet18_5.png) | 0.6068 | 248 | 208 | 88 | 152 | 63.3% | 42.2% |
| [e02a_dinov2_5](confusion_e02a_dinov2_5.png) | 0.5628 | 240 | 216 | 102 | 138 | 57.5% | 39.0% |
| [e02b_swin_5](confusion_e02b_swin_5.png) | 0.6196 | 252 | 204 | 73 | 167 | 69.6% | 45.0% |
| [e02c_convnext_5](confusion_e02c_convnext_5.png) | 0.5975 | 253 | 203 | 93 | 147 | 61.3% | 42.0% |
| [e02d_vit_5](confusion_e02d_vit_5.png) | 0.5930 | 232 | 224 | 80 | 160 | 66.7% | 41.7% |
| [e03_attention](confusion_e03_attention.png) | 0.5836 | 274 | 182 | 98 | 142 | 59.2% | 43.8% |
| [e04_position](confusion_e04_position.png) | 0.5755 | 305 | 151 | 119 | 121 | 50.4% | 44.5% |
| [e05_plane](confusion_e05_plane.png) | 0.5753 | 305 | 151 | 119 | 121 | 50.4% | 44.5% |
| [e06_gold_control](confusion_e06_gold_control.png) | 0.5860 | 353 | 103 | 124 | 116 | 48.3% | 53.0% |
| [e07_weak_labels](confusion_e07_weak_labels.png) | 0.7419 | 327 | 129 | 83 | 157 | 65.4% | 54.9% |
| [e08_triplet](confusion_e08_triplet.png) | 0.7429 | 318 | 138 | 84 | 156 | 65.0% | 53.1% |
| [e09_protocols](confusion_e09_protocols.png) | 0.7441 | 312 | 144 | 75 | 165 | 68.8% | 53.4% |
| [e10_class_mil](confusion_e10_class_mil.png) | 0.7131 | 321 | 135 | 89 | 151 | 62.9% | 52.8% |
| [e11_finetune](confusion_e11_finetune.png) | 0.7139 | 320 | 136 | 89 | 151 | 62.9% | 52.6% |
| [e12_ensemble](confusion_e12_ensemble.png) | 0.7197 | 319 | 137 | 88 | 152 | 63.3% | 52.6% |
| [r01_e09_s13_blend](confusion_r01_e09_s13_blend.png) | 0.7714 | 324 | 132 | 67 | 173 | 72.1% | 56.7% |
| [r04_confidence](confusion_r04_confidence.png) | 0.7400 | 316 | 140 | 78 | 162 | 67.5% | 53.6% |
| [r05_label_loss](confusion_r05_label_loss.png) | 0.7350 | 294 | 162 | 63 | 177 | 73.8% | 52.2% |
| [s00_triplet](confusion_s00_triplet.png) | 0.7195 | 314 | 142 | 78 | 162 | 67.5% | 53.3% |
| [s01_slice3](confusion_s01_slice3.png) | 0.7166 | 315 | 141 | 81 | 159 | 66.2% | 53.0% |
| [s02_slice5](confusion_s02_slice5.png) | 0.7325 | 325 | 131 | 78 | 162 | 67.5% | 55.3% |
| [s03_slice10](confusion_s03_slice10.png) | 0.7240 | 319 | 137 | 86 | 154 | 64.2% | 52.9% |
| [s04_sequence](confusion_s04_sequence.png) | 0.7266 | 318 | 138 | 75 | 165 | 68.8% | 54.5% |
| [s05_position](confusion_s05_position.png) | 0.7131 | 310 | 146 | 78 | 162 | 67.5% | 52.6% |
| [s06_rope_index](confusion_s06_rope_index.png) | 0.7215 | 321 | 135 | 90 | 150 | 62.5% | 52.6% |
| [s07_rope_mm](confusion_s07_rope_mm.png) | 0.7118 | 298 | 158 | 85 | 155 | 64.6% | 49.5% |
| [s08_plane](confusion_s08_plane.png) | 0.7260 | 320 | 136 | 81 | 159 | 66.2% | 53.9% |
| [s09_mlp](confusion_s09_mlp.png) | 0.7243 | 283 | 173 | 65 | 175 | 72.9% | 50.3% |
| [s10_gate](confusion_s10_gate.png) | 0.7400 | 316 | 140 | 78 | 162 | 67.5% | 53.6% |
| [s11_cross](confusion_s11_cross.png) | 0.7364 | 312 | 144 | 78 | 162 | 67.5% | 52.9% |
| [s12_finetune](confusion_s12_finetune.png) | 0.7189 | 307 | 149 | 76 | 164 | 68.3% | 52.4% |
| [s13_ensemble](confusion_s13_ensemble.png) | 0.7484 | 312 | 144 | 75 | 165 | 68.8% | 53.4% |

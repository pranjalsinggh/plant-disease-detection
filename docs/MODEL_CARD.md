# Model card

## Model and intended use

Plant Disease Detection is a supervised RGB image classifier trained from scratch with four convolutional blocks. Its reference taxonomy contains 38 crop/condition combinations, including healthy classes. It predicts an image-level category, without locating or segmenting lesions.

The intended use is education, experimentation, and evaluation of leaf-image classification. A trained checkpoint must be created before making predictions. No pretrained weights are provided, and the repository itself has no measured plant-disease accuracy.

## Data and preprocessing

The folder-based dataset is indexed before training. The index records actual class names and split counts, image dimensions and sizes, and SHA-256 content hashes. The default full-data workflow expects 38 class folders; the reference taxonomy supplies canonical names, while custom class names are supported. Training class order is included in the checkpoint; inference uses that order, not an independent hard-coded mapping.

Inputs are converted to RGB, resized to the configured square size, and normalized to `[-1, 1]`. Training adds random augmentation. Evaluation and inference use deterministic preprocessing. A leaf's orientation metadata is respected during image loading.

## Evaluation protocol

1. Establish an independent test split before training. Group photographs of the same leaf and their augmented variants into one split.
2. Index and review all image counts. Investigate skipped files and duplicate content. Hash checks catch exact duplicates only.
3. Train using `train`; select the best checkpoint using `valid`. Record configuration, dataset source/version, and random seed.
4. Evaluate the fixed checkpoint on held-out `test`. Inspect overall accuracy, class precision/recall/F1, and the confusion matrix. Do not treat checkpoint-selection accuracy as independent test performance.
5. Measure field-image performance separately, including unfamiliar crops, non-leaf images, and disease categories outside the taxonomy.

Seeded runs support repeatability on the same software and hardware, but do not promise bit-for-bit equivalence between CPU/GPU platforms or library versions. The dataset can change on disk; do not modify images after indexing.

## Reported performance

The project brief supplied 95.94% training accuracy, 94.03% validation accuracy, and training loss decreasing from 1.8891 to 0.1929 on approximately 79,000 training images. Those historical figures were not supplied with supporting artifacts and have not been reproduced. They are documented in `reference_results.json`, separate from actual run output.

Synthetic test fixtures verify indexing, training, checkpoint serialization, and inference. Their scores are not plant-disease performance measurements. The 100,000-entry benchmark verifies synthetic SQLite metadata capacity only.

## Limitations

- Controlled-background images can differ substantially from field images, lighting, camera quality, and occlusion.
- Healthy/disease predictions apply only to the training taxonomy. Multiple simultaneous conditions are outside this single-label model's design.
- The model does not reject out-of-distribution images reliably. A high softmax score is not proof that an image contains a known disease.
- Low-score warnings in the app use an illustrative 0.65 threshold, without calibration or validated clinical/agronomic meaning.
- Correct diagnosis may require symptom history and independent examination. Predictions alone do not determine treatment.

## Artifacts and licensing

The checkpoint stores model tensors, preprocessing parameters, ordered classes, and epoch metrics. It is loaded with PyTorch's restricted `weights_only=True` mode. Load only trusted local checkpoints.

The software is MIT licensed. Dataset terms, attribution requirements, and externally supplied checkpoint licenses must be checked independently before redistribution.

# Sea Metric Classifier

Lightweight alternative to Qwen SFT for the 21-class Sea problem.

It freezes `google/siglip2-base-patch16-224`, extracts full-image and bbox-crop
features, and trains a small head with cross-entropy plus supervised contrastive
loss. The validation checkpoint is selected by macro-F1.

The real-test Parquet is intentionally not used by the training script.

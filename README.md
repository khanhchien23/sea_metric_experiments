# Sea Metric Classifier

Lightweight alternative to Qwen SFT for the 21-class Sea problem.

It freezes `google/siglip2-base-patch16-224` and trains a small head with
cross-entropy plus supervised contrastive loss. The published Sea train and
validation Parquet files contain images that were already cropped by bbox;
their bbox values refer to the original 640x640 image and must **not** be
applied again. The corrected baseline therefore uses the stored crop as the
first view and the same image as the second view. Real-test images are full
photographs, so inference uses the same full-image + same-image representation.

The real-test Parquet is intentionally not used by the training script.

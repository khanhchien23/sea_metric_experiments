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

## Higher-accuracy fine-tuning pipeline

The frozen-feature baseline is not the recommended final model for the
synthetic-to-real gap. Use the original 640x640 images and fine-tune the
vision tower:

1. `pack_sea_full_parquet.py` stores both original full images and bbox crops.
2. `train_sea_finetune.py` fine-tunes SigLIP2 with a smaller backbone learning
   rate, strong but moderate augmentation, full-image loss, and auxiliary crop
   loss.
3. `infer_sea_finetune.py` evaluates real images using the full-image branch.

The real Parquet remains evaluation-only. DINOv2 can be trained with the same
trainer (`--kind dinov2`) and ensembled only after the SigLIP2 fine-tuned
baseline is measured.

### Strict synthetic-to-real protocol

This pipeline intentionally does **not** use `selected_sea_only_real.parquet`
for training, augmentation statistics, hyperparameter search, checkpoint
selection, or early stopping. The only real-data operation is the final
evaluation. Synthetic validation metrics are reported separately because they
are not expected to predict real-image accuracy perfectly.

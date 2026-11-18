# Sea Metric Classifier

## Frozen retrieval baseline (no fine-tuning)

`sea_retrieval.py` embeds each synthetic TRAIN crop once with pretrained
`facebook/dinov2-base` (CLS token, L2 normalization). Each query is embedded
with the exact same encoder/processor; exact cosine nearest-neighbor search
returns the synthetic neighbor's label as the prediction. Similarity is not
a probability. No classifier head or fine-tuned checkpoint is used.

```bash
python sea_retrieval.py build --data KhanhChien/Data-Sea-Full-Synthetic --output /kaggle/working/sea_retrieval/gallery.pt --batch_size 16
python sea_retrieval.py predict --gallery /kaggle/working/sea_retrieval/gallery.pt --data KhanhChien/Data-Sea-Full-Synthetic --filename data/validation-full-sea.parquet --column crop_image --output /kaggle/working/sea_retrieval/validation.csv
python sea_retrieval.py predict --gallery /kaggle/working/sea_retrieval/gallery.pt --data KhanhChien/Data-Qwen3.5-SFT-Sea-Only --output /kaggle/working/sea_retrieval/real_predictions.csv
```

The commands download only the requested Parquet file. Outputs include CSV
predictions with top-5 gallery IDs/labels/similarities, metrics, per-class
reports and a confusion matrix. Query ground truth is optional and is used
only for metrics, never for search. Validation and real images are not added
to the gallery. The build command rejects non-training filenames; ensure
the supplied training data really is synthetic (a filename is not proof of
provenance).

Default prediction is nearest neighbor, fixed before real evaluation. An
optional `build --rule class_topk --class_k 3` stores a class-balanced scoring
rule (mean of each class's top-k similarities). Compare choices only on
synthetic validation, not by selecting the highest real-test score.

This is one vector **per image**, not one vector for the whole dataset.
The native DINOv2 processor may center-crop wide images; this baseline does
not fix object localization or synthetic-to-real domain shift. Synthetic
validation and real-test metrics must be reported separately.

Offline tests: `python -m unittest test_sea_retrieval -v`.

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
4. `ensemble_sea_finetune.py` can combine independently fine-tuned SigLIP2
   and DINOv2 models; its weight is selected on synthetic validation only.

The real Parquet remains evaluation-only. DINOv2 can be trained with the same
trainer (`--kind dinov2`) and ensembled only after the SigLIP2 fine-tuned
baseline is measured.

### Strict synthetic-to-real protocol

This pipeline intentionally does **not** use `selected_sea_only_real.parquet`
for training, augmentation statistics, hyperparameter search, checkpoint
selection, or early stopping. The only real-data operation is the final
evaluation. Synthetic validation metrics are reported separately because they
are not expected to predict real-image accuracy perfectly.

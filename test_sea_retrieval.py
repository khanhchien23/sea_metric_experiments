"""Offline tests: no model downloads and no real-test data."""

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image

import sea_retrieval as retrieval


class RetrievalTests(unittest.TestCase):
    def test_nearest_and_normalization(self):
        gallery = torch.tensor([[10., 0.], [0., 2.], [1., 0.]])
        query = torch.tensor([[0., 20.], [3., 0.]])
        pred, sim = retrieval.search_embeddings(query, gallery, ["A", "B", "A"], ["A", "B"])
        self.assertEqual(pred.tolist(), [1, 0])
        self.assertTrue(torch.allclose(sim.max(dim=1).values, torch.ones(2)))

    def test_class_topk_small_class(self):
        pred, _ = retrieval.search_embeddings(
            torch.tensor([[1., 0.]]), torch.tensor([[1., 0.], [0., 1.]]),
            ["A", "B"], ["A", "B"], "class_topk", 3,
        )
        self.assertEqual(pred.item(), 0)

    def test_missing_local_path(self):
        with self.assertRaises(FileNotFoundError):
            retrieval.resolve_parquet("/missing/sea-retrieval-input", "data/train.parquet")

    def test_cli_build_predict_offline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            buffer = io.BytesIO()
            Image.new("RGB", (32, 32)).save(buffer, format="PNG")
            image = {"bytes": buffer.getvalue(), "path": None}
            train = root / "train-full-sea.parquet"
            query_path = root / "validation-full-sea.parquet"
            pq.write_table(pa.Table.from_pylist([
                {"id": "t1", "file_name": "train1.png", "crop_image": image, "gt_class": "Sea.A"},
                {"id": "t2", "file_name": "train2.png", "crop_image": image, "gt_class": "Sea.B"},
            ]), train)
            pq.write_table(pa.Table.from_pylist([
                {"id": "q1", "file_name": "query.png", "image_bytes": buffer.getvalue(), "gt_class": "Sea.B"},
            ]), query_path)
            from types import SimpleNamespace
            fake_model = SimpleNamespace(config=SimpleNamespace(_commit_hash="fake-revision"))
            gallery_path, output = root / "gallery.pt", root / "prediction.csv"
            with patch.object(retrieval, "load_encoder", return_value=(None, fake_model, "cpu")), patch.object(
                retrieval, "extract", side_effect=[torch.eye(2), torch.tensor([[0., 1.]])]
            ):
                retrieval.build(SimpleNamespace(
                    data=str(train), filename="data/train-full-sea.parquet", column="crop_image",
                    model_id="fake", batch_size=2, output=str(gallery_path), rule="nearest", class_k=3,
                ))
                retrieval.predict(SimpleNamespace(
                    gallery=str(gallery_path), data=str(query_path), filename="unused", column=None,
                    batch_size=2, output=str(output), neighbors=5,
                ))
            report = json.loads(output.with_suffix(".metrics.json").read_text())
            self.assertEqual(report["accuracy"], 1.)
            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["prediction"], "Sea.B")
            self.assertEqual(len(json.loads(rows[0]["neighbors"])), 2)


if __name__ == "__main__":
    unittest.main()

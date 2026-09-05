import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from learning_llm.data import EncodedTextDataset
from learning_llm.model import DecoderLanguageModel, ModelConfig
from learning_llm.scripts.train_tinystories_model import (
    _default_encoded_cache_path,
    _load_or_encode_token_ids,
    _model_dimensions_from_args,
    _planned_training_steps,
    _validate_run_size,
)
from learning_llm.training import TrainingConfig, train


class TrainingLoopTests(unittest.TestCase):
    def test_training_saves_epoch_and_final_checkpoints(self):
        torch.manual_seed(0)
        dataset = EncodedTextDataset(torch.arange(24), context_length=4)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        model = DecoderLanguageModel(
            ModelConfig(
                vocab_size=32,
                context_length=4,
                d_model=8,
                n_head=2,
                n_layer=1,
                dropout=0.0,
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "tiny.pt"
            result = train(
                model,
                loader,
                loader,
                TrainingConfig(
                    batch_size=4,
                    epochs=1,
                    max_steps=None,
                    eval_interval=10,
                    eval_batches=1,
                    checkpoint_path=checkpoint_path,
                    show_progress=False,
                ),
            )

            self.assertEqual(result.epochs, 1)
            self.assertTrue(checkpoint_path.exists())
            self.assertTrue((Path(directory) / "tiny-epoch-0001.pt").exists())

    def test_gpt2_small_preset_uses_reference_architecture(self):
        args = Namespace(
            model_preset="gpt2-small",
            context_length=None,
            d_model=None,
            n_head=None,
            n_layer=None,
            dropout=None,
        )

        dimensions = _model_dimensions_from_args(args)

        self.assertEqual(dimensions["context_length"], 1_024)
        self.assertEqual(dimensions["d_model"], 768)
        self.assertEqual(dimensions["n_head"], 12)
        self.assertEqual(dimensions["n_layer"], 12)

    def test_explicit_dimension_flags_override_preset(self):
        args = Namespace(
            model_preset="gpt2-small",
            context_length=128,
            d_model=256,
            n_head=8,
            n_layer=6,
            dropout=0.0,
        )

        dimensions = _model_dimensions_from_args(args)

        self.assertEqual(dimensions["context_length"], 128)
        self.assertEqual(dimensions["d_model"], 256)
        self.assertEqual(dimensions["n_head"], 8)
        self.assertEqual(dimensions["n_layer"], 6)
        self.assertEqual(dimensions["dropout"], 0.0)

    def test_planned_training_steps_respects_max_steps(self):
        planned_steps = _planned_training_steps(
            train_windows=1_000,
            batch_size=10,
            epochs=3,
            max_steps=50,
        )

        self.assertEqual(planned_steps, 50)

    def test_planned_training_steps_counts_full_epochs(self):
        planned_steps = _planned_training_steps(
            train_windows=1_000,
            batch_size=10,
            epochs=3,
            max_steps=None,
        )

        self.assertEqual(planned_steps, 300)

    def test_run_size_guard_rejects_accidental_huge_runs(self):
        with self.assertRaisesRegex(ValueError, "very large"):
            _validate_run_size(
                1_000_001,
                allow_long_run=False,
                safety_threshold=1_000_000,
            )

    def test_run_size_guard_allows_explicit_huge_runs(self):
        _validate_run_size(
            1_000_001,
            allow_long_run=True,
            safety_threshold=1_000_000,
        )

    def test_dataset_stride_reduces_overlapping_windows(self):
        dense = EncodedTextDataset(torch.arange(101), context_length=10, stride=1)
        sparse = EncodedTextDataset(torch.arange(101), context_length=10, stride=10)

        self.assertEqual(len(dense), 91)
        self.assertEqual(len(sparse), 10)

    def test_default_encoded_cache_path_distinguishes_full_and_limited_data(self):
        tokenizer_path = Path("artifacts/tokenizers/tinystories-bpe-50k/tokenizer.json")

        full_path = _default_encoded_cache_path(
            tokenizer_path=tokenizer_path,
            max_documents=None,
        )
        limited_path = _default_encoded_cache_path(
            tokenizer_path=tokenizer_path,
            max_documents=10000,
        )

        self.assertIn("tinystories-full", str(full_path))
        self.assertIn("tinystories-10000", str(limited_path))
        self.assertNotEqual(full_path, limited_path)

    def test_load_or_encode_token_ids_reuses_cache_without_dataset_access(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "encoded.pt"
            torch.save({"token_ids": torch.tensor([1, 2, 3], dtype=torch.int32)}, cache_path)

            with patch("learning_llm.scripts.train_tinystories_model.iter_tinystories_texts") as texts:
                token_ids = _load_or_encode_token_ids(
                    tokenizer=None,
                    eos_id=None,
                    cache_path=cache_path,
                    max_documents=None,
                    offline=True,
                    rebuild=False,
                    verbose=False,
                )

        self.assertEqual(token_ids, [1, 2, 3])
        texts.assert_not_called()


if __name__ == "__main__":
    unittest.main()

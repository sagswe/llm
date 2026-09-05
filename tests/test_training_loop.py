import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from learning_llm.data import EncodedTextDataset
from learning_llm.model import DecoderLanguageModel, ModelConfig
from learning_llm.scripts.train_tinystories_model import _model_dimensions_from_args
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


if __name__ == "__main__":
    unittest.main()

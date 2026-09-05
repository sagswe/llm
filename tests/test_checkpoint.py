import tempfile
import unittest
from pathlib import Path

import torch

from learning_llm.model import DecoderLanguageModel, ModelConfig
from learning_llm.training import load_checkpoint, load_training_state, save_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_round_trip_preserves_eval_logits(self):
        torch.manual_seed(0)
        model = DecoderLanguageModel(ModelConfig(dropout=0.0))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        token_ids = torch.tensor([[0, 1, 2, 3]])

        model.eval()
        expected_logits = model(token_ids).logits

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                step=7,
                epoch=2,
            )

            loaded_model, checkpoint = load_checkpoint(path)
            loaded_model.eval()
            actual_logits = loaded_model(token_ids).logits

        self.assertEqual(checkpoint["step"], 7)
        self.assertEqual(checkpoint["epoch"], 2)
        torch.testing.assert_close(actual_logits, expected_logits)

    def test_training_state_round_trip_restores_optimizer(self):
        torch.manual_seed(0)
        model = DecoderLanguageModel(ModelConfig(dropout=0.0))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        token_ids = torch.tensor([[0, 1, 2, 3]])
        targets = torch.tensor([[1, 2, 3, 4]])

        loss = model(token_ids, targets).loss
        loss.backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                step=3,
                epoch=1,
            )
            restored_model = DecoderLanguageModel(ModelConfig(dropout=0.0))
            restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)

            checkpoint = load_training_state(
                path,
                model=restored_model,
                optimizer=restored_optimizer,
            )

        self.assertEqual(checkpoint["step"], 3)
        self.assertTrue(restored_optimizer.state_dict()["state"])


if __name__ == "__main__":
    unittest.main()

import unittest

import torch
from torch.nn import functional as F

from learning_llm.model import DecoderLanguageModel, ModelConfig, apply_causal_mask


class FocusedVerificationTests(unittest.TestCase):
    def test_causal_mask_blocks_future_positions(self):
        scores = torch.zeros(1, 4, 4)

        masked = apply_causal_mask(scores)

        self.assertTrue(torch.isneginf(masked[0, 0, 1]))
        self.assertTrue(torch.isneginf(masked[0, 1, 3]))
        self.assertEqual(masked[0, 2, 0].item(), 0.0)
        self.assertEqual(masked[0, 3, 3].item(), 0.0)

    def test_model_loss_matches_explicit_flattened_cross_entropy(self):
        torch.manual_seed(0)
        model = DecoderLanguageModel(ModelConfig(dropout=0.0))
        token_ids = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
        targets = torch.tensor([[1, 2, 3, 4], [2, 1, 0, 5]])

        output = model(token_ids, targets)
        expected_loss = F.cross_entropy(
            output.logits.reshape(-1, model.config.vocab_size),
            targets.reshape(-1),
        )

        torch.testing.assert_close(output.loss, expected_loss)

    def test_training_step_reduces_loss_on_tiny_fixed_batch(self):
        torch.manual_seed(0)
        model = DecoderLanguageModel(
            ModelConfig(
                vocab_size=8,
                context_length=4,
                d_model=16,
                n_head=4,
                n_layer=1,
                dropout=0.0,
            )
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.0)
        token_ids = torch.tensor(
            [
                [0, 1, 2, 3],
                [1, 2, 3, 4],
                [2, 3, 4, 5],
                [3, 4, 5, 6],
            ]
        )
        targets = torch.tensor(
            [
                [1, 2, 3, 4],
                [2, 3, 4, 5],
                [3, 4, 5, 6],
                [4, 5, 6, 7],
            ]
        )
        initial_loss = model(token_ids, targets).loss.item()

        for _ in range(80):
            optimizer.zero_grad(set_to_none=True)
            loss = model(token_ids, targets).loss
            loss.backward()
            optimizer.step()

        final_loss = model(token_ids, targets).loss.item()

        self.assertLess(final_loss, initial_loss * 0.5)

    def test_generation_is_deterministic_with_fixed_seed(self):
        torch.manual_seed(0)
        model = DecoderLanguageModel(ModelConfig(dropout=0.0))
        token_ids = torch.tensor([[0, 1, 2, 3]])
        first_generator = torch.Generator().manual_seed(123)
        second_generator = torch.Generator().manual_seed(123)

        first = model.generate(
            token_ids,
            max_new_tokens=6,
            temperature=0.8,
            top_k=5,
            generator=first_generator,
        )
        second = model.generate(
            token_ids,
            max_new_tokens=6,
            temperature=0.8,
            top_k=5,
            generator=second_generator,
        )

        torch.testing.assert_close(first, second)


if __name__ == "__main__":
    unittest.main()

import unittest

import torch

from learning_llm.model import DecoderLanguageModel, LanguageModelHead, ModelConfig


class LanguageModelHeadTests(unittest.TestCase):
    def test_head_projects_hidden_states_to_vocabulary_logits(self):
        torch.manual_seed(0)
        head = LanguageModelHead(d_model=8, vocab_size=32)
        hidden_states = torch.randn(2, 5, 8)

        logits = head(hidden_states)

        self.assertEqual(tuple(logits.shape), (2, 5, 32))

    def test_head_expects_batch_sequence_and_width_dims(self):
        head = LanguageModelHead(d_model=8, vocab_size=32)

        with self.assertRaisesRegex(ValueError, "shape"):
            head(torch.randn(5, 8))


class DecoderLanguageModelTests(unittest.TestCase):
    def test_model_projects_token_ids_to_logits(self):
        torch.manual_seed(0)
        model = DecoderLanguageModel(ModelConfig(dropout=0.0))
        token_ids = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])

        output = model(token_ids, return_attention_weights=True)

        self.assertEqual(tuple(output.logits.shape), (2, 5, 32))
        self.assertEqual(len(output.attention_weights), 2)
        self.assertEqual(tuple(output.attention_weights[0].shape), (2, 2, 5, 5))
        self.assertTrue(torch.isfinite(output.logits).all())

    def test_lm_head_shares_token_embedding_parameter(self):
        model = DecoderLanguageModel(ModelConfig())

        self.assertIs(
            model.lm_head.weight,
            model.backbone.token_embedding.weight,
        )

    def test_shared_weight_receives_one_gradient(self):
        torch.manual_seed(0)
        model = DecoderLanguageModel(ModelConfig(dropout=0.0))
        output = model(torch.tensor([[0, 1, 2, 3]])).logits

        output.sum().backward()

        self.assertIs(
            model.lm_head.weight.grad,
            model.backbone.token_embedding.weight.grad,
        )
        self.assertIsNotNone(model.lm_head.weight.grad)

    def test_context_length_validation_still_comes_from_config(self):
        model = DecoderLanguageModel(ModelConfig(context_length=4))

        with self.assertRaisesRegex(ValueError, "context_length"):
            model(torch.zeros(2, 5, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()

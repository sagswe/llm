import unittest

import torch

from learning_llm.model import DecoderBackbone, ModelConfig, TransformerStack


class TransformerStackTests(unittest.TestCase):
    def test_stack_preserves_hidden_state_shape(self):
        torch.manual_seed(0)
        stack = TransformerStack(n_layer=2, d_model=8, n_head=2, dropout=0.0)
        hidden_states = torch.randn(2, 5, 8)

        output, attention_weights = stack(
            hidden_states,
            return_attention_weights=True,
        )

        self.assertEqual(tuple(output.shape), (2, 5, 8))
        self.assertEqual(len(attention_weights), 2)
        self.assertEqual(tuple(attention_weights[0].shape), (2, 2, 5, 5))

    def test_stack_can_run_twelve_decoder_blocks(self):
        torch.manual_seed(0)
        stack = TransformerStack(n_layer=12, d_model=12, n_head=3, dropout=0.0)
        hidden_states = torch.randn(2, 5, 12)

        output, attention_weights = stack(
            hidden_states,
            return_attention_weights=True,
        )

        self.assertEqual(len(stack.blocks), 12)
        self.assertEqual(tuple(output.shape), (2, 5, 12))
        self.assertEqual(len(attention_weights), 12)
        self.assertEqual(tuple(attention_weights[-1].shape), (2, 3, 5, 5))


class DecoderBackboneTests(unittest.TestCase):
    def test_backbone_runs_from_token_ids_to_normalized_hidden_states(self):
        torch.manual_seed(0)
        config = ModelConfig(
            vocab_size=32,
            context_length=16,
            d_model=8,
            n_head=2,
            n_layer=2,
            dropout=0.0,
        )
        model = DecoderBackbone(config)
        token_ids = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])

        output = model(token_ids, return_attention_weights=True)

        self.assertEqual(tuple(output.hidden_states.shape), (2, 5, 8))
        self.assertEqual(len(output.attention_weights), 2)
        self.assertEqual(tuple(output.attention_weights[0].shape), (2, 2, 5, 5))
        self.assertTrue(torch.isfinite(output.hidden_states).all())

    def test_backbone_rejects_sequences_beyond_context_length(self):
        model = DecoderBackbone(ModelConfig(context_length=4))

        with self.assertRaisesRegex(ValueError, "context_length"):
            model(torch.zeros(2, 5, dtype=torch.long))

    def test_backbone_gradients_reach_embedding_table(self):
        torch.manual_seed(0)
        model = DecoderBackbone(ModelConfig(dropout=0.0))
        output = model(torch.tensor([[0, 1, 2, 3]])).hidden_states

        output.sum().backward()

        self.assertIsNotNone(model.token_embedding.weight.grad)
        self.assertIsNotNone(model.position_embedding.embedding.weight.grad)


if __name__ == "__main__":
    unittest.main()

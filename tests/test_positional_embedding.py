import unittest

import torch

from learning_llm.PositionalEmbedding import PositionalEmbedding


class PositionalEmbeddingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.module = PositionalEmbedding(context_length=16, d_model=8)

    def test_table_contains_one_vector_per_supported_position(self):
        self.assertEqual(tuple(self.module.embedding.weight.shape), (16, 8))

    def test_addition_preserves_batch_sequence_and_width(self):
        token_embeddings = torch.zeros(2, 5, 8)

        output = self.module(token_embeddings)

        self.assertEqual(tuple(output.shape), (2, 5, 8))

    def test_same_positions_are_broadcast_across_batch(self):
        token_embeddings = torch.zeros(2, 5, 8)

        output = self.module(token_embeddings)

        torch.testing.assert_close(output[0], output[1])
        torch.testing.assert_close(
            output[0],
            self.module.embedding.weight[:5],
        )

    def test_token_information_is_preserved_by_addition(self):
        first_batch = torch.zeros(1, 5, 8)
        second_batch = torch.ones(1, 5, 8)
        token_embeddings = torch.cat((first_batch, second_batch), dim=0)

        output = self.module(token_embeddings)

        torch.testing.assert_close(output[1] - output[0], torch.ones(5, 8))

    def test_sequence_cannot_exceed_context_length(self):
        token_embeddings = torch.zeros(2, 17, 8)

        with self.assertRaisesRegex(ValueError, "exceeds context_length"):
            self.module(token_embeddings)

    def test_model_width_must_match_embedding_width(self):
        token_embeddings = torch.zeros(2, 5, 4)

        with self.assertRaisesRegex(ValueError, "model width"):
            self.module(token_embeddings)

    def test_positional_table_receives_gradients(self):
        output = self.module(torch.zeros(2, 5, 8))

        output.sum().backward()

        gradient = self.module.embedding.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.all(gradient[:5] != 0))
        self.assertTrue(torch.all(gradient[5:] == 0))


if __name__ == "__main__":
    unittest.main()

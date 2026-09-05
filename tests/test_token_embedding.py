import unittest

import torch

from learning_llm.model.embeddings import TokenEmbedding


class TokenEmbeddingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.module = TokenEmbedding(vocab_size=32, d_model=8)

    def test_table_contains_one_vector_per_vocabulary_token(self):
        self.assertEqual(tuple(self.module.embedding.weight.shape), (32, 8))

    def test_token_ids_map_to_vectors(self):
        token_ids = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])

        output = self.module(token_ids)

        self.assertEqual(tuple(output.shape), (2, 5, 8))
        torch.testing.assert_close(output[0, 1], self.module.embedding.weight[1])

    def test_token_ids_must_have_batch_and_sequence_dims(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            self.module(torch.tensor([0, 1, 2]))

    def test_token_ids_must_be_in_vocabulary_range(self):
        with self.assertRaisesRegex(ValueError, "token IDs"):
            self.module(torch.tensor([[0, 32]]))

    def test_token_ids_must_be_integers(self):
        with self.assertRaisesRegex(TypeError, "integers"):
            self.module(torch.tensor([[0.0, 1.0]]))

    def test_embedding_table_receives_gradients(self):
        output = self.module(torch.tensor([[0, 1, 2]]))

        output.sum().backward()

        gradient = self.module.embedding.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.all(gradient[:3] != 0))
        self.assertTrue(torch.all(gradient[3:] == 0))


if __name__ == "__main__":
    unittest.main()

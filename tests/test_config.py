import unittest

from learning_llm.config import ModelConfig


class ModelConfigTests(unittest.TestCase):
    def test_defaults_are_tiny_and_cpu_friendly(self):
        config = ModelConfig()

        self.assertEqual(config.vocab_size, 32)
        self.assertEqual(config.context_length, 16)
        self.assertEqual(config.d_model, 8)
        self.assertEqual(config.n_head, 2)
        self.assertEqual(config.n_layer, 2)
        self.assertEqual(config.head_dim, 4)

    def test_gpt2_small_reference_dimensions(self):
        config = ModelConfig.gpt2_small()

        self.assertEqual(config.vocab_size, 50_257)
        self.assertEqual(config.context_length, 1_024)
        self.assertEqual(config.d_model, 768)
        self.assertEqual(config.n_head, 12)
        self.assertEqual(config.n_layer, 12)
        self.assertEqual(config.head_dim, 64)
        self.assertEqual(config.dropout, 0.1)

    def test_model_width_must_split_evenly_across_heads(self):
        with self.assertRaisesRegex(ValueError, "d_model.*divisible.*n_head"):
            ModelConfig(d_model=8, n_head=3)

    def test_dimensions_must_be_positive_integers(self):
        for field in (
            "vocab_size",
            "context_length",
            "d_model",
            "n_head",
            "n_layer",
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    ModelConfig(**{field: 0})

    def test_dropout_must_be_a_probability(self):
        for dropout in (-0.1, 1.1):
            with self.subTest(dropout=dropout):
                with self.assertRaisesRegex(ValueError, "dropout"):
                    ModelConfig(dropout=dropout)


if __name__ == "__main__":
    unittest.main()

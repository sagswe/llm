import unittest

from learning_llm.config import ModelConfig
from learning_llm.tokenizer import GPT2Tokenizer


class GPT2TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = GPT2Tokenizer()

    def test_vocabulary_matches_gpt2_small_configuration(self):
        config = ModelConfig.gpt2_small()

        self.assertEqual(self.tokenizer.vocab_size, config.vocab_size)
        self.assertEqual(self.tokenizer.end_of_text_id, 50_256)

    def test_representative_text_round_trip(self):
        text = "Hello, world! A tiny story: 猫 sat on the mat."

        token_ids = self.tokenizer.encode(text)

        self.assertIsInstance(token_ids, list)
        self.assertTrue(all(isinstance(token_id, int) for token_id in token_ids))
        self.assertEqual(self.tokenizer.decode(token_ids), text)

    def test_known_gpt2_encoding(self):
        self.assertEqual(self.tokenizer.encode("hello world"), [31_373, 995])

    def test_end_of_text_is_appended_explicitly(self):
        token_ids = self.tokenizer.encode("The end.", add_end_of_text=True)

        self.assertEqual(token_ids[-1], self.tokenizer.end_of_text_id)
        self.assertEqual(
            self.tokenizer.decode(token_ids),
            "The end.<|endoftext|>",
        )

    def test_literal_end_of_text_is_rejected_as_ordinary_text(self):
        with self.assertRaises(ValueError):
            self.tokenizer.encode("before <|endoftext|> after")

    def test_invalid_token_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "token ID"):
            self.tokenizer.decode([self.tokenizer.vocab_size])

    def test_non_integer_token_id_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "integer"):
            self.tokenizer.decode([1.5])


if __name__ == "__main__":
    unittest.main()

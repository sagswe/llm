import tempfile
import unittest
from pathlib import Path

import torch

from learning_llm.data import (
    BPETokenizer,
    CharacterTokenizer,
    next_token_windows,
    train_byte_level_bpe,
)


class CharacterTokenizerTests(unittest.TestCase):
    def test_round_trip_for_known_characters(self):
        tokenizer = CharacterTokenizer.from_text("hello world")
        token_ids = tokenizer.encode("hello")

        self.assertEqual(tokenizer.decode(token_ids), "hello")


class BPETokenizerTests(unittest.TestCase):
    def test_train_save_load_and_round_trip(self):
        texts = [
            "Once upon a time there was a small red car.",
            "The small car liked short stories.",
            "Once there was a tiny story about a kind kid.",
        ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"

            tokenizer = train_byte_level_bpe(texts, path, vocab_size=128)
            loaded = BPETokenizer.from_file(path)

            self.assertTrue(path.exists())
            self.assertLessEqual(loaded.vocab_size, 128)
            token_ids = tokenizer.encode("Once upon a time.")
            self.assertEqual(loaded.decode(token_ids), "Once upon a time.")


class BatchingTests(unittest.TestCase):
    def test_next_token_windows_are_shifted_by_one(self):
        windows = list(next_token_windows([0, 1, 2, 3, 4], context_length=3))

        self.assertEqual(len(windows), 2)
        torch.testing.assert_close(windows[0][0], torch.tensor([0, 1, 2]))
        torch.testing.assert_close(windows[0][1], torch.tensor([1, 2, 3]))
        torch.testing.assert_close(windows[1][0], torch.tensor([1, 2, 3]))
        torch.testing.assert_close(windows[1][1], torch.tensor([2, 3, 4]))


if __name__ == "__main__":
    unittest.main()

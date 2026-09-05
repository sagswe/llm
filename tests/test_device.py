import unittest
from unittest.mock import patch

import torch

from learning_llm.training import resolve_device


class DeviceTests(unittest.TestCase):
    def test_explicit_device_is_preserved(self):
        self.assertEqual(resolve_device("cpu"), torch.device("cpu"))

    def test_auto_prefers_cuda(self):
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(resolve_device("auto"), torch.device("cuda"))

    def test_auto_falls_back_to_cpu_when_no_accelerator_is_available(self):
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                self.assertEqual(resolve_device("auto"), torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()

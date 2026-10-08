from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from src.embeddings import LocalEmbeddings


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name)
        (self.path / "modules.json").write_text("[]")
        self.embedder = LocalEmbeddings(self.path)

    def tearDown(self):
        self.directory.cleanup()

    def test_model_identity_changes_when_weights_change(self):
        weights = self.path / "model.safetensors"
        weights.write_bytes(b"first weights")
        first = self.embedder.identity
        weights.write_bytes(b"second weights")
        self.assertNotEqual(first, LocalEmbeddings(self.path).identity)

    def test_long_text_chunks_fit_limit_and_preserve_original_text(self):
        class Tokenizer:
            def num_special_tokens_to_add(self, pair):
                return 2

            def __call__(self, text, **kwargs):
                return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}

        self.embedder._model = Mock(tokenizer=Tokenizer(), max_seq_length=6)
        self.assertEqual(self.embedder.split_text("ABCDEFGHI", 4, 1), ["ABCD", "DEFG", "GHI"])
        with self.assertRaises(ValueError):
            self.embedder.split_text("ABCDE", 5, 1)

    def test_missing_model_fails_without_downloading(self):
        with self.assertRaisesRegex(ValueError, "prepare"):
            LocalEmbeddings(self.path / "missing")


if __name__ == "__main__":
    unittest.main()

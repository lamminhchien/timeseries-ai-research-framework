"""
Unit Test Suite for Time-Series AI Research Framework.
Validates CausalConv1dBlock, PositionalEncoding, and ExtractorTransformer forward pass.
"""

import unittest
import torch
import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.extractor_transformer import CausalConv1dBlock, ExtractorTransformer, PositionalEncoding


class TestModelArchitecture(unittest.TestCase):
    def test_causal_conv1d_strictly_preserves_length(self):
        batch, channels, seq_len = 4, 8, 60
        x = torch.randn(batch, channels, seq_len)
        block = CausalConv1dBlock(in_channels=channels, out_channels=16, kernel_size=3, dilation=2)
        out = block(x)

        # Output sequence length must match input sequence length exactly
        self.assertEqual(out.shape, (batch, 16, seq_len))

    def test_extractor_transformer_forward_pass(self):
        batch, seq_len, mkt_dim, acc_dim = 8, 60, 8, 4
        mkt = torch.randn(batch, seq_len, mkt_dim)
        acc = torch.randn(batch, acc_dim)

        model = ExtractorTransformer(
            market_feature_dim=mkt_dim,
            seq_length=seq_len,
            account_dim=acc_dim,
            tcn_channels=[32, 64],
            tfm_nhead=4,
            tfm_layers=2
        )
        out = model(mkt, acc)

        # Output shape: [batch, tcn_out_dim + account_embed_dim] = [8, 64 + 32] = [8, 96]
        self.assertEqual(out.shape, (batch, 96))
        self.assertFalse(torch.isnan(out).any())


if __name__ == "__main__":
    unittest.main()

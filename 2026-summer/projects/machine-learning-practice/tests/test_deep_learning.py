import torch

from ml_practice.deep_learning import CNNClassifier, MLPClassifier, SelfAttentionClassifier


def test_neural_network_output_shapes() -> None:
    batch = torch.rand(4, 1, 8, 8)
    assert MLPClassifier()(batch).shape == (4, 10)
    assert CNNClassifier()(batch).shape == (4, 10)
    assert SelfAttentionClassifier()(batch).shape == (4, 10)


def test_attention_rows_are_probability_distributions() -> None:
    model = SelfAttentionClassifier()
    model(torch.rand(3, 1, 8, 8))
    assert model.last_attention is not None
    row_sums = model.last_attention.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

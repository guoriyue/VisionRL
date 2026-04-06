import torch

from wm_infra.config import ModelConfig, MoEConfig, TransformerConfig
from wm_infra.layers import AttentionLayer, MoELayer, TransformerBlock, TransformerModel


def test_layers_package_imports_and_minimal_configs_work():
    moe_config = MoEConfig(
        num_experts=4,
        top_k=2,
        hidden_dim=16,
        intermediate_dim=32,
    )
    block_config = TransformerConfig(
        hidden_dim=16,
        num_heads=4,
        num_kv_heads=2,
        head_dim=4,
        max_seq_len=32,
        moe=moe_config,
    )
    model_config = ModelConfig(
        vocab_size=32,
        hidden_dim=16,
        num_layers=1,
        block=block_config,
        moe_layer_indices=[],
        intermediate_dim_dense=32,
    )

    moe = MoELayer(moe_config)
    attention = AttentionLayer(block_config)
    block = TransformerBlock(block_config, use_moe=False)
    model = TransformerModel(model_config)

    assert isinstance(moe, MoELayer)
    assert isinstance(attention, AttentionLayer)
    assert isinstance(block, TransformerBlock)
    assert isinstance(model, TransformerModel)
    assert model.embed_tokens.weight.shape == torch.Size([32, 16])

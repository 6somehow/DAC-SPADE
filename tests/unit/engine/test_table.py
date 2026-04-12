from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")

from spade.engine.engine import SparseExecutorTable
from tests.utils.gen_sparse import gen_sink_sparse_config


@pytest.fixture
def sparse_config():
    return gen_sink_sparse_config()


def test_register_config_success(sparse_config):
    executor_table = SparseExecutorTable()

    index = executor_table.register_config(sparse_config)

    assert index == 1
    assert len(executor_table) == 2
    assert executor_table.get_config(index) is sparse_config
    assert executor_table.attn_dtype == sparse_config.attn_dtype


def test_register_config_invalid_type():
    executor_table = SparseExecutorTable()

    with pytest.raises(TypeError, match="Only SparseHeadConfig objects"):
        executor_table.register_config("not_a_config")


def test_register_config_rejects_mixed_attention_dtype(sparse_config):
    executor_table = SparseExecutorTable()
    other_config = sparse_config.copy()
    other_config.attn_dtype = torch.float16

    executor_table.register_config(sparse_config)
    with pytest.raises(TypeError, match="attn_dtype"):
        executor_table.register_config(other_config)


def test_compile_materializes_registered_configs(sparse_config):
    executor_table = SparseExecutorTable()
    executor_table.register_config(sparse_config)

    with patch("spade.engine.engine.compile", return_value=MagicMock()) as mock_compile:
        executor_table.compile("torch", "bnsh")

    mock_compile.assert_called_once_with(sparse_config, "torch", "bnsh")
    assert len(executor_table.sparse_executors) == 2


def test_get_config_returns_none_for_dense_or_out_of_range(sparse_config):
    executor_table = SparseExecutorTable()
    executor_table.register_config(sparse_config)

    assert executor_table.get_config(0) is None
    assert executor_table.get_config(2) is None

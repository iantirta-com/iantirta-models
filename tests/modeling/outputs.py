
import pytest
import torch
import torch.utils._pytree as pytree

from iantirta.models.common.modeling_outputs import BaseModelOutput


# ============================================================================
# Shared Fixtures (Optimized for minimal tensor creation overhead)
# ============================================================================

@pytest.fixture(scope="module")
def dummy_tensors():
    """Generates reusable dummy tensors for testing."""
    return {
        "last_hidden_state": torch.randn(2, 8, 16),
        "hidden_states": (torch.randn(2, 8, 16), torch.randn(2, 8, 16)),
        "attentions": (torch.randn(2, 4, 8, 8),),
    }


# ============================================================================
# 1. Flexible Instantiation Tests
# ============================================================================

class TestInstantiation:
    """Tests all backwards-compatible ways to instantiate BaseModelOutput."""

    def test_standard_kwargs(self, dummy_tensors):
        out = BaseModelOutput(
            last_hidden_state=dummy_tensors["last_hidden_state"],
            hidden_states=dummy_tensors["hidden_states"],
        )
        assert torch.equal(out.last_hidden_state, dummy_tensors["last_hidden_state"])
        assert out.hidden_states == dummy_tensors["hidden_states"]
        assert out.attentions is None

    def test_single_tensor_positional(self, dummy_tensors):
        # Passing a single tensor positionally sets the first field (last_hidden_state)
        out = BaseModelOutput(dummy_tensors["last_hidden_state"])
        assert torch.equal(out.last_hidden_state, dummy_tensors["last_hidden_state"])
        assert out.hidden_states is None
        assert out.attentions is None

    def test_dict_unpacking(self, dummy_tensors):
        # Passing a dictionary as the 1st positional arg
        raw_dict = {
            "last_hidden_state": dummy_tensors["last_hidden_state"],
            "attentions": dummy_tensors["attentions"],
        }
        out = BaseModelOutput(raw_dict)
        assert torch.equal(out.last_hidden_state, dummy_tensors["last_hidden_state"])
        assert out.attentions == dummy_tensors["attentions"]
        assert out.hidden_states is None

    def test_key_value_tuple_list_unpacking(self, dummy_tensors):
        # Passing a list of (key, value) tuples as 1st positional arg
        kv_pairs = [
            ("last_hidden_state", dummy_tensors["last_hidden_state"]),
            ("hidden_states", dummy_tensors["hidden_states"]),
        ]
        out = BaseModelOutput(kv_pairs)
        assert torch.equal(out.last_hidden_state, dummy_tensors["last_hidden_state"])
        assert out.hidden_states == dummy_tensors["hidden_states"]


# ============================================================================
# 2. Hybrid Access & Indexing Tests (Dict + Tuple + Dataclass)
# ============================================================================

class TestAccessAndIndexing:
    """Tests attribute access, dict key access, integer tuple indexing, and to_tuple()."""

    def test_attribute_and_dict_access_sync(self, dummy_tensors):
        out = BaseModelOutput(last_hidden_state=dummy_tensors["last_hidden_state"])
        
        # Key access
        assert torch.equal(out["last_hidden_state"], dummy_tensors["last_hidden_state"])
        
        # Dynamic attribute mutation syncs to dict access
        new_tensor = torch.randn(2, 8, 16)
        out.last_hidden_state = new_tensor
        assert torch.equal(out["last_hidden_state"], new_tensor)

    @pytest.mark.parametrize(
        "index, expected_tensor_key",
        [
            (0, "last_hidden_state"),
            (1, "attentions"),  # hidden_states is None, so it gets skipped in to_tuple!
        ],
    )
    def test_integer_tuple_indexing(self, dummy_tensors, index, expected_tensor_key):
        # hidden_states is omitted (None) to test that None-skipping integer indexing works
        out = BaseModelOutput(
            last_hidden_state=dummy_tensors["last_hidden_state"],
            attentions=dummy_tensors["attentions"],
        )
        assert out[index] is dummy_tensors[expected_tensor_key]

    def test_to_tuple_skips_none(self, dummy_tensors):
        out = BaseModelOutput(
            last_hidden_state=dummy_tensors["last_hidden_state"],
            hidden_states=None,
            attentions=dummy_tensors["attentions"],
        )
        tup = out.to_tuple()
        
        # Length should be 2 because hidden_states=None is filtered out
        assert len(tup) == 2
        assert torch.equal(tup[0], dummy_tensors["last_hidden_state"])
        assert tup[1] == dummy_tensors["attentions"]


# ============================================================================
# 3. Immutability & Exception Safety Tests
# ============================================================================

class TestSafetyAndExceptions:
    """Tests disabled methods and invalid unpacking scenarios."""

    def test_pop_raises_exception(self, dummy_tensors):
        out = BaseModelOutput(last_hidden_state=dummy_tensors["last_hidden_state"])
        with pytest.raises(Exception, match="You cannot use ``pop``"):
            out.pop("last_hidden_state")

    def test_update_raises_exception(self, dummy_tensors):
        out = BaseModelOutput(last_hidden_state=dummy_tensors["last_hidden_state"])
        with pytest.raises(Exception, match="You cannot use ``update``"):
            out.update({"hidden_states": dummy_tensors["hidden_states"]})

    def test_invalid_key_value_iterator_raises_value_error(self):
        # Malformed key-value items (not 2-element tuples)
        invalid_kv = [("last_hidden_state", torch.randn(2, 2)), "invalid_string_item"]
        with pytest.raises(ValueError, match="Cannot set key/value"):
            BaseModelOutput(invalid_kv)


# ============================================================================
# 4. PyTree / PyTorch Integration Tests
# ============================================================================

class TestPyTreeIntegration:
    """Verifies PyTorch PyTree flattening and unflattening (used in torch.compile)."""

    def test_pytree_flatten_and_unflatten(self, dummy_tensors):
        original_out = BaseModelOutput(
            last_hidden_state=dummy_tensors["last_hidden_state"],
            hidden_states=dummy_tensors["hidden_states"],
        )

        # Flatten using PyTorch internals
        leaves, treespec = pytree.tree_flatten(original_out)

        # Unflatten back into class instance
        reconstructed_out = pytree.tree_unflatten(leaves, treespec)

        # Assert correct class type and deep tensor equality
        assert isinstance(reconstructed_out, BaseModelOutput)
        assert torch.equal(reconstructed_out.last_hidden_state, original_out.last_hidden_state)
        assert len(reconstructed_out.hidden_states) == len(original_out.hidden_states)
        for t1, t2 in zip(reconstructed_out.hidden_states, original_out.hidden_states):
            assert torch.equal(t1, t2)


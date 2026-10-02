import torch
import torch.utils._pytree as pytree


# 1. Define a custom output object (similar to HF ModelOutput)
class ModelPrediction:

    def __init__(self, logits, hidden_states=None, loss=None):
        self.logits = logits
        self.hidden_states = hidden_states
        self.loss = loss

    def __repr__(self):
        return (
            f"ModelPrediction(logits={tuple(self.logits.shape)}, "
            f"has_hidden={self.hidden_states is not None}, loss={self.loss})"
        )


# 2. Define the Flatten function
# Must return: (list_of_leaf_tensors, context_metadata)
def _prediction_flatten(pred: ModelPrediction):
    leaves = [pred.logits, pred.hidden_states, pred.loss]
    context = None  # Static metadata if needed (e.g. key order or non-tensor attributes)
    return leaves, context


# 3. Define the Unflatten function
# Reconstructs the object from (list_of_leaf_tensors, context_metadata)
def _prediction_unflatten(leaves, context):
    logits, hidden_states, loss = leaves
    return ModelPrediction(logits=logits, hidden_states=hidden_states, loss=loss)


# 4. Register the custom object type with PyTorch
pytree.register_pytree_node(
    ModelPrediction,
    _prediction_flatten,
    _prediction_unflatten,
)

# ==========================================
# Demonstration
# ==========================================

# Create an instance with dummy tensors
original_obj = ModelPrediction(
    logits=torch.randn(2, 10),
    hidden_states=torch.randn(2, 128),
    loss=torch.tensor(0.35),
)

print("1. Original Object:")
print(f"   {original_obj}\n")

# Flattening: PyTorch extracts all internal tensors into a flat list
leaves, treespec = pytree.tree_flatten(original_obj)

print("2. Flattened Leaves (Tensors PyTorch can process):")
print(f"   {leaves}\n")

print("3. TreeSpec (Metadata used to reconstruct):")
print(f"   {treespec}\n")

# Unflattening: PyTorch rebuilds the original object structure
reconstructed_obj = pytree.tree_unflatten(leaves, treespec)

print("4. Reconstructed Object:")
print(f"   {reconstructed_obj}")

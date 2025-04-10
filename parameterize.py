import torch
import torch.nn as nn
from torch.nn.utils import parametrize

# 1. Basic Parametrization Example
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

# 2. Custom Orthogonal Parametrization
class OrthogonalMatrix(nn.Module):
    def forward(self, X):
        # Use QR decomposition to create orthogonal matrix
        q, r = torch.linalg.qr(X)
        # Make the orthogonal matrix unique
        d = torch.diag(r)
        ph = d.sign()
        q *= ph.view(-1, 1)
        return q

# Apply orthogonal parametrization
model = SimpleNet()
parametrize.register_parametrization(model.linear, "weight", 
                                   OrthogonalMatrix())

# 3. Custom Positive Parametrization
class PositiveLinear(nn.Module):
    def forward(self, X):
        return torch.abs(X)  # Ensures weights are always positive

# Apply custom parametrization
model2 = SimpleNet()
parametrize.register_parametrization(model2.linear, "weight", 
                                   PositiveLinear())

# 4. Key Features of parametrize:
# - Allows modifying parameters during training
# - Maintains original parameter while applying transformation
# - Can be removed using remove_parametrizations()
# - Supports multiple parametrizations on same parameter

# Example: Remove parametrization
parametrize.remove_parametrizations(model.linear, "weight")

# Note: parametrizations are part of the model's state_dict
print("Parametrized:", parametrize.is_parametrized(model2.linear))
print("Parametrized parameters:", [name for name, _ in model2.linear.parametrizations.items()])

# Test the orthogonal parametrization
with torch.no_grad():
    # Need to re-register the parametrization since we removed it
    parametrize.register_parametrization(model.linear, "weight", OrthogonalMatrix())
    # Get the parametrized weight
    weight = model.linear.weight
    # Check if it's orthogonal by multiplying with its transpose
    product = torch.mm(weight, weight.t())
    # Should be close to identity matrix
    identity = torch.eye(weight.size(0))
    print("\nTesting orthogonality:")
    print("Max difference from identity:", torch.max(torch.abs(product - identity)).item())

# Print weights before and after parametrization
def print_weight_comparison(model, parametrization):
    # Create a fresh model
    fresh_model = SimpleNet()
    
    # Store original weights
    original_weights = fresh_model.linear.weight.clone().detach()
    print("\nOriginal weights:")
    print(original_weights)
    
    # Apply parametrization
    parametrize.register_parametrization(fresh_model.linear, "weight", parametrization)
    
    # Get parametrized weights
    parametrized_weights = fresh_model.linear.weight.clone().detach()
    print("\nAfter parametrization:")
    print(parametrized_weights)
    
    return original_weights, parametrized_weights

# Test with Orthogonal parametrization
print("\n=== Testing Orthogonal Parametrization ===")
orig_orth, param_orth = print_weight_comparison(model, OrthogonalMatrix())

# Test with Positive parametrization
print("\n=== Testing Positive Parametrization ===")
orig_pos, param_pos = print_weight_comparison(model2, PositiveLinear())

# Print some analysis
print("\nOrthogonal Parametrization Analysis:")
print("Max absolute value before:", torch.max(torch.abs(orig_orth)).item())
print("Max absolute value after:", torch.max(torch.abs(param_orth)).item())

print("\nPositive Parametrization Analysis:")
print("Min value before:", torch.min(orig_pos).item())
print("Min value after (should be positive):", torch.min(param_pos).item())

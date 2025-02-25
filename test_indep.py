import numpy as np
from scipy.stats import chi2
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr, t
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim


def fisher_aggregate_pvalues(p_values):
    """
    Aggregates a list of p-values using Fisher's method.

    Parameters:
    p_values (list or array): A list of L p-values in (0,1].

    Returns:
    float: The aggregated p-value.
    """
    L = len(p_values)
    xi = np.sum(np.log(p_values))
    p_hat = 1 - chi2.cdf(-2 * xi, df=2 * L)
    
    return p_hat



def cosine_similarity_matching(W1, W2):
    """
    Matches rows of matrices W1 and W2 based on cosine similarity.

    Parameters:
    W1 (numpy.ndarray): Matrix of shape (h, d).
    W2 (numpy.ndarray): Matrix of shape (h, d).

    Returns:
    numpy.ndarray: Permutation π mapping rows of W1 to rows of W2.

    # Example usage
    W1 = np.random.rand(5, 3)  # Random matrix with 5 rows and 3 columns
    W2 = np.random.rand(5, 3)  # Another random matrix

    permutation = cosine_similarity_matching(W1, W2)
    print("Permutation π:", permutation)
    """
    # Compute cosine distance (1 - cosine similarity)
    C = 1 - cdist(W1, W2, metric='cosine')

    # Solve the Linear Assignment Problem (maximize similarity)
    row_ind, col_ind = linear_sum_assignment(-C)  # Negate to maximize similarity

    return col_ind  # π mapping rows of W1 to W2


def spearman_p_value(permutation1, permutation2):
    """
    Compute the p-value from the Spearman rank correlation of two permutations.

    Args:
        permutation1 (list or np.array): A permutation of indices.
        permutation2 (list or np.array): Another permutation of indices.

    Returns:
        float: The p-value from the Spearman correlation test.

    # Example usage
    perm1 = np.random.permutation(10)  # Example permutation 1
    perm2 = np.random.permutation(10)  # Example permutation 2

    p_value = spearman_p_value(perm1, perm2)
    p_value
    """
    h = len(permutation1)  # Number of matched hidden units
    
    # Compute Spearman rank correlation
    r, _ = spearmanr(permutation1, permutation2)
    
    # Compute t-statistic
    t_statistic = r * np.sqrt((h - 2) / (1 - r**2)) if abs(r) < 1 else np.inf
    
    # Compute p-value using the t-distribution with (h-2) degrees of freedom
    p_value = 1 - t.cdf(t_statistic, df=h-2)
    
    return p_value

class GLUMlp(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GLUMlp, self).__init__()
        self.G = nn.Linear(input_dim, hidden_dim)
        self.U = nn.Linear(input_dim, hidden_dim)
        self.D = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, X):
        gate_output = torch.relu(self.G(X))
        up_output = self.U(X)
        Y_pred = self.D(gate_output * up_output)
        return Y_pred

def train_glu_mlp_approximation(X, target_outputs, hidden_dim, num_epochs=500, lr=0.001):
    """
    Trains a GLU-like MLP to approximate the target outputs using PyTorch.
    
    Args:
        X (np.ndarray): Input data of shape (d, N).
        target_outputs (np.ndarray): Target outputs of shape (d, N).
        hidden_dim (int): Hidden layer size.
    
    Returns:
        GLUMlp: Trained GLU MLP model.
    """
    X = X.detach()
    target_outputs = target_outputs.detach()
    N, d = X.shape
    d_out = target_outputs.shape[1]
    
    model = GLUMlp(d, hidden_dim, d_out)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        Y_pred = model(X)
        loss = criterion(Y_pred, target_outputs)
        
        loss.backward()
        optimizer.step()
    
    return model, loss


def generalized_robust_test(model1, model2, L, hidden_dim=100, num_samples=5000, input_dim=784):
    """
    Implements Algorithm 5: the generalized robust test.
    
    Args:
        model1, model2: Two standard MLP models with L layers and ReLU activation.
        hidden_dim (int): Hidden dimension for the GLU approximation.
        num_samples (int): Number of samples from P (standard Gaussian).
        input_dim (int): Dimensionality of the input space.
    
    Returns:
        float: The p-value computed from the robust test ϕ_MATCH.
    """
    # Sample input data X from a standard Gaussian (P)
    X = torch.randn(num_samples, input_dim) # Shape: (num_samples, input_dim)
    model1.to('cpu')
    model2.to('cpu')
    
    with torch.no_grad():
        # For each model, extract the representation at layer L.
        representation1 = X
        for i, layer in enumerate(model1.model):
            representation1 = layer(representation1)
            
            # Check if the layer index matches L (considering ReLU layers)
            if i == L:
                break
        representation2 = X
        for i, layer in enumerate(model2.model):
            representation2 = layer(representation2)
            
            # Check if the layer index matches L (considering ReLU layers)
            if i == L:
                break
    
    # Train proxy GLU MLPs to approximate the target outputs
    proxy1, loss1 = train_glu_mlp_approximation(X, representation1, hidden_dim)
    print(loss1.item())
    proxy2, loss2 = train_glu_mlp_approximation(X, representation2, hidden_dim)
    print(loss2.item())
    
    # Extract features for matching.
    # H_gate: use the gate projection (proxy['G'] @ X)
    # H_up: use the up projection (proxy['U'] @ X)
    with torch.no_grad():
        proxy1.eval()
        proxy2.eval()
        H_gate1 = proxy1.G(X).numpy()
        H_gate2 = proxy2.G(X).numpy()
        H_up1 = proxy1.U(X).numpy()
        H_up2 = proxy2.U(X).numpy()

    
    # Compute matching permutation for gate projections
    permutation_gate = cosine_similarity_matching(H_gate1.T, H_gate2.T)
    # Compute matching permutation for up projections
    permutation_up = cosine_similarity_matching(H_up1.T, H_up2.T)
    
    # Compute the robust test statistic ϕ_MATCH as the Spearman p-value 
    # between the two permutations (matching orders)
    p_value = spearman_p_value(permutation_up, permutation_gate)
    # Alternatively, one might combine both matching results (e.g., aggregate two p-values)
    # For simplicity, we use one here.
    
    return p_value

# Example usage (do not run automatically):
# theta1 = np.random.randn(256, 256)  # Simulated model parameters for model 1
# theta2 = np.random.randn(256, 256)  # Simulated model parameters for model 2
# p_val = generalized_robust_test(theta1, theta2)
# print("Generalized robust test p-value:", p_val)

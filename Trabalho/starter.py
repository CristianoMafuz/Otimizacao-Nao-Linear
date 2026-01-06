import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt
import numpy as np

# --------------------------
# Dataset and Model
# --------------------------
def get_mnist_dataloader(batch_size=64):
    # Load MNIST dataset and wrap it in a DataLoader
    transform = transforms.ToTensor()  # convert images to tensors
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, dataset

def create_model():
    # Define a simple 2-layer MLP with ReLU activation
    return nn.Sequential(
        nn.Flatten(),                     # flatten image to vector
        nn.Linear(784, 128), nn.ReLU(),  # hidden layer
        nn.Linear(128, 10)               # output layer (10 classes)
    )

# --------------------------
# Training
# --------------------------
def train_model(model, dataloader, device, epochs=1, lr=1e-3):
    # Basic model training loop
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()

# --------------------------
# Gradient Leakage Simulation
# --------------------------
def sample_single(dataset, device):
    # Randomly select one (image, label) pair
    x, y = random.choice(dataset)
    x = x.unsqueeze(0).to(device)  # add batch dimension
    y = torch.tensor([y], device=device)
    return x, y

def sample_batch(dataset, batch_size, device):
    # Sample a batch of images from dataset
    indices = random.sample(range(len(dataset)), batch_size)
    xs, ys = zip(*[dataset[i] for i in indices])
    xs = torch.stack(xs).to(device)
    ys = torch.tensor(ys, device=device)
    return xs, ys

def compute_gradient(model, x, y, reduction='mean'):
    # Compute gradient of loss w.r.t model parameters
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction=reduction)
    loss = loss_fn(model(x), y)
    grads = torch.autograd.grad(loss, model.parameters())
    return torch.cat([g.detach().view(-1) for g in grads])  # flatten all gradients into one vector

# --------------------------
# Reconstruction
# --------------------------
def reconstruct_single(model, g_star, lambda_reg=1e-4, iters=300, lr=0.1, seed=0, device='cpu'):
    # Reconstruct a single image + label from leaked gradient
    torch.manual_seed(seed)
    x_hat = torch.rand(1, 1, 28, 28, requires_grad=True, device=device)  # random initial image
    y_logits = torch.zeros(1, 10, requires_grad=True, device=device)     # learnable label logits
    optimizer = optim.LBFGS([x_hat, y_logits], lr=lr, max_iter=20)

    loss_trace = []  # to store objective value at each iteration

    def closure():
        optimizer.zero_grad()
        y_prob = y_logits.softmax(dim=-1)  # turn logits into probabilities
        loss = (model(x_hat) * y_prob).sum()  # soft one-hot prediction
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        grad_diff = torch.cat([g.view(-1) for g in grads]) - g_star
        obj = grad_diff.pow(2).sum() + lambda_reg * x_hat.pow(2).sum()
        obj.backward()
        loss_trace.append(obj.item())
        return obj

    for _ in range(iters):
        optimizer.step(closure)

    predicted_label = y_logits.argmax(dim=-1).item()
    return x_hat.detach().cpu(), predicted_label, loss_trace

def reconstruct_batch(model, g_star, batch_size, lambda_reg=1e-4, iters=300, lr=0.1, seed=0, device='cpu'):
    # Reconstruct a batch of images and labels from gradient
    torch.manual_seed(seed)
    x_hat = torch.rand(batch_size, 1, 28, 28, requires_grad=True, device=device)
    y_logits = torch.zeros(batch_size, 10, requires_grad=True, device=device)
    optimizer = optim.LBFGS([x_hat, y_logits], lr=lr, max_iter=20)

    loss_trace = []

    def closure():
        optimizer.zero_grad()
        y_prob = y_logits.softmax(dim=-1)
        loss = (model(x_hat) * y_prob).sum()
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        grad_diff = torch.cat([g.view(-1) for g in grads]) - g_star
        obj = grad_diff.pow(2).sum() + lambda_reg * x_hat.pow(2).sum()
        obj.backward()
        loss_trace.append(obj.item())
        return obj

    for _ in range(iters):
        optimizer.step(closure)

    predicted_labels = y_logits.argmax(dim=-1).cpu().tolist()
    return x_hat.detach().cpu(), predicted_labels, loss_trace

# --------------------------
# Evaluation and Visualization
# --------------------------
def plot_reconstruction(x_gt, x_recon, title):
    # Visual comparison between ground truth and reconstructed image
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(x_gt.squeeze().numpy(), cmap='gray')
    axes[0].set_title('Ground Truth')
    axes[1].imshow(x_recon.squeeze().numpy(), cmap='gray')
    axes[1].set_title('Reconstructed')
    fig.suptitle(title)
    plt.show()

def plot_loss(loss_trace):
    # Plot the loss trajectory during optimization
    plt.plot(loss_trace)
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.title('Loss over Iterations')
    plt.grid(True)
    plt.show()

# --------------------------
# Main
# --------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader, dataset = get_mnist_dataloader()
    model = create_model()
    train_model(model, dataloader, device)

    # Single inversion example
    x_gt, y_gt = sample_single(dataset, device)
    g_star = compute_gradient(model, x_gt, y_gt)
    x_recon, y_recon, loss_trace = reconstruct_single(model, g_star, device=device)
    print(f"[Single] True: {y_gt.item()}, Reconstructed: {y_recon}")
    plot_reconstruction(x_gt.cpu(), x_recon, f"Single Sample: True={y_gt.item()}, Recon={y_recon}")
    plot_loss(loss_trace)

    # Batch inversion example
    x_batch, y_batch = sample_batch(dataset, 4, device)
    g_star_batch = compute_gradient(model, x_batch, y_batch)
    x_recon_batch, y_recon_batch, batch_loss_trace = reconstruct_batch(model, g_star_batch, 4, device=device)
    print(f"[Batch] True: {y_batch.cpu().tolist()}, Reconstructed: {y_recon_batch}")

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt
import numpy as np
import csv
from typing import Tuple, List, Optional
import os

DIR = "imgs_fashionmnist_improved"
os.makedirs(DIR, exist_ok=True)

# --------------------------
# 1. Dataset Loading para Fashion-MNIST
# --------------------------
def get_fashionmnist_dataloader(batch_size=64, normalize=True):
    """Load Fashion-MNIST com normalização opcional"""
    if normalize:
        # ESTATÍSTICAS ESPECÍFICAS DO Fashion-MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))  # Fashion-MNIST statistics
        ])
    else:
        transform = transforms.ToTensor()
    
    # Usar FashionMNIST
    dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, dataset

# --------------------------
# 2. CALCULAR ESTATÍSTICAS CORRETAS DO Fashion-MNIST
# --------------------------
def calculate_fashionmnist_statistics():
    """Calcular média e desvio padrão corretos do Fashion-MNIST"""
    transform = transforms.ToTensor()
    dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=1000, shuffle=False)
    
    mean = 0.0
    std = 0.0
    total_samples = 0
    
    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    print(f"Fashion-MNIST statistics - Mean: {mean.item():.4f}, Std: {std.item():.4f}")
    return mean.item(), std.item()

# --------------------------
# 3. LABELS DO Fashion-MNIST (Para melhor interpretação)
# --------------------------
FASHION_MNIST_LABELS = {
    0: 'T-shirt/top',
    1: 'Trouser', 
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

def get_fashion_label_name(label_idx):
    """Converter índice para nome da classe Fashion-MNIST"""
    return FASHION_MNIST_LABELS.get(label_idx, f"Unknown({label_idx})")

# --------------------------
# 4. FUNÇÃO DE VISUALIZAÇÃO MELHORADA PARA Fashion-MNIST
# --------------------------
def save_fashion_comparison_image(x_gt, x_recon, y_gt, y_pred, filename):
    """Salvar comparação visual com labels do Fashion-MNIST"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(x_gt.cpu().squeeze(), cmap='gray')
    gt_label = y_gt.item() if isinstance(y_gt, torch.Tensor) else y_gt
    plt.title(f'Ground Truth\nLabel: {gt_label} ({get_fashion_label_name(gt_label)})')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(x_recon.squeeze(), cmap='gray')
    pred_label = y_pred[0] if isinstance(y_pred, (list, np.ndarray)) else y_pred
    plt.title(f'Reconstructed\nLabel: {pred_label} ({get_fashion_label_name(pred_label)})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def create_model(hidden_size=128, activation='relu', use_bn=False):
    """Create model with configurable architecture"""
    activations = {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'leaky_relu': nn.LeakyReLU(0.2)
    }
    
    layers = [nn.Flatten()]
    
    if use_bn:
        layers.extend([
            nn.Linear(784, hidden_size),
            nn.BatchNorm1d(hidden_size),
            activations[activation],
            nn.Linear(hidden_size, 10)
        ])
    else:
        layers.extend([
            nn.Linear(784, hidden_size),
            activations[activation],
            nn.Linear(hidden_size, 10)
        ])
    
    return nn.Sequential(*layers)

# --------------------------
# Enhanced Training
# --------------------------
def train_model(model, dataloader, device, epochs=5, lr=1e-3, weight_decay=1e-5):
    """Improved training with better convergence"""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    loss_fn = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

# --------------------------
# Enhanced Gradient Computation
# --------------------------
def compute_gradient_robust(model, x, y, reduction='mean', normalize=False):
    """More robust gradient computation with optional normalization"""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction=reduction)
    
    # Ensure gradients are computed
    for param in model.parameters():
        param.requires_grad_(True)
    
    loss = loss_fn(model(x), y)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=False, retain_graph=False)
    
    # Concatenate and optionally normalize
    grad_vec = torch.cat([g.contiguous().view(-1) for g in grads if g is not None])
    
    if normalize:
        grad_vec = grad_vec / (grad_vec.norm() + 1e-8)
    
    return grad_vec

# --------------------------
# Advanced Reconstruction Methods
# --------------------------
def total_variation_loss(x):
    """Total variation loss for smoother images"""
    return (torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])) + 
            torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])))

def group_consistency_loss(x_batch):
    """Encourage consistency within batch for batch attacks"""
    if x_batch.size(0) <= 1:
        return torch.tensor(0.0, device=x_batch.device)
    
    mean_img = x_batch.mean(dim=0, keepdim=True)
    return F.mse_loss(x_batch, mean_img.expand_as(x_batch))

def reconstruct_advanced(model, g_star, batch_size=1, 
                        lambda_l2=1e-2, lambda_tv=1e-4, lambda_gc=0.0,
                        iters=1000, lr=0.05, restarts=5, 
                        use_scheduler=False, early_stopping=True,
                        device='cpu', normalize_data=False):
    """Advanced reconstruction with multiple improvements"""
    
    best_result = None
    best_loss = float('inf')
    
    # Adaptive learning rate schedule
    def get_lr_schedule(optimizer, warmup_steps=100):
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return 0.95 ** ((step - warmup_steps) // 50)
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    for restart in range(restarts):
        # Better initialization - ensure leaf tensors
        if normalize_data:
            x_hat = torch.randn(batch_size, 1, 28, 28, device=device) * 0.1
        else:
            x_hat = torch.rand(batch_size, 1, 28, 28, device=device) * 0.8 + 0.1
            
        y_logits = torch.randn(batch_size, 10, device=device) * 0.1
        
        # Make them leaf tensors with requires_grad
        x_hat = x_hat.detach().requires_grad_(True)
        y_logits = y_logits.detach().requires_grad_(True)
        
        # Use different optimizers for different restarts
        if restart < restarts // 2:
            optimizer = optim.Adam([x_hat, y_logits], lr=lr, betas=(0.9, 0.999))
        else:
            optimizer = optim.RMSprop([x_hat, y_logits], lr=lr*0.5, alpha=0.99)
            
        if use_scheduler:
            scheduler = get_lr_schedule(optimizer)
        
        loss_trace = []
        best_iter_loss = float('inf')
        patience = 200
        no_improve = 0
        
        for iteration in range(iters):
            optimizer.zero_grad()
            
            # Compute soft labels
            y_prob = F.softmax(y_logits, dim=-1)
            
            # Forward pass
            logits = model(x_hat)
            
            # Compute loss for gradient matching
            if batch_size == 1:
                loss_ce = -(y_prob * F.log_softmax(logits, dim=-1)).sum()
            else:
                loss_ce = F.cross_entropy(logits, y_prob.argmax(dim=-1), reduction='mean')
            
            # Compute gradients
            grads = torch.autograd.grad(loss_ce, model.parameters(), 
                                      create_graph=True, retain_graph=True)
            grad_vec = torch.cat([g.contiguous().view(-1) for g in grads])
            
            # Main gradient matching loss
            grad_diff = grad_vec - g_star.to(grad_vec.device)
            loss_grad = grad_diff.pow(2).sum()
            
            # Regularization terms
            loss_l2 = lambda_l2 * x_hat.pow(2).sum()
            loss_tv = lambda_tv * total_variation_loss(x_hat)
            loss_gc = lambda_gc * group_consistency_loss(x_hat) if batch_size > 1 else 0
            
            # Total loss
            # total_loss = loss_grad + loss_l2
            total_loss = loss_grad + loss_l2 + loss_tv + loss_gc
            
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([x_hat, y_logits], max_norm=1.0)
            
            optimizer.step()
            if use_scheduler:
                scheduler.step()
            
            # Project to valid range
            with torch.no_grad():
                if normalize_data:
                    # For normalized data, use appropriate bounds
                    x_hat.clamp_(-2.0, 2.0)
                else:
                    x_hat.clamp_(0.0, 1.0)
            
            current_loss = total_loss.item()
            loss_trace.append(current_loss)
            
            # Early stopping
            if early_stopping:
                if current_loss < best_iter_loss:
                    best_iter_loss = current_loss
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve > patience:
                        break
        
        # Check if this restart is the best
        final_loss = loss_trace[-1] if loss_trace else float('inf')
        if final_loss < best_loss:
            best_loss = final_loss
            predicted_labels = y_logits.argmax(dim=-1).cpu().numpy()
            best_result = (x_hat.detach().cpu(), predicted_labels, loss_trace, restart)
    
    x_recon, y_pred, trace, best_restart = best_result
    return x_recon, y_pred, trace, best_restart

# --------------------------
# Enhanced Evaluation Metrics
# --------------------------
def compute_metrics(x_gt, x_recon, y_gt, y_pred):
    """Compute comprehensive evaluation metrics"""
    # MSE
    mse = F.mse_loss(x_recon, x_gt.cpu()).item()
    
    # PSNR
    psnr = 20 * torch.log10(1.0 / torch.sqrt(torch.tensor(mse)))
    
    # Structural Similarity (simplified)
    def ssim_simple(img1, img2):
        mu1 = img1.mean()
        mu2 = img2.mean()
        sigma1_sq = ((img1 - mu1) ** 2).mean()
        sigma2_sq = ((img2 - mu2) ** 2).mean()
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))
        return ssim.item()
    
    ssim = ssim_simple(x_gt.cpu().squeeze(), x_recon.squeeze())
    
    # Label accuracy
    if isinstance(y_pred, (list, np.ndarray)):
        label_acc = (y_gt.cpu().numpy() == y_pred).mean()
    else:
        label_acc = float(y_gt.item() == y_pred)
    
    return {
        'mse': mse,
        'psnr': psnr.item(),
        'ssim': ssim,
        'label_accuracy': label_acc
    }

# --------------------------
# Comprehensive Experiment
# --------------------------
# --------------------------
# 5. EXPERIMENTO PRINCIPAL PARA Fashion-MNIST
# --------------------------
def comprehensive_fashionmnist_experiment():
    """Experimento abrangente para Fashion-MNIST"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    configs = [
        {'normalize': False, 'hidden_size': 128, 'use_bn': False},
        {'normalize': True, 'hidden_size': 256, 'use_bn': True},
    ]
    
    all_results = []
    x_gt = None
    y_gt = None
    
    for config_id, config in enumerate(configs):
        print(f"\n=== Fashion-MNIST Configuration {config_id + 1}: {config} ===")
        
        # Usar Fashion-MNIST dataloader
        dataloader, dataset = get_fashionmnist_dataloader(normalize=config['normalize'])
        model = create_model(hidden_size=config['hidden_size'], use_bn=config['use_bn'])
        
        # Fashion-MNIST pode precisar de mais treinamento que MNIST
        train_model(model, dataloader, device, epochs=1, lr=1e-3)
        
        # Sample data
        if x_gt is None and y_gt is None:
            x_gt, y_gt = random.choice(dataset)
            x_gt = x_gt.unsqueeze(0).to(device)
            y_gt = torch.tensor([y_gt], device=device)
            print(f"Selected Fashion-MNIST sample: {get_fashion_label_name(y_gt.item())}")
        
        for exp_id in range(5):
            print(f"\nFashion-MNIST Experiment {exp_id + 1}/5")
            
            # Compute ground truth gradient
            g_star = compute_gradient_robust(model, x_gt, y_gt, normalize=True)
            
            # Fashion-MNIST pode precisar de mais iterações devido à complexidade
            x_recon, y_pred, loss_trace, best_restart = reconstruct_advanced(
                model, g_star, batch_size=1,
                lambda_l2=1e-2, lambda_tv=2e-4,  # TV um pouco maior para Fashion-MNIST
                iters=1200, lr=0.04, restarts=4,  # Mais iterações e restarts
                device=device, normalize_data=config['normalize']
            )
            
            # Evaluate
            metrics = compute_metrics(x_gt, x_recon, y_gt, y_pred[0])
            
            # Save results
            result = {
                'dataset': 'fashion-mnist',
                'config_id': config_id,
                'exp_id': exp_id,
                'normalize': config['normalize'],
                'hidden_size': config['hidden_size'],
                'use_bn': config['use_bn'],
                'best_restart': best_restart,
                'y_gt': y_gt.item(),
                'y_gt_name': get_fashion_label_name(y_gt.item()),
                'y_pred': y_pred[0],
                'y_pred_name': get_fashion_label_name(y_pred[0]),
                **metrics
            }
            all_results.append(result)
            
            print(f"True: {get_fashion_label_name(y_gt.item())}, "
                  f"Pred: {get_fashion_label_name(y_pred[0])}")
            print(f"MSE: {metrics['mse']:.6f}, PSNR: {metrics['psnr']:.2f}, "
                  f"SSIM: {metrics['ssim']:.4f}, Label Acc: {metrics['label_accuracy']:.0f}")
            
            # Visualizações melhoradas
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(loss_trace)
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Convergence')
            plt.grid(True)
            plt.yscale('log')
            
            plt.subplot(1, 3, 2)
            plt.imshow(x_gt.cpu().squeeze(), cmap='gray')
            plt.title(f'Ground Truth\n{get_fashion_label_name(y_gt.item())}')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(x_recon.squeeze(), cmap='gray')
            plt.title(f'Reconstructed\n{get_fashion_label_name(y_pred[0])}')
            plt.axis('off')
                        
            plt.tight_layout()
            plt.savefig(f'{DIR}/fashionmnist_config{config_id}_exp{exp_id}.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
    
    # Save results com campos adicionais
    fieldnames = ['dataset', 'config_id', 'exp_id', 'normalize', 'hidden_size', 'use_bn', 
                  'best_restart', 'y_gt', 'y_gt_name', 'y_pred', 'y_pred_name', 
                  'mse', 'psnr', 'ssim', 'label_accuracy']
    
    with open(f'{DIR}/fashionmnist_comprehensive_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    
    print_fashionmnist_summary(all_results)
    return all_results

# --------------------------
# 6. ESTUDO ARQUITETURAL PARA Fashion-MNIST
# --------------------------
def comprehensive_fashionmnist_architectural_study():
    """Estudo arquitetural para Fashion-MNIST (mais complexo que MNIST)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} for Fashion-MNIST study")
    
    # Arquiteturas podem precisar ser maiores para Fashion-MNIST
    architectures = {
        'small_mlp': lambda: create_model(hidden_size=64),
        'standard_mlp': lambda: create_model(hidden_size=128),
        'large_mlp': lambda: create_model(hidden_size=256),  # Maior para Fashion-MNIST
        'deep_mlp': lambda: create_deep_mlp([512, 256, 128]),  # Mais capacidade
        'cnn': create_cnn_model,
        'with_dropout': lambda: create_model_with_dropout(dropout_rate=0.3),
        'with_batchnorm': lambda: create_model(hidden_size=256, use_bn=True)  # Maior
    }
    
    # Métodos de ataque otimizados para Fashion-MNIST
    # attack_methods = {
    #     'basic': lambda model, g_star, **kwargs: basic_gia_attack(
    #         model, g_star, iters=600, **kwargs),
    #     'tv_regularized': lambda model, g_star, **kwargs: tv_gia_attack(
    #         model, g_star, iters=1000, lambda_tv=2e-4, **kwargs),
    #     'frequency_regularized': freq_gia_attack,
    #     'multi_term': lambda model, g_star, **kwargs: multi_term_gia_attack(
    #         model, g_star, iters=1200, **kwargs)
    # }
    
    attack_methods = {
        'basic': basic_gia_attack,
        'tv_regularized': tv_gia_attack,
        'frequency_regularized': freq_gia_attack,
        'multi_term': multi_term_gia_attack
    }
    
    configs = [
        {'normalize': False, 'name': 'unnormalized'},
        {'normalize': True, 'name': 'normalized'}
    ]
    
    all_results = []
    
    # Dados Fashion-MNIST
    dataloader, dataset = get_fashionmnist_dataloader(batch_size=64, normalize=False)
    x_gt, y_gt = random.choice(dataset)
    x_gt = x_gt.unsqueeze(0).to(device)
    y_gt = torch.tensor([y_gt], device=device)
    
    print(f"Testing with Fashion-MNIST item: {get_fashion_label_name(y_gt.item())}")
    
    for config in configs:
        print(f"\n{'='*50}")
        print(f"Fashion-MNIST Configuration: {config['name']}")
        print(f"{'='*50}")
        
        if config['normalize']:
            transform = transforms.Normalize((0.2860,), (0.3530,))  # Fashion-MNIST stats
            x_test = transform(x_gt.squeeze(0)).unsqueeze(0)
        else:
            x_test = x_gt
        
        for arch_name, model_func in architectures.items():
            print(f"\nTesting Fashion-MNIST with architecture: {arch_name}")
            
            if callable(model_func):
                model = model_func()
            else:
                model = model_func
                
            # Usar Fashion-MNIST dataloader
            if config['normalize']:
                train_loader, _ = get_fashionmnist_dataloader(normalize=True)
            else:
                train_loader, _ = get_fashionmnist_dataloader(normalize=False)
                
            # Fashion-MNIST precisa de mais treinamento
            train_model(model, train_loader, device, epochs=1, lr=1e-3)
            
            for attack_name, attack_func in attack_methods.items():
                print(f"  Fashion-MNIST Attack: {attack_name}")
                
                try:
                    result = evaluate_attack(
                        model, attack_func, x_test, y_gt, 
                        device=device, normalize_data=config['normalize']
                    )
                    
                    result_entry = {
                        'dataset': 'fashion-mnist',
                        'config': config['name'],
                        'architecture': arch_name,
                        'attack': attack_name,
                        'true_class': get_fashion_label_name(y_gt.item()),
                        'pred_class': get_fashion_label_name(result['y_pred'][0]),
                        'mse': result['metrics']['mse'],
                        'psnr': result['metrics']['psnr'],
                        'ssim': result['metrics']['ssim'],
                        'label_accuracy': result['metrics']['label_accuracy'],
                        'final_loss': result['loss_trace'][-1] if result['loss_trace'] else float('inf'),
                        'best_restart': result['best_restart']
                    }
                    
                    all_results.append(result_entry)
                    
                    print(f"    True: {get_fashion_label_name(y_gt.item())}, "
                          f"Pred: {get_fashion_label_name(result['y_pred'][0])}")
                    print(f"    MSE: {result['metrics']['mse']:.6f}, "
                          f"PSNR: {result['metrics']['psnr']:.2f}, "
                          f"SSIM: {result['metrics']['ssim']:.4f}")
                    
                    save_fashion_comparison_image(
                        x_test, result['x_recon'], y_gt, result['y_pred'],
                        f"{DIR}/fashionmnist_{config['name']}_{arch_name}_{attack_name}.png"
                    )
                        
                except Exception as e:
                    print(f"    Error with Fashion-MNIST: {str(e)}")
                    continue
    
    save_fashionmnist_comprehensive_results(all_results)
    analyze_fashionmnist_results(all_results)
    
    return all_results

# --------------------------
# 7. FUNÇÕES AUXILIARES PARA Fashion-MNIST
# --------------------------
def print_fashionmnist_summary(all_results):
    """Summary específico para Fashion-MNIST"""
    print("\n" + "="*60)
    print("Fashion-MNIST GIA ATTACK SUMMARY STATISTICS")
    print("="*60)
    
    config_results = {}
    for result in all_results:
        config_id = result['config_id']
        if config_id not in config_results:
            config_results[config_id] = {'mse': [], 'psnr': [], 'ssim': [], 'label_accuracy': []}
        
        config_results[config_id]['mse'].append(result['mse'])
        config_results[config_id]['psnr'].append(result['psnr'])
        config_results[config_id]['ssim'].append(result['ssim'])
        config_results[config_id]['label_accuracy'].append(result['label_accuracy'])
    
    for config_id, metrics in config_results.items():
        print(f"\nFashion-MNIST Configuration {config_id + 1}:")
        print(f"  Average MSE: {np.mean(metrics['mse']):.6f} ± {np.std(metrics['mse']):.6f}")
        print(f"  Average PSNR: {np.mean(metrics['psnr']):.2f} ± {np.std(metrics['psnr']):.2f}")
        print(f"  Average SSIM: {np.mean(metrics['ssim']):.4f} ± {np.std(metrics['ssim']):.4f}")
        print(f"  Label Accuracy: {np.mean(metrics['label_accuracy']):.2f}")

def save_fashionmnist_comprehensive_results(results):
    """Salvar resultados Fashion-MNIST"""
    fieldnames = ['dataset', 'config', 'architecture', 'attack', 'true_class', 'pred_class',
                  'mse', 'psnr', 'ssim', 'label_accuracy', 'final_loss', 'best_restart']
    
    with open(f'{DIR}/fashionmnist_comprehensive_architectural_study.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def analyze_fashionmnist_results(results):
    """Análise específica para Fashion-MNIST"""
    print("\n" + "="*60)
    print("Fashion-MNIST COMPREHENSIVE ANALYSIS RESULTS")
    print("="*60)
    
    # Análise por arquitetura
    arch_results = {}
    for result in results:
        arch = result['architecture']
        if arch not in arch_results:
            arch_results[arch] = {'mse': [], 'psnr': [], 'ssim': [], 'label_acc': []}
        
        arch_results[arch]['mse'].append(result['mse'])
        arch_results[arch]['psnr'].append(result['psnr'])
        arch_results[arch]['ssim'].append(result['ssim'])
        arch_results[arch]['label_acc'].append(result['label_accuracy'])
    
    print("\nFashion-MNIST RESULTS BY ARCHITECTURE:")
    for arch, metrics in arch_results.items():
        print(f"\n{arch.upper()} (Fashion-MNIST):")
        print(f"  MSE: {np.mean(metrics['mse']):.6f} ± {np.std(metrics['mse']):.6f}")
        print(f"  PSNR: {np.mean(metrics['psnr']):.2f} ± {np.std(metrics['psnr']):.2f}")
        print(f"  SSIM: {np.mean(metrics['ssim']):.4f} ± {np.std(metrics['ssim']):.4f}")
        print(f"  Label Acc: {np.mean(metrics['label_acc']):.2f} ± {np.std(metrics['label_acc']):.2f}")

    # Análise por método de ataque
    attack_results = {}
    for result in results:
        attack = result['attack']
        if attack not in attack_results:
            attack_results[attack] = {'mse': [], 'psnr': [], 'ssim': [], 'label_acc': []}
        
        attack_results[attack]['mse'].append(result['mse'])
        attack_results[attack]['psnr'].append(result['psnr'])
        attack_results[attack]['ssim'].append(result['ssim'])
        attack_results[attack]['label_acc'].append(result['label_accuracy'])
    
    print("\nFashion-MNIST RESULTS BY ATTACK METHOD:")
    for attack, metrics in attack_results.items():
        print(f"\n{attack.upper()} (Fashion-MNIST):")
        print(f"  MSE: {np.mean(metrics['mse']):.6f} ± {np.std(metrics['mse']):.6f}")
        print(f"  PSNR: {np.mean(metrics['psnr']):.2f} ± {np.std(metrics['psnr']):.2f}")
        print(f"  SSIM: {np.mean(metrics['ssim']):.4f} ± {np.std(metrics['ssim']):.4f}")
        print(f"  Label Acc: {np.mean(metrics['label_acc']):.2f} ± {np.std(metrics['label_acc']):.2f}")



# =============================================================================
# ADICIONAR ESSAS FUNÇÕES AO SEU CÓDIGO EXISTENTE
# =============================================================================

# --------------------------
# 1. NOVAS ARQUITETURAS
# --------------------------

def create_cnn_model():
    """CNN model for comparison with MLP"""
    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(64 * 16, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

def create_deep_mlp(hidden_sizes=[256, 128, 64]):
    """Deep MLP with multiple hidden layers"""
    layers = [nn.Flatten()]
    prev_size = 784
    
    for hidden_size in hidden_sizes:
        layers.extend([
            nn.Linear(prev_size, hidden_size),
            nn.ReLU()
        ])
        prev_size = hidden_size
    
    layers.append(nn.Linear(prev_size, 10))
    return nn.Sequential(*layers)

def create_model_with_dropout(hidden_size=128, dropout_rate=0.3):
    """Model with dropout for regularization study"""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_size, 10)
    )

# --------------------------
# 2. NOVAS FUNÇÕES DE REGULARIZAÇÃO
# --------------------------

def frequency_regularization(x, alpha=1e-4):
    """Regularização no domínio da frequência"""
    # Aplicar FFT 2D
    fft = torch.fft.fft2(x)
    
    # Criar máscara para altas frequências (bordas do espectro)
    h, w = x.shape[-2:]
    center_h, center_w = h // 2, w // 2
    y, x_coord = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    
    # Distância do centro
    dist = torch.sqrt((y - center_h)**2 + (x_coord - center_w)**2)
    high_freq_mask = (dist > min(h, w) * 0.3).float().to(x.device)
    
    # Penalizar altas frequências
    high_freq_penalty = torch.sum(torch.abs(fft) * high_freq_mask)
    return alpha * high_freq_penalty

def edge_preserving_tv(x, beta=0.3):
    """Total Variation que preserva bordas importantes"""
    # Gradientes horizontais e verticais
    dx = x[:, :, :-1, :] - x[:, :, 1:, :]
    dy = x[:, :, :, :-1] - x[:, :, :, 1:]
    
    # Magnitude dos gradientes
    dx_abs = torch.abs(dx)
    dy_abs = torch.abs(dy)
    
    # Penalidade menor para gradientes grandes (bordas)
    tv_x = torch.sum(dx_abs / (1 + beta * dx_abs))
    tv_y = torch.sum(dy_abs / (1 + beta * dy_abs))
    
    return tv_x + tv_y

# --------------------------
# 3. MÉTODOS DE ATAQUE APRIMORADOS
# --------------------------

def basic_gia_attack(model, g_star, batch_size=1, device='cpu', **kwargs):
    """Ataque GIA básico (sua implementação atual simplificada)"""
    return reconstruct_advanced(
        model, g_star, batch_size=batch_size,
        lambda_l2=1e-2, lambda_tv=0, 
        iters=600, lr=0.05, restarts=1,
        device=device, **kwargs
    )

def tv_gia_attack(model, g_star, batch_size=1, device='cpu', **kwargs):
    """Ataque GIA com regularização TV"""
    return reconstruct_advanced(
        model, g_star, batch_size=batch_size,
        lambda_l2=1e-2, lambda_tv=2e-4,
        iters=1000, lr=0.05, restarts=3,
        device=device, **kwargs
    )

def freq_gia_attack(model, g_star, batch_size=1, device='cpu', normalize_data=False, **kwargs):
    """Ataque GIA com regularização de frequência"""
    
    # Modificar a função de reconstrução para incluir freq regularization
    best_result = None
    best_loss = float('inf')
    
    for restart in range(3):
        # Inicialização
        if normalize_data:
            x_hat = torch.randn(batch_size, 1, 28, 28, device=device) * 0.1
        else:
            x_hat = torch.rand(batch_size, 1, 28, 28, device=device) * 0.8 + 0.1
            
        y_logits = torch.randn(batch_size, 10, device=device) * 0.1
        x_hat = x_hat.detach().requires_grad_(True)
        y_logits = y_logits.detach().requires_grad_(True)
        
        optimizer = optim.Adam([x_hat, y_logits], lr=0.05)
        
        for iteration in range(800):
            optimizer.zero_grad()
            
            y_prob = F.softmax(y_logits, dim=-1)
            logits = model(x_hat)
            
            if batch_size == 1:
                loss_ce = -(y_prob * F.log_softmax(logits, dim=-1)).sum()
            else:
                loss_ce = F.cross_entropy(logits, y_prob.argmax(dim=-1), reduction='mean')
            
            grads = torch.autograd.grad(loss_ce, model.parameters(), 
                                      create_graph=True, retain_graph=True)
            grad_vec = torch.cat([g.contiguous().view(-1) for g in grads])
            
            # Losses
            loss_grad = (grad_vec - g_star.to(grad_vec.device)).pow(2).sum()
            loss_l2 = 1e-2 * x_hat.pow(2).sum()
            loss_freq = frequency_regularization(x_hat)  # NOVA REGULARIZAÇÃO
            
            total_loss = loss_grad + loss_l2 + loss_freq
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_([x_hat, y_logits], max_norm=1.0)
            optimizer.step()
            
            with torch.no_grad():
                if normalize_data:
                    x_hat.clamp_(-2.0, 2.0)
                else:
                    x_hat.clamp_(0.0, 1.0)
        
        final_loss = total_loss.item()
        if final_loss < best_loss:
            best_loss = final_loss
            predicted_labels = y_logits.argmax(dim=-1).cpu().numpy()
            best_result = (x_hat.detach().cpu(), predicted_labels, [final_loss], restart)
    
    return best_result

def multi_term_gia_attack(model, g_star, batch_size=1, device='cpu', normalize_data=False, **kwargs):
    """Ataque GIA com múltiplos termos de regularização"""
    
    best_result = None
    best_loss = float('inf')
    
    for restart in range(5):  # Mais restarts para melhor resultado
        # Inicialização
        if normalize_data:
            x_hat = torch.randn(batch_size, 1, 28, 28, device=device) * 0.1
        else:
            x_hat = torch.rand(batch_size, 1, 28, 28, device=device) * 0.8 + 0.1
            
        y_logits = torch.randn(batch_size, 10, device=device) * 0.1
        x_hat = x_hat.detach().requires_grad_(True)
        y_logits = y_logits.detach().requires_grad_(True)
        
        # Usar diferentes otimizadores por restart
        if restart < 3:
            optimizer = optim.Adam([x_hat, y_logits], lr=0.05, betas=(0.9, 0.999))
        else:
            optimizer = optim.RMSprop([x_hat, y_logits], lr=0.03, alpha=0.99)
        
        for iteration in range(1200):
            optimizer.zero_grad()
            
            y_prob = F.softmax(y_logits, dim=-1)
            logits = model(x_hat)
            
            if batch_size == 1:
                loss_ce = -(y_prob * F.log_softmax(logits, dim=-1)).sum()
            else:
                loss_ce = F.cross_entropy(logits, y_prob.argmax(dim=-1), reduction='mean')
            
            grads = torch.autograd.grad(loss_ce, model.parameters(), 
                                      create_graph=True, retain_graph=True)
            grad_vec = torch.cat([g.contiguous().view(-1) for g in grads])
            
            # Múltiplos termos de regularização
            loss_grad = (grad_vec - g_star.to(grad_vec.device)).pow(2).sum()
            loss_l2 = 1e-2 * x_hat.pow(2).sum()
            loss_tv = 1e-4 * total_variation_loss(x_hat)
            loss_freq = frequency_regularization(x_hat)
            loss_edge_tv = 1e-5 * edge_preserving_tv(x_hat)
            
            # Pesos adaptativos baseados na iteração
            w_freq = min(1.0, iteration / 200)  # Aumenta com o tempo
            w_edge = max(0.5, 1.0 - iteration / 800)  # Diminui com o tempo
            
            total_loss = (loss_grad + loss_l2 + loss_tv + 
                         w_freq * loss_freq + w_edge * loss_edge_tv)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([x_hat, y_logits], max_norm=1.0)
            optimizer.step()
            
            with torch.no_grad():
                if normalize_data:
                    x_hat.clamp_(-2.0, 2.0)
                else:
                    x_hat.clamp_(0.0, 1.0)
        
        final_loss = total_loss.item()
        if final_loss < best_loss:
            best_loss = final_loss
            predicted_labels = y_logits.argmax(dim=-1).cpu().numpy()
            best_result = (x_hat.detach().cpu(), predicted_labels, [final_loss], restart)
    
    return best_result

# --------------------------
# 4. FUNÇÃO PARA AVALIAR ATAQUES
# --------------------------

def evaluate_attack(model, attack_func, x_gt, y_gt, device='cpu', normalize_data=False):
    """Avaliar a qualidade de um ataque específico"""
    
    # Computar gradiente ground truth
    g_star = compute_gradient_robust(model, x_gt, y_gt, normalize=True)
    
    # Executar ataque
    x_recon, y_pred, loss_trace, best_restart = attack_func(
        model, g_star, batch_size=1, device=device, normalize_data=normalize_data
    )
    
    # Avaliar qualidade
    metrics = compute_metrics(x_gt, x_recon, y_gt, y_pred[0])
    
    return {
        'metrics': metrics,
        'loss_trace': loss_trace,
        'best_restart': best_restart,
        'x_recon': x_recon,
        'y_pred': y_pred
    }

# --------------------------
# 6. FUNÇÕES AUXILIARES
# --------------------------

def save_comparison_image(x_gt, x_recon, y_gt, y_pred, filename):
    """Salvar comparação visual"""
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(x_gt.cpu().squeeze(), cmap='gray')
    plt.title(f'Ground Truth (Label: {y_gt.item()})')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(x_recon.squeeze(), cmap='gray')
    plt.title(f'Reconstructed (Label: {y_pred[0]})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


# --------------------------
# 8. MAIN ATUALIZADO PARA Fashion-MNIST
# --------------------------
if __name__ == "__main__":
    seed = 51
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    print("="*60)
    print("Fashion-MNIST GRADIENT INVERSION ATTACK STUDY")
    print("="*60)
    
    print("Fashion-MNIST classes:", list(FASHION_MNIST_LABELS.values()))
    
    # Escolher experimento
    # experiment_choice = input("Choose experiment:\n1. Basic Fashion-MNIST experiment\n2. Comprehensive Fashion-MNIST architectural study\nChoice (1 or 2): ")
    
    # if experiment_choice == "1":
    #     results = comprehensive_fashionmnist_experiment()
    # elif experiment_choice == "2":
    #     results = comprehensive_fashionmnist_architectural_study()
    # else:
    #     print("Running default Fashion-MNIST experiment...")
    results1 = comprehensive_fashionmnist_experiment()
    # results2 = comprehensive_fashionmnist_architectural_study()
        
    print(f"\nExperiment completed! Results saved in {DIR}/")
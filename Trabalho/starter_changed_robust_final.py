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
import os

DIR = "imgs_reconstructed"
os.makedirs(DIR, exist_ok=True)

# --------------------------
# Dataset and Model Improvements
# --------------------------
def get_mnist_dataloader(batch_size=64, normalize=True):
    """Load MNIST with optional normalization"""
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST statistics
        ])
    else:
        transform = transforms.ToTensor()
    
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, dataset

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
                        device='cpu', normalize_data=False, lf=False, mt=False):
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
        
        # NOT USED 
        # Use different optimizers for different restarts
        # if restart < restarts // 2:
        #     optimizer = optim.Adam([x_hat, y_logits], lr=lr, betas=(0.9, 0.999))
        # else:
        #     optimizer = optim.RMSprop([x_hat, y_logits], lr=lr*0.5, alpha=0.99)
        
        optimizer = optim.Adam([x_hat, y_logits], lr=lr, betas=(0.9, 0.999))
            
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
            
            if lf or mt:
                loss_freq = frequency_regularization(x_hat)  # NOVA REGULARIZAÇÃO
            else:
                loss_freq = 0
                
            if mt:
                loss_edge_tv = 1e-5 * edge_preserving_tv(x_hat)
                w_freq = min(1.0, iteration / 200)  # Aumenta com o tempo
                w_edge = max(0.5, 1.0 - iteration / 800)  # Diminui com o tempo
            else:
                loss_edge_tv = 0
                w_freq = 1
                w_edge = 1
                
            
            loss_gc = lambda_gc * group_consistency_loss(x_hat) if batch_size > 1 else 0
            
            # Total loss
            # total_loss = loss_grad + loss_l2
            
            # DEBUG
            # if iteration % 100 == 0 or mt and (w_freq * loss_freq == 0 or w_edge * loss_edge_tv == 0):
            #     print("deu ruim")
            #     print(loss_freq.item() if torch.is_tensor(loss_freq) else loss_freq)
            #     print(w_freq)
            #     print((w_edge * loss_edge_tv).item())
            
            total_loss = loss_grad + loss_l2 + loss_tv + w_freq * loss_freq + w_edge * loss_edge_tv + loss_gc
            
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
# 1. NOVAS ARQUITETURAS
# --------------------------

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

def classic_gia_attack_original(model, g_star, batch_size=1, device='cpu', normalize_data=False):
    # Reconstruct a single image + label from leaked gradient
    # torch.manual_seed(seed)
    x_hat = torch.rand(1, 1, 28, 28, requires_grad=True, device=device)  # random initial image
    y_logits = torch.zeros(1, 10, requires_grad=True, device=device)     # learnable label logits
    optimizer = optim.LBFGS([x_hat, y_logits], lr=0.1, max_iter=20)

    loss_trace = []  # to store objective value at each iteration

    def closure():
        optimizer.zero_grad()
        y_prob = y_logits.softmax(dim=-1)  # turn logits into probabilities
        loss = (model(x_hat) * y_prob).sum()  # soft one-hot prediction
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        grad_diff = torch.cat([g.view(-1) for g in grads]) - g_star
        obj = grad_diff.pow(2).sum() + 1e-4 * x_hat.pow(2).sum()
        obj.backward()
        loss_trace.append(obj.item())
        return obj

    for _ in range(300):
        optimizer.step(closure)

    # predicted_label = y_logits.argmax(dim=-1).item()
    predicted_label = y_logits.argmax(dim=-1).cpu().numpy()
    return x_hat.detach().cpu(), predicted_label, loss_trace, 0

def basic_gia_attack(model, g_star, batch_size=1, device='cpu', **kwargs):
    """Ataque GIA básico (sua implementação atual simplificada)"""
    return reconstruct_advanced(
        model, g_star, batch_size=batch_size,
        lambda_l2=1e-2, lambda_tv=0, 
        iters=800, lr=0.05, restarts=1,
        device=device, **kwargs
    )

def tv_gia_attack(model, g_star, batch_size=1, device='cpu', **kwargs):
    """Ataque GIA com regularização TV"""
    return reconstruct_advanced(
        model, g_star, batch_size=batch_size,
        lambda_l2=1e-2, lambda_tv=1e-4,
        iters=800, lr=0.05, restarts=3,
        device=device, **kwargs
    )
    
def freq_gia_attack(model, g_star, batch_size=1, device='cpu', **kwargs):
    """Ataque GIA com regularização freq """
    return reconstruct_advanced(
        model, g_star, batch_size=batch_size,
        lambda_l2=1e-2, lambda_tv=0,
        iters=800, lr=0.05, restarts=3,
        device=device, lf=True, **kwargs
    )
    
def multi_term_gia_attack(model, g_star, batch_size=1, device='cpu', **kwargs):
    """Ataque GIA com regularização mt """
    return reconstruct_advanced(
        model, g_star, batch_size=batch_size,
        lambda_l2=1e-2, lambda_tv=0,
        iters=800, lr=0.05, restarts=3,
        device=device, mt=True, **kwargs
    )

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
# 5. ESTUDO ARQUITETURAL COMPLETO
# --------------------------

def comprehensive_architectural_study():
    """
    Investigar como diferentes arquiteturas afetam GIA + Aprimoramentos nos ataques
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # PARTE 1: Variação Arquitetural
    architectures = {
        'shallow_mlp': lambda: create_model(hidden_size=64),
        'standard_mlp': lambda: create_model(hidden_size=128),
        'deep_mlp': lambda: create_deep_mlp([256, 128, 64]),
        'cnn': create_cnn_model,
        'with_dropout': lambda: create_model_with_dropout(dropout_rate=0.3),
        'with_batchnorm': lambda: create_model(hidden_size=128, use_bn=True)
    }

    
    # PARTE 2: Métodos de Ataque
    attack_methods = {
        'classic': classic_gia_attack_original,
        'basic': basic_gia_attack,
        'tv_regularized': tv_gia_attack,
        'frequency_regularized': freq_gia_attack,
        'multi_term': multi_term_gia_attack
    }
    
    # Configurações experimentais
    configs = [
        {'normalize': False, 'name': 'unnormalized'},
        {'normalize': True, 'name': 'normalized'}
    ]
    
    all_results = []
    
    # Obter dados de teste
    dataloader, dataset = get_mnist_dataloader(batch_size=64, normalize=False)
    x_gt, y_gt = random.choice(dataset)
    x_gt = x_gt.unsqueeze(0).to(device)
    y_gt = torch.tensor([y_gt], device=device)
    
    for config in configs:
        print(f"\n{'='*50}")
        print(f"Configuration: {config['name']}")
        print(f"{'='*50}")
        
        # Ajustar dados se necessário
        if config['normalize']:
            transform = transforms.Normalize((0.1307,), (0.3081,))
            x_test = transform(x_gt.squeeze(0)).unsqueeze(0)
        else:
            x_test = x_gt
        
        for arch_name, model_func in architectures.items():
            print(f"\nTesting architecture: {arch_name}")
            
            # Criar e treinar modelo
            if callable(model_func):
                model = model_func()
            else:
                model = model_func
                
            # Treinamento rápido
            if config['normalize']:
                train_loader, _ = get_mnist_dataloader(normalize=True)
            else:
                train_loader, _ = get_mnist_dataloader(normalize=False)
                
            train_model(model, train_loader, device, epochs=1, lr=1e-3)
            
            for attack_name, attack_func in attack_methods.items():
                print(f"  Attack: {attack_name}")
                
                try:
                    # Executar ataque
                    result = evaluate_attack(
                        model, attack_func, x_test, y_gt, 
                        device=device, normalize_data=config['normalize']
                    )
                    
                    # Salvar resultado
                    result_entry = {
                        'config': config['name'],
                        'architecture': arch_name,
                        'attack': attack_name,
                        'mse': result['metrics']['mse'],
                        'psnr': result['metrics']['psnr'],
                        'ssim': result['metrics']['ssim'],
                        'label_accuracy': result['metrics']['label_accuracy'],
                        'final_loss': result['loss_trace'][-1] if result['loss_trace'] else float('inf'),
                        'best_restart': result['best_restart']
                    }
                    
                    all_results.append(result_entry)
                    
                    print(f"    MSE: {result['metrics']['mse']:.6f}, "
                          f"PSNR: {result['metrics']['psnr']:.2f}, "
                          f"SSIM: {result['metrics']['ssim']:.4f}")
                    
                    # Salvar imagem de exemplo
                    save_comparison_image(
                        x_test, result['x_recon'], y_gt, result['y_pred'],
                        f"{DIR}/example_{config['name']}_{arch_name}_{attack_name}.png"
                    )
                        
                except Exception as e:
                    print(f"    Error: {str(e)}")
                    continue
    
    # Salvar resultados
    save_comprehensive_results(all_results)
    analyze_comprehensive_results(all_results)
    
    return all_results

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

def save_comprehensive_results(results):
    """Salvar resultados em CSV"""
    fieldnames = ['config', 'architecture', 'attack', 'mse', 'psnr', 'ssim', 
                  'label_accuracy', 'final_loss', 'best_restart']
    
    with open(f'{DIR}/comprehensive_architectural_study.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def analyze_comprehensive_results(results):
    """Analisar e imprimir estatísticas dos resultados"""
    print("\n" + "="*60)
    print("COMPREHENSIVE ANALYSIS RESULTS")
    print("="*60)
    
    # Agrupar por arquitetura
    arch_results = {}
    for result in results:
        arch = result['architecture']
        if arch not in arch_results:
            arch_results[arch] = {'mse': [], 'psnr': [], 'ssim': [], 'label_acc': []}
        
        arch_results[arch]['mse'].append(result['mse'])
        arch_results[arch]['psnr'].append(result['psnr'])
        arch_results[arch]['ssim'].append(result['ssim'])
        arch_results[arch]['label_acc'].append(result['label_accuracy'])
    
    print("\nRESULTS BY ARCHITECTURE:")
    for arch, metrics in arch_results.items():
        print(f"\n{arch.upper()}:")
        print(f"  MSE: {np.mean(metrics['mse']):.6f} ± {np.std(metrics['mse']):.6f}")
        print(f"  PSNR: {np.mean(metrics['psnr']):.2f} ± {np.std(metrics['psnr']):.2f}")
        print(f"  SSIM: {np.mean(metrics['ssim']):.4f} ± {np.std(metrics['ssim']):.4f}")
        print(f"  Label Acc: {np.mean(metrics['label_acc']):.2f} ± {np.std(metrics['label_acc']):.2f}")
    
    # Agrupar por método de ataque
    attack_results = {}
    for result in results:
        attack = result['attack']
        if attack not in attack_results:
            attack_results[attack] = {'mse': [], 'psnr': [], 'ssim': [], 'label_acc': []}
        
        attack_results[attack]['mse'].append(result['mse'])
        attack_results[attack]['psnr'].append(result['psnr'])
        attack_results[attack]['ssim'].append(result['ssim'])
        attack_results[attack]['label_acc'].append(result['label_accuracy'])
    
    print("\nRESULTS BY ATTACK METHOD:")
    for attack, metrics in attack_results.items():
        print(f"\n{attack.upper()}:")
        print(f"  MSE: {np.mean(metrics['mse']):.6f} ± {np.std(metrics['mse']):.6f}")
        print(f"  PSNR: {np.mean(metrics['psnr']):.2f} ± {np.std(metrics['psnr']):.2f}")
        print(f"  SSIM: {np.mean(metrics['ssim']):.4f} ± {np.std(metrics['ssim']):.4f}")
        print(f"  Label Acc: {np.mean(metrics['label_acc']):.2f} ± {np.std(metrics['label_acc']):.2f}")

# --------------------------
# 7. FUNÇÃO MAIN
# --------------------------

if __name__ == "__main__":
    # 47, 49, 51, 54, 68
    seed = 47
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Executar estudo abrangente
    results = comprehensive_architectural_study()
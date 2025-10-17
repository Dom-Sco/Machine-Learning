import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import os
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.nn.utils import clip_grad_norm_

# --------------------------
#  Hyperparameters
# --------------------------
latent_dim = 100
num_classes = 10
embed_dim = 50
batch_size = 128
epochs = 500
lr = 5e-5             
lambda_gp = 5          
n_critic = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Flags & safety thresholds
use_spectral_norm = False       
g_loss_threshold = 2000.0     
g_loss_patience = 3              
max_grad_norm = 5.0              

# ==========================================================
#  Helpers to optionally apply spectral norm
# ==========================================================
def Conv2d(in_c, out_c, k, s, p, bias=True):
    conv = nn.Conv2d(in_c, out_c, k, s, p, bias=bias)
    return spectral_norm(conv) if use_spectral_norm else conv

def ConvTranspose2d(in_c, out_c, k, s, p, bias=True):
    conv = nn.ConvTranspose2d(in_c, out_c, k, s, p, bias=bias)
    return spectral_norm(conv) if use_spectral_norm else conv

def Linear(in_f, out_f, bias=True):
    lin = nn.Linear(in_f, out_f, bias=bias)
    return spectral_norm(lin) if use_spectral_norm else lin

# ==========================================================
#  Conditional BatchNorm2d
# ==========================================================
class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        # sensible init
        nn.init.ones_(self.embed.weight.data[:, :num_features])
        nn.init.zeros_(self.embed.weight.data[:, num_features:])

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, dim=1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        return gamma * out + beta

# ==========================================================
#  Generator (ResNet-style)
# ==========================================================
class GenResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_classes, upsample=True):
        super().__init__()
        self.upsample = upsample
        self.cbn1 = ConditionalBatchNorm2d(in_ch, num_classes)
        self.cbn2 = ConditionalBatchNorm2d(out_ch, num_classes)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.upsample_layer = nn.Upsample(scale_factor=2, mode='nearest')
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1, 1, 0)

    def forward(self, x, y):
        h = self.cbn1(x, y)
        h = torch.relu(h)
        if self.upsample:
            h = self.upsample_layer(h)
        h = self.conv1(h)
        h = self.cbn2(h, y)
        h = torch.relu(h)
        h = self.conv2(h)
        if self.upsample:
            x = self.upsample_layer(x)
        return h + self.shortcut(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 4 * 4 * 256)
        self.blocks = nn.ModuleList([
            GenResBlock(256, 256, num_classes),
            GenResBlock(256, 128, num_classes),
            GenResBlock(128, 64, num_classes),
        ])
        self.bn = ConditionalBatchNorm2d(64, num_classes)
        self.conv_out = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, z, y):
        h = self.fc(z).view(-1, 256, 4, 4)
        for block in self.blocks:
            h = block(h, y)
        h = self.bn(h, y)
        h = torch.relu(h)
        return torch.tanh(self.conv_out(h))

# ==========================================================
#  Critic (ResNet-style) with optional spectral norm
# ==========================================================
class DiscResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.conv1 = Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)
        self.conv2 = Conv2d(out_channels, out_channels, 3, 1, 1, bias=True)
        self.learnable_skip = (in_channels != out_channels)
        if self.learnable_skip:
            self.skip_conv = Conv2d(in_channels, out_channels, 1, 1, 0, bias=True)
        self.avg_pool = nn.AvgPool2d(2)

    def forward(self, x):
        h = F.leaky_relu(x, 0.2, inplace=True)
        h = self.conv1(h)
        h = F.leaky_relu(h, 0.2, inplace=True)
        h = self.conv2(h)
        if self.downsample:
            h = self.avg_pool(h)
        skip = x
        if self.learnable_skip:
            skip = self.skip_conv(skip)
        if self.downsample:
            skip = self.avg_pool(skip)
        return h + skip

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, 32 * 32)
        self.block1 = DiscResBlock(4, 64)
        self.block2 = DiscResBlock(64, 128)
        self.block3 = DiscResBlock(128, 256)
        self.block4 = DiscResBlock(256, 512, downsample=False)
        # final linear wrapped in spectral_norm if flag on
        self.linear = Linear(512, 1, bias=True)
        # init bias small
        if not use_spectral_norm:
            nn.init.zeros_(self.linear.bias)

    def forward(self, img, labels):
        label_map = self.label_emb(labels).view(-1, 1, 32, 32)
        x = torch.cat([img, label_map], dim=1)
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = torch.relu(h)
        h = h.mean([2, 3])
        return self.linear(h).view(-1)

# ==========================================================
#  Gradient Penalty
# ==========================================================
def compute_gradient_penalty(critic, real_imgs, fake_imgs, labels):
    alpha = torch.rand(real_imgs.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)
    d_interpolates = critic(interpolates, labels)
    grad_outputs = torch.ones_like(d_interpolates, device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gp_term = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return lambda_gp * gp_term, gp_term.item()

# ==========================================================
#  Data + dirs
# ==========================================================
os.makedirs("cifar10_images_wgan_gp", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
train_data = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

# ==========================================================
#  Models / optimizers / load latest checkpoint
# ==========================================================
generator = Generator().to(device)
critic = Critic().to(device)
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.9))
optimizer_D = optim.Adam(critic.parameters(), lr=lr, betas=(0.0, 0.9))

checkpoint_dir = "checkpoints"

def load_latest_checkpoint(generator, critic, optimizer_G, optimizer_D):
    ckpts = [f for f in os.listdir(checkpoint_dir) if f.startswith("wgan_gp_resnet_") and f.endswith(".pth")]
    if not ckpts:
        print("🚀 No checkpoints found — starting from scratch.")
        return 1
    ckpts.sort(key=lambda f: int(f.split("_")[-1].split(".")[0]))
    latest_ckpt = ckpts[-1]
    path = os.path.join(checkpoint_dir, latest_ckpt)
    print(f"🔁 Loading checkpoint: {latest_ckpt}")
    ckpt = torch.load(path, map_location=device)
    generator.load_state_dict(ckpt["generator"])
    critic.load_state_dict(ckpt["critic"])
    if "optimizer_G" in ckpt and "optimizer_D" in ckpt:
        optimizer_G.load_state_dict(ckpt["optimizer_G"])
        optimizer_D.load_state_dict(ckpt["optimizer_D"])
    start_epoch = ckpt.get("epoch", 0) + 1
    print(f"✅ Resuming from epoch {start_epoch}")
    return start_epoch

start_epoch = load_latest_checkpoint(generator, critic, optimizer_G, optimizer_D)

# training log
log_path = os.path.join(checkpoint_dir, "training_log.csv")
if not os.path.exists(log_path):
    with open(log_path, "w") as f:
        f.write("epoch,iter,D_loss,G_loss,E_D_real,E_D_fake,GP,critic_grad_norm,gen_grad_norm\n")

# ==========================================================
#  Training loop
# ==========================================================
consecutive_large_g = 0

def get_grad_norm(model):
    total = 0.0
    count = 0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
            count += 1
    return (total ** 0.5) if count > 0 else 0.0

for epoch in range(start_epoch, epochs + 1):
    progress_bar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
    for i, (imgs, labels) in enumerate(progress_bar):
        imgs, labels = imgs.to(device), labels.to(device)
        bs = imgs.size(0)

        # ---------------------
        # Train Critic
        # ---------------------
        z = torch.randn(bs, latent_dim, device=device)
        gen_labels = torch.randint(0, num_classes, (bs,), device=device)
        fake_imgs = generator(z, gen_labels).detach()

        optimizer_D.zero_grad()
        real_validity = critic(imgs, labels)
        fake_validity = critic(fake_imgs, gen_labels)
        gp, gp_scalar = compute_gradient_penalty(critic, imgs, fake_imgs, labels)
        d_loss = fake_validity.mean() - real_validity.mean() + gp
        d_loss.backward()

        # clip critic grads (safety)
        clip_grad_norm_(critic.parameters(), max_grad_norm)
        critic_grad_norm = get_grad_norm(critic)

        optimizer_D.step()

        # ---------------------
        # Train Generator (periodically)
        # ---------------------
        g_loss = torch.tensor(0.0, device=device)
        if i % n_critic == 0:
            optimizer_G.zero_grad()
            z = torch.randn(bs, latent_dim, device=device)
            gen_labels = torch.randint(0, num_classes, (bs,), device=device)
            gen_imgs = generator(z, gen_labels)
            fake_for_g = critic(gen_imgs, gen_labels)
            g_loss = -fake_for_g.mean()
            g_loss.backward()

            # clip generator grads (safety)
            clip_grad_norm_(generator.parameters(), max_grad_norm)
            gen_grad_norm = get_grad_norm(generator)

            optimizer_G.step()
        else:
            gen_grad_norm = 0.0

        # ---------------------
        # Logging per some iterations
        # ---------------------
        if i % 50 == 0:
            e_d_real = real_validity.mean().item()
            e_d_fake = fake_validity.mean().item()
            d_loss_item = d_loss.item()
            g_loss_item = g_loss.item()
            # update tqdm
            progress_bar.set_postfix({
                "D_loss": f"{d_loss_item:.4f}",
                "G_loss": f"{g_loss_item:.4f}",
                "E[D(real)]": f"{e_d_real:.4f}",
                "E[D(fake)]": f"{e_d_fake:.4f}",
                "GP": f"{gp_scalar:.4f}"
            })
            
            with open(log_path, "a") as f:
                f.write(f"{epoch},{i},{d_loss_item:.6f},{g_loss_item:.6f},{e_d_real:.6f},{e_d_fake:.6f},{gp_scalar:.6f},{critic_grad_norm:.6f},{gen_grad_norm:.6f}\n")

        # ---------------------
        # LR backoff safeguard
        # ---------------------
        if isinstance(g_loss, torch.Tensor) and (abs(g_loss.item()) > g_loss_threshold):
            consecutive_large_g += 1
            if consecutive_large_g >= g_loss_patience:
                # halve learning rates
                for param_group in optimizer_G.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5
                for param_group in optimizer_D.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5
                print(f"⚠️  Large |G_loss| detected for {consecutive_large_g} steps — halving LRs. New lr_G={optimizer_G.param_groups[0]['lr']:.2e}, lr_D={optimizer_D.param_groups[0]['lr']:.2e}")
                consecutive_large_g = 0
        else:
            if isinstance(g_loss, torch.Tensor) and (abs(g_loss.item()) <= g_loss_threshold):
                consecutive_large_g = 0

    with torch.no_grad():
        fixed_z = torch.randn(10, latent_dim, device=device)
        fixed_labels = torch.arange(0, 10, device=device)
        gen_imgs = generator(fixed_z, fixed_labels)
        save_image(gen_imgs, f"cifar10_images_wgan_gp/{epoch:03d}.png", nrow=10, normalize=True)

    if epoch % 10 == 0:
        torch.save({
            "epoch": epoch,
            "generator": generator.state_dict(),
            "critic": critic.state_dict(),
            "optimizer_G": optimizer_G.state_dict(),
            "optimizer_D": optimizer_D.state_dict(),
        }, os.path.join(checkpoint_dir, f"wgan_gp_resnet_{epoch:03d}.pth"))
        print(f"💾 Checkpoint saved at epoch {epoch}")

    print(f"✅ Epoch [{epoch}/{epochs}] - last D_loss: {d_loss.item():.4f}, last G_loss: {g_loss.item():.4f}")

print("🎉 Training complete! Images saved in ./cifar10_images_wgan_gp")

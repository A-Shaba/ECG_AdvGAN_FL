import torch, torch.nn.functional as F

class AdvGANWrapper(torch.nn.Module):
    def __init__(self, target_model, G, D, eps=0.03):
        super().__init__()
        self.target = target_model.eval()
        for p in self.target.parameters(): p.requires_grad=False
        self.G, self.D, self.eps = G, D, eps

    def perturb(self, x):
        delta = self.G(x)
        delta = torch.tanh(delta) * self.eps
        return torch.clamp(x + delta, -1, 1), delta

def advgan_losses(D, target_model, x, y, x_adv, lambda_adv=1.0):
    # generator: fool discriminator + increase target loss
    logits_adv = target_model(x_adv)
    g_loss_cls = F.cross_entropy(logits_adv, y)  # make model wrong
    g_loss_gan = -D(x_adv).mean()
    # discriminator: real > fake
    d_loss = (F.softplus(-D(x)).mean() + F.softplus(D(x_adv)).mean())
    g_loss = lambda_adv*g_loss_cls + g_loss_gan
    return g_loss, d_loss

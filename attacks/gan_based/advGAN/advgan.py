# advgan.py
import torch
import torch.nn.functional as F

class AdvGANWrapper(torch.nn.Module):
    """
    Lightweight wrapper combining target model, generator G, discriminator D (D may be None).
    Usage:
        wrap = AdvGANWrapper(target_model, G, D, eps=0.03)
        x_adv, delta = wrap.perturb(x)
    Notes:
        - Expects input x to be in the same normalized range as training (e.g. [-1,1]).
        - G should output a perturbation map which we scale with tanh and eps.
    """
    def __init__(self, target_model, G, D=None, eps=0.03):
        super().__init__()
        self.target = target_model.eval()
        for p in self.target.parameters():
            p.requires_grad = False
        self.G, self.D, self.eps = G, D, eps

    def perturb(self, x):
        """
        Returns (x_adv, delta) where:
          delta = tanh(G(x)) * eps
          x_adv = clamp(x + delta, -1, 1)  # assuming normalized inputs in [-1,1]
        """
        delta_raw = self.G(x)
        delta = torch.tanh(delta_raw) * self.eps
        x_adv = torch.clamp(x + delta, -1.0, 1.0)
        return x_adv, delta


def cw_loss(logits, labels, targeted=False, y_target=None, kappa=0.0):
    """
    Carlini-Wagner style hinge/margin loss.
    Minimizes this quantity:
      - Untargeted:  max(correct_logit - max_other_logit + kappa, 0)
      - Targeted:    max(max_other_logit - target_logit + kappa, 0)
    Returns mean over batch.
    - logits: [B, C]
    - labels: true labels (for untargeted) or ignored in targeted mode
    - y_target: target labels in targeted mode
    """
    c = logits.size(1)
    if targeted:
        assert y_target is not None
        target_logits = logits.gather(1, y_target.unsqueeze(1)).squeeze(1)
        # mask to exclude target class
        mask = torch.ones_like(logits).bool()
        mask.scatter_(1, y_target.unsqueeze(1), False)
        other_logits = logits.masked_fill(~mask, float("-inf"))
        max_other, _ = other_logits.max(dim=1)
        # loss: max(max_other - target_logit + kappa, 0)
        loss = torch.clamp(max_other - target_logits + kappa, min=0.0)
    else:
        one_hot = F.one_hot(labels, num_classes=c).bool()
        true_logits = logits.masked_fill(~one_hot, float("-inf"))
        correct_logit = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        # compute max of other logits
        mask = ~one_hot
        other_logits = logits.masked_fill(~mask, float("-inf"))
        max_other, _ = other_logits.max(dim=1)
        # loss: max(correct_logit - max_other + kappa, 0)
        loss = torch.clamp(correct_logit - max_other + kappa, min=0.0)

    return loss.mean()


def advgan_losses(
    D, target_model, x, y, x_adv,
    lambda_adv=1.0,
    lambda_gan=0.0,
    lambda_pert=0.05,
    targeted=False,
    y_target=None,
    attack_loss_type="cw",   # "ce" or "cw"
    kappa=0.0,
    device=None
):
    """
    Compute generator and discriminator losses for AdvGAN.

    Returns:
      g_loss, d_loss  (both scalars tensors)
    """

    device = device or x.device

    # --- Classification term for generator ---
    logits_adv = target_model(x_adv)

    if attack_loss_type == "ce":
        if targeted:
            if y_target is None:
                raise ValueError("y_target must be provided for targeted attack with CE")
            g_loss_cls = F.cross_entropy(logits_adv, y_target.to(device))
        else:
            # For untargeted CE formulation, we want to maximize classifier loss.
            # Equivalent to minimizing negative CE.
            g_loss_cls = -F.cross_entropy(logits_adv, y.to(device))
    elif attack_loss_type == "cw":
        # CW returns a quantity to minimize; for untargeted mode we use targeted=False
        g_loss_cls = cw_loss(logits_adv, y.to(device), targeted=targeted, y_target=y_target, kappa=kappa)
    else:
        raise ValueError("attack_loss_type must be 'ce' or 'cw'")

    # --- GAN term (make D think x_adv is real) ---
    if D is not None:
        D_x_adv = D(x_adv)
        # Use BCEWithLogitsLoss, labels ones → want D(x_adv) -> 1
        target_ones = torch.ones_like(D_x_adv, device=device)
        g_loss_gan = F.binary_cross_entropy_with_logits(D_x_adv, target_ones)
    else:
        g_loss_gan = torch.tensor(0.0, device=device)

    # --- Perturbation size (L2 on delta) ---
    delta = (x_adv - x).view(x.size(0), -1)
    g_loss_pert = delta.norm(p=2, dim=1).mean()

    # Combine generator losses (weights configurable)
    g_loss = lambda_adv * g_loss_cls
    if lambda_gan > 0 and D is not None:
        g_loss = g_loss + lambda_gan * g_loss_gan
    if lambda_pert > 0:
        g_loss = g_loss + lambda_pert * g_loss_pert

    # --- Discriminator loss (real vs fake) ---
    if D is not None:
        D_real = D(x)
        D_fake = D_x_adv.detach()
        ones = torch.ones_like(D_real, device=device)
        zeros = torch.zeros_like(D_fake, device=device)
        d_loss_real = F.binary_cross_entropy_with_logits(D_real, ones)
        d_loss_fake = F.binary_cross_entropy_with_logits(D_fake, zeros)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
    else:
        d_loss = torch.tensor(0.0, device=device)

    return g_loss, d_loss
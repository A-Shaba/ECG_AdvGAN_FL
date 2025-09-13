import torch
import torch.nn.functional as F

class AdvGANWrapper(torch.nn.Module):
    def __init__(self, target_model, G, D, eps=0.03):
        super().__init__()
        self.target = target_model.eval()
        for p in self.target.parameters():
            p.requires_grad = False
        self.G, self.D, self.eps = G, D, eps

    def perturb(self, x):
        delta = self.G(x)
        # Limit perturbation magnitude
        delta = torch.tanh(delta) * self.eps
        # Return perturbed sample + perturbation itself
        return torch.clamp(x + delta, -1, 1), delta


def advgan_losses(
    D, target_model, x, y, x_adv,
    lambda_adv=5.0,
    lambda_gan=0.0,

    lambda_pert=0.05,
    targeted=False,
    y_target=None,
    attack_loss_type="ce",   # "ce" or "cw"
    kappa=0
):
    """
    Compute AdvGAN losses for Generator (G) and Discriminator (D).

    Args:
        D: discriminator network
        target_model: victim classifier
        x: original batch
        y: true labels
        x_adv: adversarial examples
        lambda_adv: weight for adversarial classification loss
        lambda_pert: weight for perturbation regularization
        targeted: if True → targeted attack, else untargeted
        y_target: required if targeted=True
    """

    # === Generator loss ===
    logits_adv = target_model(x_adv)


    if targeted:
        if y_target is None:
            raise ValueError("y_target must be provided for targeted attack")
        # In targeted attacks we minimize CE wrt target label
        g_loss_cls = F.cross_entropy(logits_adv, y_target)

    else:
        # In untargeted attacks we maximize CE wrt true label
        # g_loss_cls = -F.cross_entropy(logits_adv, y)
        g_loss_cls = cw_loss(logits_adv, y, targeted=False, kappa=0)
        print("Important!!!! =====#@Logits adv:", g_loss_cls)



    # Fool discriminator
    D_x_adv = D(x_adv)
    g_loss_gan = F.binary_cross_entropy_with_logits(
        D_x_adv, torch.ones_like(D_x_adv)
    )

    # Perturbation regularization (L2 norm)
    delta = (x_adv - x).view(x.size(0), -1)
    g_loss_pert = delta.norm(p=2, dim=1).mean()

    # Final generator loss
    # g_loss = g_loss_gan + lambda_adv * g_loss_cls + lambda_pert * g_loss_pert
    g_loss = lambda_adv * g_loss_cls

    if lambda_gan > 0:
        g_loss += lambda_gan * g_loss_gan
    if lambda_pert > 0:
        g_loss += lambda_pert * g_loss_pert


    # === Discriminator loss ===
    d_real = F.binary_cross_entropy_with_logits(
        D(x), torch.ones_like(D(x))
    )
    d_fake = F.binary_cross_entropy_with_logits(
        D_x_adv.detach(), torch.zeros_like(D_x_adv)
    )
    d_loss = 0.5 * (d_real + d_fake)

    return g_loss, d_loss


def cw_loss(logits, y, targeted=False, y_target=None, kappa=0):
    one_hot = F.one_hot(y, num_classes=logits.size(1)).float()
    correct_logit = torch.sum(one_hot * logits, dim=1)

    if targeted:
        # For targeted: max(other - target, -kappa)
        target_logit = logits.gather(1, y_target.unsqueeze(1)).squeeze(1)
        other_logit = torch.max((1 - F.one_hot(y_target, logits.size(1))).float() * logits, dim=1)[0]
        loss = torch.clamp(other_logit - target_logit, min=-kappa)
    else:
        # For untargeted: max(correct - best_other, -kappa)
        other_logit = torch.max((1 - one_hot) * logits, dim=1)[0]
        loss = torch.clamp(correct_logit - other_logit, min=-kappa)

    return torch.mean(loss)
import torch


class ContrastiveLoss(torch.nn.Module):
    """Contrastive loss function."""

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def get_distance(self, x0, x1):
        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)
        mdist = self.margin - dist
        mdist = torch.clamp(mdist, min=0.0)
        return mdist, dist_sq

    def forward(self, x0, x1, y):
        mdist, dist_sq = self.get_distance(x0, x1)
        loss = y * dist_sq + (1 - y) * torch.pow(mdist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss

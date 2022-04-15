import torch
import torchvision


class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).float().view(1, len(mean), 1, 1)
        self.std = torch.tensor(std).float().view(1, len(mean), 1, 1)
    
    @torch.no_grad()
    def forward(self, batch):
        self.mean = self.mean.to(batch["image"].device)
        self.std = self.std.to(batch["image"].device)
        batch["image"] = (batch["image"] - self.mean) / self.std
        return batch

class ColorJitter(torchvision.transforms.ColorJitter):
    #torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
    
    def __init__(self) -> None:
        super().__init__(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01)
    
    @torch.no_grad()    
    def forward(self, batch):
        batch["image"] = super().forward(batch["image"])
        return batch
    
class GaussianBlur(torchvision.transforms.GaussianBlur):
    #torchvision.transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))
    
    def __init__(self, kernel_size) -> None:
        super().__init__(kernel_size=kernel_size)

    @torch.no_grad()    
    def forward(self, batch):
        batch["image"] = super().forward(batch["image"])
        return batch
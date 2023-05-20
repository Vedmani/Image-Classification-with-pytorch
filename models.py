from torch import nn
import timm


class EfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.efficientnet = timm.create_model(model_name="efficientnet_b0", pretrained=True, num_classes=25)

        """# Set requires_grad to False for all parameters except the output layer
            for name, param in self.efficientnet.named_parameters():
                if not name.startswith('classifier'):
                    param.requires_grad = False"""
        # print number of parameters including final layer
        trainable_params = sum(p.numel() for p in self.efficientnet.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.efficientnet.parameters())
        """print("Efficientnet_b0 with 25 classes initialized")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Total parameters: {total_params}")"""

    def forward(self, x):
        return self.efficientnet(x)

import torch
import torch.nn as nn

class Classifier(nn.Module):
    """
    Multilayer Perceptron (MLP) classifier model.

    Args:
        input_size (int): Dimensionality of input features.
        hidden_size (int): Dimensionality of the hidden layer.
        n_classes (int): Number of output classes.
        n_layers (int): Number of layers in the MLP.
        dropout (float): Dropout probability.

    Returns:
        torch.Tensor: Output tensor after passing through MLP.
    """

    def __init__(self,
                 input_size: int = 512 * 2,
                 hidden_size: int = 512,
                 n_classes: int = 2,
                 n_layers: int = 1,
                 dropout: float = 0.2):

        super(Classifier, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),  # Gaussian Error Linear Units activation function
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return x


class VQAModel(nn.Module):
    def __init__(
        self,
        visual_encoder,
        text_encoder,
        classifier
    ):
        super(VQAModel, self).__init__()
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
        self.classifier = classifier
        

    def forward(self, image, answer):
        text_out = self.text_encoder(answer)
        image_out = self.visual_encoder(image)
        x = torch.cat((text_out, image_out), dim=1)
        x = self.classifier(x)

        return x

    def freeze(self, visual=True, textual=True, clas=False):
        if visual:
            for n,p in self.visual_encoder.named_parameters():
                p.requires_grad = False
        if textual:
            for n,p in self.text_encoder.named_parameters():
                p.requires_grad = False
        if clas:
            for n,p in self.classifier.named_parameters():
                p.requires_grad = False
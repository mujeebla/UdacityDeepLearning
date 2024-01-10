import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.4) -> None:
        super(MyModel, self).__init__()

        # Define the layers
        self.conv_block1 = self._create_conv_block(3, 64)  
        self.conv_block2 = self._create_conv_block(64, 128, dropout=0.1)
        self.conv_block3 = self._create_conv_block(128, 256, dropout=0.1)
        self.conv_block4 = self._create_conv_block(256, 512, dropout=0.1)
        self.conv_block5 = self._create_conv_block(512, 1024, dropout=0.1)  # 7x7
        self.conv_block6 = self._create_conv_block2(1024, 2048, dropout=0.1)  # 4x4
        self.conv_block7 = self._create_conv_block(2048, 4096, dropout=0.2)  #2x2


        # Flatten and Fully Connected layers
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(4096 * 2 * 2, 10000),  # Adjusted for the output of the last conv block
            nn.Dropout(dropout),
            nn.BatchNorm1d(10000),
            nn.ReLU(),
            nn.Linear(10000, 6000),
            nn.Dropout(dropout),
            nn.BatchNorm1d(6000),
            nn.ReLU(),
            nn.Linear(6000, num_classes)
        )

    def _create_conv_block(self, in_channels, out_channels, dropout=0.0, kernel_size=3):
        layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
        ]
        if dropout > 0:
            layers.insert(3, nn.Dropout2d(dropout))  # Adding spatial dropout before max pooling
        return nn.Sequential(*layers)
    
    # Second Conve block without max pooling
    def _create_conv_block2(self, in_channels, out_channels, dropout=0.0, kernel_size=2):
        layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
        ]
        if dropout > 0:
            layers.insert(3, nn.Dropout2d(dropout))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through all conv blocks
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)  # Forward pass through the new conv block
        x = self.conv_block7(x)  # Forward pass through the new conv block
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
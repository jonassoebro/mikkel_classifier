import pytest
import torch
import yaml
from models.model import ResNet18
from src.data.data import get_data

# Read from config
with open("config/config.yaml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    print("Read successful")

# Get data
batch_size = 4
train_dataloader, val_dataloader = get_data(batch_size)
images, labels = next(iter(train_dataloader))
model = ResNet18(pretrained=config['model']['pretrained'], in_dim=config['model']['in_dim'], out_dim=config['model']['out_dim'])
    
def test_model_output_shape():
    # Forward pass a batch
    preds = model(images)
    assert preds.shape == (batch_size, 1)

def test_error_on_wrong_shape():
   with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model(torch.randn(1,2,3))

from models.model import ResNet18

import yaml
from src.data.data import get_data

# Read from config
with open("config/config.yaml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    print("Read successful")

# Get data
batch_size = 4
train_dataloader, val_dataloader = get_data(batch_size)
images, labels = next(iter(train_dataloader))

# Forward pass a batch
model = ResNet18(pretrained=config['model']['pretrained'], in_dim=config['model']['in_dim'], out_dim=config['model']['out_dim'])
preds = model(images)
assert preds.shape == (batch_size, 1)

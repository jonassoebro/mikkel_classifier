from src.data.data import get_data#, download_data
train_dataloader, val_dataloader = get_data(1) # Batch size of 1 to assert len of dataloader

# TODO: Get these numbers by reading amount of files in data folder so it stays up to date
N_train = 953
N_valid = 238

# Assert data amount
assert len(train_dataloader) == N_train
assert len(val_dataloader) == N_valid

# Assert image shapes and label values
for batch in train_dataloader:
    images, labels = batch
    assert images.shape == (1,3,64,64)
    assert (labels[0]) == 0 or (labels[0]) == 1

for batch in val_dataloader:
    images, labels = batch
    assert images.shape == (1,3,64,64)
    assert (labels[0]) == 0 or (labels[0]) == 1


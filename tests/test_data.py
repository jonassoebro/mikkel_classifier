import os.path
import pytest
import googleapiclient
from src.data.data import get_data#, download_data
from tests import _PATH_DATA
train_dataloader, val_dataloader = get_data(1) # Batch size of 1 to assert len of dataloader

# TODO: Get these numbers by reading amount of files in data folder so it stays up to date
N_train = 953
N_valid = 238

# Assert data amount
@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data_samples():
    assert len(train_dataloader) == N_train, "Dataset did not have the correct number of samples"
    assert len(val_dataloader) == N_valid, "Dataset did not have the correct number of samples"

    # Assert image shapes and label values
    for batch in train_dataloader:
        images, labels = batch
        assert images.shape == (1,3,64,64), "Input image shape not as expected."
        assert (labels[0]) == 0 or (labels[0]) == 1, "Label takes an unexpected value other than 0 or 1"

    for batch in val_dataloader:
        images, labels = batch
        assert images.shape == (1,3,64,64), "Input image shape not as expected."
        assert (labels[0]) == 0 or (labels[0]) == 1, "Label takes an unexpected value other than 0 or 1"


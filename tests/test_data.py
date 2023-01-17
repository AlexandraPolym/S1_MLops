from tests import _PATH_DATA
from data import CorruptMnist
import numpy as np
import os.path




# # class TestData:
# dataset_train = CorruptMnist(train=True)
# dataset_test = CorruptMnist(train=False)
#     # def __init__(self):
#     #     self.dataset_train = CorruptMnist(train=True)
#     #     self.dataset_test = CorruptMnist(train=False)        
    
# def test_len_dataset(dataset):
#     if dataset == dataset_train:
#         assert len(dataset) == 25000
#     else:    
#         assert len(dataset) == 5000


# def test_data_shape(dataset):
#     #     #assert np.all(dataset.data.size()[1:] == torch.Size([1, 28, 28])), f"Expected data shape (1, 28, 28) but got {dataset.data.size()[1:]}"
#     assert dataset.data.shape[1:] == (1, 28, 28), f"Expected data shape (1, 28, 28) but got {dataset.data.shape[1:]}"


# def test_all_labels_represented(dataset):
#     unique_labels = np.unique(dataset.targets.numpy())
#     assert unique_labels.size == 10, f"Expected 10 unique labels but got {unique_labels.size}"   


# # if __name__ == "__main__":

# #   #  @pytest.mark.skipif(not os.path.exists(file_path), reason="Data files not found")
# test_len_dataset(dataset_train)
# test_len_dataset(dataset_test)
# test_all_labels_represented(dataset_train)
# test_all_labels_represented(dataset_test)
# test_data_shape(dataset_train)
# test_data_shape(dataset_test)
#######################################################################################

# @pytest.mark.skipif(not os.path.exists(''), reason="Data files not found")

def test_data():
    dataset_train = CorruptMnist(train=True)
    dataset_test = CorruptMnist(train=False)
    
    assert len(dataset_train) == 25000
    assert len(dataset_test) == 5000


    assert dataset_train.data.shape[1:] == (1, 28, 28), f"Expected data shape (1, 28, 28) but got {dataset_train.data.shape[1:]}"
    assert dataset_test.data.shape[1:] == (1, 28, 28), f"Expected data shape (1, 28, 28) but got {dataset_test.data.shape[1:]}"

    unique_labels_train = np.unique(dataset_train.targets.numpy())
    assert unique_labels_train.size == 10, f"Expected 10 unique labels but got {unique_labels_train.size}"   

    unique_labels_test = np.unique(dataset_test.targets.numpy())
    assert unique_labels_test.size == 10, f"Expected 10 unique labels but got {unique_labels_test.size}"  


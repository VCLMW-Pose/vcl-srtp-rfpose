from torch.utils.data import Dataset, DataLoader


def build_dataset():
    pass


class GenericDataset(Dataset):
    """General dataset class
        General dataset specifies the general processing pipeline of dataset.
        To deploy a dataset, you must implement _setup__ and _getitem__
        which
    """

    def __init__(self):
        pass

    def _setup__(self):
        raise NotImplementedError

    def __getitem__(self):
        pass

    def _getitem__(self):
        raise NotImplementedError

    def __len__(self):
        pass
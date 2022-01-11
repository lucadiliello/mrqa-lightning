from torch.utils.data import Dataset


class MapDataset(Dataset):
    r""" MapDataset provides List functionalities into a Dataset object. """

    def __init__(self, data):
        super().__init__()
        self.data = list(iter(data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> dict:
        r""" Get dict of data at a given position. """
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx

        assert 0 <= idx < len(self), (f"Received index out of range {idx}, range: {0} <= idx < {len(self)}")
        return self.data[idx]

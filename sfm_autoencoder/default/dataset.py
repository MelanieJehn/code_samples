from torch.utils.data import Dataset


class DescriptorDataset(Dataset):
  """
  A Dataset of descriptors
  """

  def __init__(self, descriptors, transform=None):
    """
    Args:
        descriptors (list): List containing surf descriptors.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """
    self.transform = transform
    self.data = descriptors

  def __len__(self):
    # The length of the dataset is simply the length of self.df
    return len(self.data)

  def __getitem__(self, idx):
    sample = self.data[idx]
    if self.transform:
        sample = self.transform(sample)
    return sample

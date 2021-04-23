import json
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class VWUnansDataset(Dataset):
    """VizWiz Unanswerability Dataset"""

    def __init__(self, json_file, image_dir, test_data=False):
        """
        :param json_file: annotation file
        :param image_dir: annotated image directory
        :param test_data: indicate data is test data
        """
        self.image_dir = image_dir
        with open(json_file, 'r') as f:
            self.data = json.load(f)
            if not test_data:
                for idx in range(len(self.data)):
                    self.data[idx].pop("answers")
                    self.data[idx].pop("answer_type")

        # image processing
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not "image_vec" in self.data[idx]:
            path = os.path.join(self.image_dir, self.data[idx]["image"])
            img = Image.open(path)
            self.data[idx]["image_vec"] = self.to_tensor(img)

        # TODO convert text to indices

        return self.data[idx]

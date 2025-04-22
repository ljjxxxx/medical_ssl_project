import torch
import logging
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChestXrayDataset(Dataset):
    """胸部X光监督学习数据集"""

    def __init__(
            self,
            data_dir,
            csv_file,
            transform=None,
            split='train'
    ):
        """
        初始化胸部X光数据集

        Args:
            data_dir: 数据目录
            csv_file: CSV文件路径
            transform: 图像变换
            split: 数据集划分 ('train', 'val', 'test')
        """
        self.data_dir = Path(data_dir)
        self.df = pd.read_csv(csv_file)

        disease_labels_path = self.data_dir / "disease_labels.txt"
        if disease_labels_path.exists():
            with open(disease_labels_path, "r") as f:
                self.disease_labels = f.read().strip().split("\n")
        else:
            self.disease_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
                                   'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
                                   'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']

        if split:
            self.df = self.df[self.df['split'] == split].reset_index(drop=True)

        self.transform = transform
        logger.info(f"加载了 {split} 数据集，共 {len(self.df)} 个样本")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['Image Index']
        img_path = self.data_dir / "images" / img_name

        try:
            with Image.open(img_path) as img:
                image = img.convert('RGB')

                label = []
                for disease in self.disease_labels:
                    label.append(float(self.df.iloc[idx][disease]))
                label = torch.tensor(label, dtype=torch.float32)

                if self.transform:
                    image = self.transform(image)

                return {'image': image, 'label': label, 'filename': img_name}

        except Exception as e:
            logger.error(f"加载图像 {img_path} 时出错: {e}")
            image = torch.zeros((3, 224, 224), dtype=torch.float32)
            label = torch.zeros(len(self.disease_labels), dtype=torch.float32)
            return {'image': image, 'label': label, 'filename': img_name}


class SSLChestXrayDataset(Dataset):
    """胸部X光自监督学习数据集"""

    def __init__(
            self,
            data_dir,
            csv_file,
            transform=None,
            split='train'
    ):
        """
        初始化SSL数据集

        Args:
            data_dir: 数据目录
            csv_file: CSV文件路径
            transform: 图像变换
            split: 数据集划分 ('train', 'val')
        """
        self.data_dir = Path(data_dir)
        self.df = pd.read_csv(csv_file)

        if split:
            self.df = self.df[self.df['split'] == split].reset_index(drop=True)

        self.transform = transform
        logger.info(f"加载了SSL {split} 数据集，共 {len(self.df)} 个样本")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['Image Index']
        img_path = self.data_dir / "images" / img_name

        try:
            with Image.open(img_path) as img:
                image = img.convert('RGB')

                if self.transform:
                    view1 = self.transform(image)
                    view2 = self.transform(image)
                else:
                    default_transform = transforms.ToTensor()
                    view1 = default_transform(image)
                    view2 = default_transform(image)

                return {'views': [view1, view2], 'filename': img_name}

        except Exception as e:
            logger.error(f"加载图像 {img_path} 时出错: {e}")
            view = torch.zeros((3, 224, 224), dtype=torch.float32)
            return {'views': [view, view], 'filename': img_name}
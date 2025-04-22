import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import logging
import urllib.request
import tarfile
from pathlib import Path
from tqdm import tqdm

from .dataset import ChestXrayDataset, SSLChestXrayDataset
from .transforms import get_ssl_transforms, get_eval_transforms

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UrlProgress(tqdm):
    """用于在下载过程中显示进度条的URL请求钩子"""

    def update_to(self, blocks=1, block_size=1, total_size=None):
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def download_chestxray14_dataset(output_dir="./datasets/chestxray14", use_subset=False, subset_size=5000):
    """
    下载并处理ChestX-ray14数据集

    Args:
        output_dir: 输出目录
        use_subset: 是否使用子集
        subset_size: 子集大小

    Returns:
        output_path: 数据集路径
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if (output_path / "processed_dataset.csv").exists():
        logger.info(f"数据集已存在于 {output_path}")
        return output_path

    image_urls = [
        "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz",
        "https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz",
        "https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz",
        "https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz",
        "https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz",
        "https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz",
        "https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz",
        "https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz",
        "https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz",
        "https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz",
        "https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz",
        "https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz"
    ]

    data_entry_url = "https://data.broadinstitute.org/ml4h/datasets/nih-chest-xray-decathlon/Data_Entry_2017_v2020.csv"

    data_entry_path = output_path / "Data_Entry_2017.csv"

    # # 下载数据标签
    # logger.info(f"正在从 {data_entry_url} 下载数据标签CSV")
    # with UrlProgress(unit='B', unit_scale=True, miniters=1, desc="下载数据标签") as progress:
    #     urllib.request.urlretrieve(data_entry_url, data_entry_path, reporthook=progress.update_to)

    images_dir = output_path / "images"
    if not images_dir.exists():
        images_dir.mkdir(exist_ok=True)

        parts_to_download = 1 if use_subset else len(image_urls)

        for i, url in enumerate(image_urls[:parts_to_download], 1):
            archive_path = output_path / f"images_part{i}.tar.gz"

            logger.info(f"正在下载第 {i}/{parts_to_download} 部分图像 ({url})")
            with UrlProgress(unit='B', unit_scale=True, miniters=1, desc=f"下载图像部分 {i}") as progress:
                urllib.request.urlretrieve(url, archive_path, reporthook=progress.update_to)

            logger.info(f"正在解压第 {i} 部分图像到 {images_dir}")
            with tarfile.open(archive_path, "r:gz") as tar:
                if use_subset and i == 1:
                    members = tar.getmembers()[:subset_size]
                    for member in tqdm(members, desc=f"解压部分 {i}"):
                        tar.extract(member, path=output_path)
                else:
                    for member in tqdm(tar.getmembers(), desc=f"解压部分 {i}"):
                        tar.extract(member, path=output_path)

            archive_path.unlink()

        extracted_dir = output_path / "images"
        if not extracted_dir.exists():
            potential_dirs = [d for d in output_path.iterdir() if d.is_dir() and d.name != 'images']
            if potential_dirs:
                extracted_dir = potential_dirs[0]
                extracted_dir.rename(images_dir)

    logger.info("处理数据集标签")
    df = pd.read_csv(data_entry_path)

    disease_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
                      'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
                      'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']

    for disease in disease_labels:
        df[disease] = df['Finding Labels'].apply(lambda x: 1 if disease in x else 0)

    df['No_Finding'] = df['Finding Labels'].apply(lambda x: 1 if 'No Finding' in x else 0)

    df['Labels'] = df.apply(lambda row: [row[disease] for disease in disease_labels], axis=1)

    available_images = [f.name for f in images_dir.glob('*.png')]
    df = df[df['Image Index'].isin(available_images)].reset_index(drop=True)

    logger.info(f"找到 {len(df)} 张有效的带标签图像")

    np.random.seed(42)
    train_idx = np.random.choice(
        df.index, size=int(0.7 * len(df)), replace=False)
    remaining = df.index.difference(train_idx)
    val_idx = np.random.choice(
        remaining, size=int(0.5 * len(remaining)), replace=False)
    test_idx = remaining.difference(val_idx)

    df.loc[train_idx, 'split'] = 'train'
    df.loc[val_idx, 'split'] = 'val'
    df.loc[test_idx, 'split'] = 'test'

    processed_path = output_path / "processed_dataset.csv"
    df.to_csv(processed_path, index=False)

    with open(output_path / "disease_labels.txt", "w") as f:
        f.write("\n".join(disease_labels))

    logger.info(f"数据集准备完成，保存在 {output_path}")
    logger.info(f"总图像数: {len(df)}")

    return output_path


def create_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    创建数据加载器

    Args:
        data_dir: 数据目录
        batch_size: 批处理大小
        num_workers: 工作进程数

    Returns:
        dataloaders: 数据加载器字典
    """
    csv_file = os.path.join(data_dir, "processed_dataset.csv")

    disease_labels_path = Path(data_dir) / "disease_labels.txt"
    if disease_labels_path.exists():
        with open(disease_labels_path, "r") as f:
            disease_labels = f.read().strip().split("\n")
    else:
        disease_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
                          'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
                          'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']

    ssl_transform = get_ssl_transforms()
    eval_transform = get_eval_transforms()

    ssl_train_dataset = SSLChestXrayDataset(
        data_dir=data_dir,
        csv_file=csv_file,
        transform=ssl_transform,
        split='train'
    )

    ssl_val_dataset = SSLChestXrayDataset(
        data_dir=data_dir,
        csv_file=csv_file,
        transform=ssl_transform,
        split='val'
    )

    supervised_train_dataset = ChestXrayDataset(
        data_dir=data_dir,
        csv_file=csv_file,
        transform=ssl_transform,
        split='train'
    )

    supervised_val_dataset = ChestXrayDataset(
        data_dir=data_dir,
        csv_file=csv_file,
        transform=eval_transform,
        split='val'
    )

    supervised_test_dataset = ChestXrayDataset(
        data_dir=data_dir,
        csv_file=csv_file,
        transform=eval_transform,
        split='test'
    )

    persistent_workers = True if num_workers > 0 else False

    ssl_train_loader = DataLoader(
        ssl_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.backends.mps.is_available() else False,
        persistent_workers=persistent_workers,
        drop_last=True
    )

    ssl_val_loader = DataLoader(
        ssl_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.backends.mps.is_available() else False,
        persistent_workers=persistent_workers
    )

    supervised_train_loader = DataLoader(
        supervised_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.backends.mps.is_available() else False,
        persistent_workers=persistent_workers
    )

    supervised_val_loader = DataLoader(
        supervised_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.backends.mps.is_available() else False,
        persistent_workers=persistent_workers
    )

    supervised_test_loader = DataLoader(
        supervised_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.backends.mps.is_available() else False,
        persistent_workers=persistent_workers
    )

    return {
        'ssl': {
            'train': ssl_train_loader,
            'val': ssl_val_loader
        },
        'supervised': {
            'train': supervised_train_loader,
            'val': supervised_val_loader,
            'test': supervised_test_loader
        },
        'disease_labels': disease_labels
    }
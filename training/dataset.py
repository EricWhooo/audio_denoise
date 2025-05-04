"""
改进点
1. 保持对 VoiceBank-DEMAND 的原生支持，同时允许自动按比例拆分
   train / val 集合，或者显式指定 val 目录。
2. 增加 create_dataloaders() 工具，可一行得到 train/val/test
   三个 DataLoader。
3. 默认随机种子固定，保证可复现。
"""

import os
from glob import glob
from typing import Optional, Tuple, Sequence

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from audio.tools import wav_to_fbank


class VoiceBankDEMANDDataset(Dataset):
    """
    VoiceBank + DEMAND 数据集
    目录结构:
        root/
          clean_trainset_wav/
          noisy_trainset_wav/
          clean_testset_wav/
          noisy_testset_wav/
    """

    def __init__(
        self,
        root_dir: str,
        subset: str,
        mel_params: dict,
        pairs_list: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        """
        参数:
            root_dir:     数据根目录
            subset:       'train' | 'test'（val 由外部 random_split 产生）
            mel_params:   传给 wav_to_fbank 的参数
            pairs_list:   若不为 None，则直接使用给定 (noisy, clean) 路径对
        """
        super().__init__()
        assert subset in ("train", "test"), f"subset 不合法: {subset}"
        self.mel_params = mel_params

        if pairs_list is not None:  # 来自 random_split
            self.pairs = pairs_list
            return

        clean_dir = os.path.join(root_dir, f"clean_{subset}set_wav")
        noisy_dir = os.path.join(root_dir, f"noisy_{subset}set_wav")
        pattern = os.path.join(noisy_dir, "*", "*", "*.wav")
        noisy_files = sorted(glob(pattern))
        if len(noisy_files) == 0:
            raise RuntimeError(f"没有找到 noisy wav，检查路径: {pattern}")

        self.pairs = []
        for n_path in noisy_files:
            fname = os.path.basename(n_path)
            c_path = os.path.join(clean_dir, fname)
            if not os.path.exists(c_path):
                raise FileNotFoundError(f"缺少 clean 对应文件: {c_path}")
            self.pairs.append((n_path, c_path))

    def __len__(self):
        return len(self.pairs)

    def _load_mel(self, path: str):
        fbank, _, _ = wav_to_fbank(path, **self.mel_params)  # (T, n_mel)
        return fbank.unsqueeze(0)  # → (1, T, n_mel)

    def __getitem__(self, idx: int):
        noisy_path, clean_path = self.pairs[idx]
        noisy_mel = self._load_mel(noisy_path)
        clean_mel = self._load_mel(clean_path)
        return noisy_mel, clean_mel


def _make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )


def create_dataloaders(
    root_dir: str,
    mel_params: dict,
    batch_size: int,
    val_ratio: float = 0.1,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 2023,
):
    """
    返回 train_loader, val_loader, test_loader
    如果 val_ratio == 0 则不拆分，val_loader 返回 None。
    """
    # --- train dataset & split -------------
    full_train_set = VoiceBankDEMANDDataset(root_dir, "train", mel_params)
    if val_ratio > 0.0:
        n_total = len(full_train_set)
        n_val = int(n_total * val_ratio)
        n_train = n_total - n_val
        generator = torch.Generator().manual_seed(seed)
        train_set, val_set = random_split(
            full_train_set, [n_train, n_val], generator=generator
        )
        # random_split 返回子集索引，需要再次封装为数据集实例以便 __getitem__
        train_set = VoiceBankDEMANDDataset(
            root_dir,
            "train",
            mel_params,
            pairs_list=[full_train_set.pairs[i] for i in train_set.indices],
        )
        val_set = VoiceBankDEMANDDataset(
            root_dir,
            "train",
            mel_params,
            pairs_list=[full_train_set.pairs[i] for i in val_set.indices],
        )
    else:
        train_set, val_set = full_train_set, None

    # --- test dataset ----------------------
    test_set = VoiceBankDEMANDDataset(root_dir, "test", mel_params)

    train_loader = _make_loader(train_set, batch_size, True, num_workers, pin_memory)
    val_loader = (
        _make_loader(val_set, batch_size, False, num_workers, pin_memory)
        if val_set is not None
        else None
    )
    test_loader = _make_loader(test_set, batch_size, False, num_workers, pin_memory)
    return train_loader, val_loader, test_loader
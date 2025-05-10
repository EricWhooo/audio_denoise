import os
from glob import glob
from typing import Optional, Tuple, Sequence

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from audio.tools import wav_to_fbank

class VoiceBankDEMANDDataset(Dataset):
    """
    VoiceBank-DEMAND dataset.
    Structure:
        root/
          clean_trainset_56spk_wav/
          noisy_trainset_56spk_wav/
          clean_trainset_28spk_wav/
          noisy_trainset_28spk_wav/
          clean_testset_wav/
          noisy_testset_wav/
    """

    def __init__(
        self,
        root_dir: str,
        subset: str,
        mel_params: dict,
        subset_type: str = "56spk",
        pairs_list: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        """
        Args:
            root_dir: dataset root
            subset: 'train' | 'val' | 'test'
            mel_params: params for wav_to_fbank
            subset_type: '56spk' | '28spk' (for train/val)
            pairs_list: use given (noisy, clean) pairs if not None
        """
        super().__init__()
        assert subset in ("train", "val", "test")
        self.mel_params = mel_params

        if pairs_list is not None:
            self.pairs = pairs_list
            return

        if subset == "test":
            clean_dir = os.path.join(root_dir, "clean_testset_wav")
            noisy_dir = os.path.join(root_dir, "noisy_testset_wav")
        elif subset in ("train", "val"):
            clean_dir = os.path.join(root_dir, f"clean_trainset_{subset_type}_wav")
            noisy_dir = os.path.join(root_dir, f"noisy_trainset_{subset_type}_wav")
        else:
            raise ValueError(f"Unknown subset: {subset}")

        noisy_files = sorted(glob(os.path.join(noisy_dir, "*.wav")))
        if len(noisy_files) == 0:
            raise RuntimeError(f"No noisy wavs found: {noisy_dir}")

        self.pairs = []
        for n_path in noisy_files:
            fname = os.path.basename(n_path)
            c_path = os.path.join(clean_dir, fname)
            if not os.path.exists(c_path):
                raise FileNotFoundError(f"Missing clean file: {c_path}")
            self.pairs.append((n_path, c_path))

    def __len__(self):
        return len(self.pairs)

    def _load_mel(self, path: str):
        fbank, _, _ = wav_to_fbank(path, **self.mel_params)
        return fbank.unsqueeze(0)  # (1, T, n_mel)

    def __getitem__(self, idx: int):
        noisy_path, clean_path = self.pairs[idx]
        noisy_mel = self._load_mel(noisy_path)
        clean_mel = self._load_mel(clean_path)
        return noisy_mel, clean_mel

def _make_loader(dataset, batch_size, shuffle, num_workers, pin_memory):
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
    subset_type: str = "56spk",
):
    """
    Returns train_loader, val_loader, test_loader.
    If val_ratio == 0, val_loader is None.
    """
    full_train_set = VoiceBankDEMANDDataset(root_dir, "train", mel_params, subset_type)
    if val_ratio > 0.0:
        n_total = len(full_train_set)
        n_val = int(n_total * val_ratio)
        n_train = n_total - n_val
        generator = torch.Generator().manual_seed(seed)
        train_indices, val_indices = random_split(
            range(n_total), [n_train, n_val], generator=generator
        )
        train_set = VoiceBankDEMANDDataset(
            root_dir,
            "train",
            mel_params,
            subset_type,
            pairs_list=[full_train_set.pairs[i] for i in train_indices.indices],
        )
        val_set = VoiceBankDEMANDDataset(
            root_dir,
            "train",
            mel_params,
            subset_type,
            pairs_list=[full_train_set.pairs[i] for i in val_indices.indices],
        )
    else:
        train_set, val_set = full_train_set, None

    test_set = VoiceBankDEMANDDataset(root_dir, "test", mel_params)

    train_loader = _make_loader(train_set, batch_size, True, num_workers, pin_memory)
    val_loader = (
        _make_loader(val_set, batch_size, False, num_workers, pin_memory)
        if val_set is not None
        else None
    )
    test_loader = _make_loader(test_set, 1, False, num_workers, pin_memory)
    return train_loader, val_loader, test_loader
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class AudioDataset(Dataset):
    def __init__(self, audio_paths, texts, vocab):
        self.audio_paths = audio_paths
        self.texts = texts
        self.vocab = vocab

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, index):
        waveform, sample_rate = torchaudio.load(self.audio_paths[index])

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        fbank = kaldi.fbank(waveform, num_mel_bins=80, sample_frequency=sample_rate)

        fbank = (fbank - fbank.mean()) / (fbank.std() + 1e-6)
        target = torch.tensor([self.vocab.get(char, 1) for char in self.texts[index]], dtype=torch.long)

        return fbank, target


def collate_fn(batch):
    fbanks = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    input_lengths = torch.tensor([f.shape[0] // 4 for f in fbanks], dtype=torch.long)
    target_lengths = torch.tensor([t.shape[0] for t in targets], dtype=torch.long)
    fbanks_padded = pad_sequence(fbanks, batch_first=True, padding_value=0.0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    return fbanks_padded, targets_padded, input_lengths, target_lengths
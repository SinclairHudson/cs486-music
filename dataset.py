import os
from torch.utils.data import Dataset
import torchaudio
torchaudio.set_audio_backend("sox_io")
from torchaudio.transforms import InverseSpectrogram, Spectrogram

class LofiDataset(Dataset):
    """
    Dataset of audio clips, in a folder.
    """
    def __init__(self, root_path: str,
                 spectrogram: torchaudio.transforms.Spectrogram,
                 length=45):
        super(LofiDataset).__init__()
        self.root_path = root_path
        self.min_length = length  # length in seconds
        self.audio_files = os.listdir(root_path)
        self.transform = spectrogram

        # filter audio_files by 
        self.good_audio_files = []
        for i, audio_file in enumerate(self.audio_files):
            info = torchaudio.info(os.path.join(self.root_path, audio_file))
            length = info.num_frames / info.sample_rate  # length in seconds
            if length >= self.min_length:
                self.good_audio_files.append(os.path.join(self.root_path, audio_file))

        print(f"Initialized LofiDataset with {self.__len__()} songs " \
              f"of length {self.min_length}.")
        print(f"Total audio time is {self.__len__() * self.min_length} seconds.")

    def __len__(self):
        return len(self.good_audio_files)

    def  __getitem__(self, index):
        raw_audio, sample_rate = torchaudio.load(self.good_audio_files[index])
        stop = sample_rate * self.min_length
        raw_audio = raw_audio[0][:stop]
        return self.transform(raw_audio)

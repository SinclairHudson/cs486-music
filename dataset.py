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
        self.clips = []
        for i, audio_file in enumerate(self.audio_files):
            info = torchaudio.info(os.path.join(self.root_path, audio_file))
            length = info.num_frames / info.sample_rate  # length in seconds
            num_clips = int(length // self.min_length)
            for x in range(num_clips):
                self.clips.append((os.path.join(self.root_path, audio_file), x))  # store file and clip number

        print(f"Initialized LofiDataset with {self.__len__()} songs " \
              f"of length {self.min_length}.")
        print(f"Total audio time is {self.__len__() * self.min_length} seconds.")
        print(f"input size for a single instance is {self.__getitem__(0).size()}.")

    def __len__(self):
        return len(self.clips)

    def  __getitem__(self, index):
        file_name, clip_index = self.clips[index]
        raw_audio, sample_rate = torchaudio.load(file_name)
        clip_length_in_samples = sample_rate * self.min_length
        # left channel, and take the segment
        raw_audio = raw_audio[0][clip_length_in_samples * clip_index:
                                 clip_length_in_samples * (clip_index+1)]
        return self.transform(raw_audio)

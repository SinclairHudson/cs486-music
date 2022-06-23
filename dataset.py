import os
from torch.utils.data import Dataset
import torchaudio
torchaudio.set_audio_backend("sox_io")

class LofiDataset(Dataset):
    """
    Dataset of audio clips, in a folder.
    """
    def __init__(self, root_path: str, length=45):
        super(LofiDataset).__init__()
        self.root_path = root_path
        self.min_length = 45  # length in seconds
        self.audio_files = os.listdir(root_path)

        # filter audio_files by 
        good = []
        for i, audio_file in enumerate(self.audio_files):
            info = torchaudio.info(os.path.join(self.root_path, audio_file))
            raw_audio = torchaudio.load(os.path.join(self.root_path, audio_file))
            breakpoint()
            if length >= self.min_length:
                # keep

                pass

        # filter by the audio files that are of suitable length
        self.audio_files = self.audio_files[good]

    def __len__(self):
        return len(self.audio_files)

    def  __getitem__(self, index):
        raw_audio = torchaudio.load(self.audio_files[index])
        crop = None # TODO crop the audio to min_length in length

        return crop

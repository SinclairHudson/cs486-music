import torch
import torchaudio
from resnet_compressor import ResCompressor
from torchaudio.transforms import InverseSpectrogram, Spectrogram
from utils import ml_representation_to_audio, spectrogram_to_ml_representation

device = torch.device("cuda:0")

file_name = "/media/sinclair/datasets4/lofi/mrbrightside.wav"
output_name = "io/noway.wav"

raw_audio, sample_rate = torchaudio.load(file_name)
# left channel
raw_audio = raw_audio[0]
s = Spectrogram(n_fft=800, return_complex=True, power=None)
inv = InverseSpectrogram(n_fft=800)
x_imaginary = s(raw_audio)
x = spectrogram_to_ml_representation(x_imaginary.unsqueeze(0).to(device))

compressor = ResCompressor(step_size=16, vocab_size=256, beta=1).to(device)
start_point = "io/best_epoch_run_spring-sunset-42.pth"
compressor.load_state_dict(torch.load(start_point))

with torch.no_grad():
    xhat, _, _ = compressor(x)
    sample = ml_representation_to_audio(xhat)
    torchaudio.save(output_name, sample, sample_rate=44100)

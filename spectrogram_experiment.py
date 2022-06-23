import torch
import torchaudio
from torchaudio.transforms import InverseSpectrogram, Spectrogram

waveform, sample_rate = torchaudio.load('/media/sinclair/datasets4/lofi/2022-02-03-20:22:57.wav', normalize=True)

print(waveform.shape)
transform = Spectrogram(n_fft=800, return_complex=True, power=None)
inv = InverseSpectrogram(n_fft=800)

spectrogram = transform(waveform)
print(spectrogram.shape)
print(spectrogram.type())

waveform_back = inv(spectrogram)

torchaudio.save("secondlofi.wav", waveform_back, sample_rate=44100)





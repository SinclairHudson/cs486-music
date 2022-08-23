# Aleatory LoFi

This repo seeks to generate LoFi dataset, using a spectrogram representation.

### Recording audio
audio-recorder apt package
https://askubuntu.com/questions/229352/how-to-record-output-to-speakers

Audio is recorded from spotify, using a playlist of LoFi music.
**Volume on the linux machine must be set to 100%, volume on Spotify must be 100%.**
Recording on a different volume isn't being explored for now, 
technically it's just more data augmentation but not needed for now.


## TODO
1. positional encodings so the decoder knows what frequency is what (position-wise)
2. Re-generate dataset; test set could be hard because of a different volume used to record
3. Data augmentation
4. Clean up file structure


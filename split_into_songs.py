from pydub import AudioSegment, silence
from tqdm import tqdm
import math

OFFSET = 0  # the start of the conter. Don't overwrite your earlier generated songs!
MIN_SONG_LENGTH = 45
DB_SILENCE_THRESH = -21
MIN_SILENCE_LEN = 900
myaudio = AudioSegment.from_wav("/media/sinclair/datasets4/lofi/test_dataset.wav")
dBFS = myaudio.dBFS
len_in_ms = len(myaudio)
print(f"loaded a file that was {len_in_ms}ms long, {len_in_ms/(60 * 1000)} mins long, {len_in_ms/(60 * 60 * 1000)} hours long.")
print(f"detecting silence, this takes a while. 10x faster than playing the file, approximately")
print(f"expect this to run for {len_in_ms/(60 * 1000 * 10)} minutes.")
silence = silence.detect_silence(myaudio, min_silence_len=MIN_SILENCE_LEN, silence_thresh=dBFS+DB_SILENCE_THRESH)

silence = [((start/1000),(stop/1000)) for start, stop in silence] # in sec
print(f"found {len(silence)} silence segments.")
silence.insert(0, (0, 0)) # make the start a valid beginning
silence.append((len_in_ms-1, len_in_ms-1)) # make the end a valid ending cutoff

print(f"cutting audio into non-silent segments greater than {MIN_SONG_LENGTH} seconds.")
song_counter = 0
for sil, (start, end) in tqdm(enumerate(silence)):
    if sil == 0:
        continue  # skip the first silence, not interesting
    # if the end of the last silence and the start of this silence are far enough apart,
    last_end = silence[sil - 1][1]
    if abs(last_end - start) > MIN_SONG_LENGTH:  # if the distance to the next silence is greater than 60 seconds
        song = myaudio[last_end * 1000: (start)*1000]
        song.export(f"/media/sinclair/datasets4/lofi/test_splits/song_{song_counter + OFFSET}.wav", format="wav")
        song_counter+=1
print(f"exported {song_counter} songs. set OFFSET to {song_counter + OFFSET}")


from pydub import AudioSegment, silence
from tqdm import tqdm
import os

MIN_SONG_LENGTH = 45
DB_SILENCE_THRESH = -21
MIN_SILENCE_LEN = 900

ROOT_DIR = "/media/sinclair/datasets/lofi"

trainset_files = ["99-100-00.wav", "99-100-01.wav", "laptop-100-100-001.wav"]
testset_files = ["99-100-02.wav", "laptop-100-100-000.wav"]

def export_songs(origin_files, split_dir: str) -> None:
    for file in origin_files:
        print(f"splitting up the file {file}")
        myaudio = AudioSegment.from_wav(os.path.join(ROOT_DIR, file))

        dBFS = myaudio.dBFS
        len_in_ms = len(myaudio)
        print(f"loaded a file that was {len_in_ms}ms long, {len_in_ms/(60 * 1000)} mins long, {len_in_ms/(60 * 60 * 1000)} hours long.")
        print(f"detecting silence, this takes a while. 10x faster than playing the file, approximately")
        print(f"expect this song to be split in {len_in_ms/(60 * 1000 * 10)} minutes.")
        sil = silence.detect_silence(myaudio, min_silence_len=MIN_SILENCE_LEN, silence_thresh=dBFS+DB_SILENCE_THRESH)

        sil = [((start/1000),(stop/1000)) for start, stop in sil] # in sec
        print(f"found {len(sil)} silence segments.")
        sil.insert(0, (0, 0)) # make the start a valid beginning
        sil.append((len_in_ms-1, len_in_ms-1)) # make the end a valid ending cutoff

        print(f"cutting audio into non-silent segments greater than {MIN_SONG_LENGTH} seconds.")
        song_counter = 0
        for s, (start, end) in tqdm(enumerate(sil)):
            if s == 0:
                continue  # skip the first silence, not interesting
            # if the end of the last silence and the start of this silence are far enough apart,
            last_end = sil[s - 1][1]
            if abs(last_end - start) > MIN_SONG_LENGTH:  # if the distance to the next silence is greater than 60 seconds
                song = myaudio[last_end * 1000: (start)*1000]
                song.export(os.path.join(ROOT_DIR, split_dir, f"song_{song_counter}.wav"), format="wav")
                song_counter+=1
        print(f"exported {song_counter} songs.")

export_songs(trainset_files, "train_splits")
export_songs(testset_files, "test_splits")


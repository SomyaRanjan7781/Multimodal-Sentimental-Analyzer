import librosa
import numpy as np
MAX_LEN = 130   # fixed length (approx)
def audio_to_numbers(path):
    y, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    # padding / trimming
    if mfcc.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)))
    else:
        mfcc = mfcc[:, :MAX_LEN]
    return mfcc
def audio_label_from_filename(path):
    filename = path.split("\\")[-1]   # windows path
    emotion_code = filename.split("-")[2]

    if emotion_code == "03":      # happy
        return 2
    elif emotion_code == "01":    # neutral
        return 1
    elif emotion_code in ["04", "05"]:  # sad, angry
        return 0
    else:
        return None

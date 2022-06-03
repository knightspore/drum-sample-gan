import librosa
from soundfile import SoundFile


def SaveWavFile(spec, filename):
    wav_data = librosa.feature.inverse.mel_to_audio(
        spec[0], sr=44100, n_fft=1024, hop_length=None
    )
    with SoundFile("./output/" + filename + ".wav", "w", 44100, 1, "PCM_24") as f:
        f.write(wav_data)

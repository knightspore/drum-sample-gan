from utils.process_audio import ProcessAudio


class CreateDS:
    def __init__(self, df):
        self.df = df
        self.duration = 4000
        self.sr = 44100
        self.channel = 2

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_file = self.df.iloc[idx]["path"]
        class_id = self.df.iloc[idx]["class"]

        aud = ProcessAudio.open(audio_file)
        reaud = ProcessAudio.resample(aud, self.sr)
        rechan = ProcessAudio.rechannel(reaud, self.channel)
        dur_aud = ProcessAudio.pad_trunc(rechan, self.duration)
        sgram = ProcessAudio.spectro_gram(dur_aud, n_mels=64, n_fft=1024, hop_len=None)
        aug_sgram = ProcessAudio.spectro_augment(
            sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2
        )

        return aug_sgram, class_id, idx

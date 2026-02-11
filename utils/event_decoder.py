import random

import numpy
import numpy as np
import soundfile as sf
import librosa



class EventDecoding:
    """
    A class used to configure the decoding of audio events.

    Attributes
    ----------
    sample_rate : int
        Defines the sample rate to which the audio should be resampled. This standardizes the input data's sample rate, making it consistent for model processing.
    extracted_interval : float
        Denotes the fixed duration (in seconds) of the audio segment that is randomly extracted from the extended audio event.
    """

    def __init__(
        self,
        sample_rate: int = 32_000,
        extracted_interval: float = 7.0,
        hf_cache_root: str = "/data",
    ):
        self.extracted_interval = extracted_interval
        self.sample_rate = sample_rate
        self.hf_cache_root = hf_cache_root

    def _load_audio(self, path, duration, start=None, end=None, sr=None):
        pad_left  =  0 #s
        pad_right  = 0 #s
        if start is not None and end is not None:
            if start < 0:  # first snippet case
                pad_left = - start
                start = 0
            if end > duration:  # last snippet case
                pad_right = end - duration
                end = duration

            start, end = int(start * sr), int(end * sr)
        audio, sr = sf.read(path, start=start, stop=end)


        if audio.ndim != 1:
            audio = audio.swapaxes(1, 0)
            audio = librosa.to_mono(audio)
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate

        # print("pad_left", pad_left, "pad_right", pad_right)
        if pad_left:
            audio = np.pad(audio, (int(pad_left * self.sample_rate), 0), mode='constant', constant_values=0)
        if pad_right:
            audio = np.pad(audio, (0, int(pad_right * self.sample_rate)), mode='constant', constant_values=0)

        return audio, sr


    def __call__(self, batch):

        audios, srs, starts, ends = [], [], [], []
        batch_len = len(batch.get("filepath", []))
        for b_idx in range(batch_len):

            file_info = sf.info(batch["filepath"][b_idx])

            sr = file_info.samplerate
            duration = file_info.duration

            if (batch.get("start_time", []) or batch.get("end_time", [])) and (
                batch["start_time"][b_idx] or batch["end_time"][b_idx]
            ):
                start, end = batch["start_time"][b_idx], batch["end_time"][b_idx]
            else:
                start, end = None, None

            audio, sr = self._load_audio(batch["filepath"][b_idx], duration, start, end, sr)
            audios.append(audio)
            srs.append(sr)
            starts.append(start)
            ends.append(end)

        if not isinstance(batch.get("filepath", []), list):
            batch["filepath"] = batch["filepath"].tolist()
        if batch.get("filepath", None):
            batch["audio"] = [
                {"path": path, "array": audio, "samplerate": sr, 'start': start, 'end': end}
                for audio, path, sr, start, end in zip(audios, batch["filepath"], srs, starts, ends)
            ]
        return batch
import random

import numpy
import numpy as np
import soundfile as sf
import librosa


class EventDecoder:
    """
    Audio event decoding and preprocessing utility.

    This class loads audio files (or segments of audio files),
    optionally trims them to specified start/end times, resamples
    them to a target sampling rate, converts to mono if necessary,
    and pads boundary segments when required.

    It is designed to be used as a callable preprocessing step
    (e.g., inside dataset transforms).

    Parameters
    ----------
    sample_rate : int, optional
        Target sampling rate for all loaded audio. Audio will be
        resampled if the original sampling rate differs.
        Default: 32000.
    extracted_interval : float, optional
        Duration (in seconds) of the audio segment intended for
        extraction during event processing. (Currently stored but
        not enforced directly inside this class.)
        Default: 7.0.
    hf_cache_root : str, optional
        Root directory for cached datasets (reserved for external use).
        Default: "/data".

    Notes
    -----
    - Uses `soundfile` for partial audio loading.
    - Uses `librosa` for mono conversion and resampling.
    - Pads with zeros if requested segment exceeds file boundaries.
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
        """
        Load an audio file and apply preprocessing.

        Parameters
        ----------
        path : str
           Path to the audio file.
        duration : float
           Total duration of the audio file in seconds.
        start : float, optional
           Start time (in seconds) for partial loading.
        end : float, optional
           End time (in seconds) for partial loading.
        sr : int, optional
           Original sampling rate of the file.

        Returns
        -------
        tuple
           (audio_array, sample_rate)
           - audio_array : np.ndarray
               Mono, resampled, padded audio signal.
           - sample_rate : int
               Sampling rate after optional resampling.
        """
        pad_left = 0  # padding needed at beginning (seconds)
        pad_right = 0  # padding needed at end (seconds)

        # Handle boundary conditions for partial segments
        if start is not None and end is not None:
            if start < 0:  # first snippet case
                pad_left = - start
                start = 0
            if end > duration:  # last snippet case
                pad_right = end - duration
                end = duration

            start, end = int(start * sr), int(end * sr)

        # Load audio (full or partial)
        audio, sr = sf.read(path, start=start, stop=end)

        # Convert multi-channel to mono if necessary
        if audio.ndim != 1:
            audio = audio.swapaxes(1, 0)
            audio = librosa.to_mono(audio)

        # Resample if needed
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate

        # Apply zero-padding if segment exceeded boundaries
        if pad_left:
            audio = np.pad(audio, (int(pad_left * self.sample_rate), 0), mode='constant', constant_values=0)
        if pad_right:
            audio = np.pad(audio, (0, int(pad_right * self.sample_rate)), mode='constant', constant_values=0)

        return audio, sr

    def __call__(self, batch):
        """
        Decode and preprocess a batch of audio file entries.

        Parameters
        ----------
        batch : dict
            Dictionary containing at least:
                - "filepath": list of file paths
            Optionally:
                - "start_time": list of start times (seconds)
                - "end_time": list of end times (seconds)

        Returns
        -------
        dict
            Updated batch dictionary with a new key:
                "audio": list of dictionaries containing:
                    - "path": file path
                    - "array": processed audio waveform
                    - "samplerate": sampling rate
                    - "start": start time used
                    - "end": end time used

        Notes
        -----
        - Handles per-sample partial loading.
        - Converts non-list filepath inputs to list.
        - Preserves original batch structure while adding decoded audio.
        """
        audios, srs, starts, ends = [], [], [], []
        batch_len = len(batch.get("filepath", []))
        for b_idx in range(batch_len):

            file_info = sf.info(batch["filepath"][b_idx])

            sr = file_info.samplerate
            duration = file_info.duration

            # Determine start/end if provided
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

        # Ensure filepath is list-like
        if not isinstance(batch.get("filepath", []), list):
            batch["filepath"] = batch["filepath"].tolist()

        # Attach processed audio metadata
        if batch.get("filepath", None):
            batch["audio"] = [
                {"path": path, "array": audio, "samplerate": sr, 'start': start, 'end': end}
                for audio, path, sr, start, end in zip(audios, batch["filepath"], srs, starts, ends)
            ]
        return batch





class EventDecoderTrain(EventDecoder):
    def __init__(self, min_len: float = 1, max_len: float = 5, sample_rate: int = 32_000, extension_time: float = 8,
                 extracted_interval: float = 5, hf_cache_root: str = "/data"):
        super().__init__(sample_rate, extracted_interval, hf_cache_root)
        self.min_len = min_len  # in seconds
        self.max_len = max_len
        self.sample_rate = sample_rate
        self.extension_time = extension_time
        self.extracted_interval = extracted_interval

    def _load_audio(self, path, duration, start=None, end=None, sr=None):
        if start is not None and end is not None:
            if (end - start) < self.min_len:
                end = start + self.min_len
            if self.max_len and end - start > self.max_len:
                end = start + self.max_len
            start, end = int(start * sr), int(end * sr)
        if not end:
            end = int(self.max_len * sr)

        audio, sr = sf.read(path, start=start, stop=end)

        if audio.ndim != 1:
            audio = audio.swapaxes(1, 0)
            audio = librosa.to_mono(audio)
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate


        return audio, sr

    def _time_shifting(self, start, end, total_duration):
        event_duration = end - start

        if event_duration < self.extension_time:
            side_extension_time = (self.extension_time - event_duration) / 2
            new_start_time = max(0, start - side_extension_time)
            new_end_time = min(total_duration, end + side_extension_time)

            if new_end_time - new_start_time < self.extension_time:
                if new_start_time == 0:
                    new_end_time = min(self.extension_time, total_duration)
                elif new_end_time == total_duration:
                    new_start_time = max(0, total_duration - self.extension_time)

        else:  # longer than extraction time
            new_start_time = start
            new_end_time = end

        # Ensure max_start_interval is non-negative
        max_start_interval = max(0, new_end_time - self.extracted_interval)
        random_start = random.uniform(new_start_time, max_start_interval)
        random_end = random_start + self.extracted_interval
        return random_start, random_end

    def __call__(self, batch):

        audios, srs, starts, ends = [], [], [], []
        batch_len = len(batch.get("filepath", []))
        for b_idx in range(batch_len):

            try:
                file_info = sf.info(batch["filepath"][b_idx])
            except:
                print(batch["filepath"][b_idx], flush=True)

            sr = file_info.samplerate
            duration = file_info.duration

            if not isinstance(batch.get("detected_events", []), list):
                batch["detected_events"] = batch["detected_events"].tolist()
            if batch.get("detected_events", []) and batch["detected_events"][b_idx]:
                start, end = batch["detected_events"][b_idx]
                if self.extension_time:
                    # time shifting
                    start, end = self._time_shifting(start, end, duration)
            elif (batch.get("start_time", []) or batch.get("end_time", [])) and (
                    batch["start_time"][b_idx] or batch["end_time"][b_idx]
            ):
                start, end = batch["start_time"][b_idx], batch["end_time"][b_idx]
            else:
                # start, end = None, None
                if duration > self.extracted_interval:
                    start = random.uniform(0, duration - self.extracted_interval)
                else:
                    start = 0
                end = start + self.extracted_interval

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
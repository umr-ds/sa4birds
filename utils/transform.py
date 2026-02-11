from typing import List, Union

import numpy as np
import torch
from torchaudio.compliance.kaldi import fbank
from torchaudio.transforms import Spectrogram, MelScale
from torchvision import transforms
from transformers import SequenceFeatureExtractor, BatchFeature
from transformers.utils import PaddingStrategy

from utils.event_decoder import EventDecoding
from utils.power_to_db import PowerToDB
import torch.nn.functional as F


class DefaultFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ["input_values", "attention_mask"]

    def __init__(
        self,
        feature_size: int = 1,
        sampling_rate: int = 32000,
        padding_value: float = 0.0,
        return_attention_mask: bool = False,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        self.return_attention_mask = return_attention_mask

    def __call__(
        self,
        waveform: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        padding: Union[bool, str, PaddingStrategy] = False,
        max_length: int = None,
        truncation: bool = False,
        return_attention_mask: bool = False):

        waveform_encoded = BatchFeature({"input_values": waveform})

        padded_inputs = self.pad(
            waveform_encoded,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_attention_mask=return_attention_mask
        )

        padded_inputs["input_values"] = torch.from_numpy(padded_inputs["input_values"])
        attention_mask = padded_inputs.get("attention_mask")

        if attention_mask is not None:
            padded_inputs["attention_mask"] = attention_mask


        return padded_inputs


class BaseTransform:
    def __init__(self,
                 config,
                 train: bool = True,
                 noisy_training: bool= False,
                 event_decoder= EventDecoding()
                 ):
        self.train = train
        self.noisy_training = noisy_training
        self.config = config
        self.frontend_cfg = config.frontend
        self.target_length = self.frontend_cfg.val_target_length

        self.feature_extractor = DefaultFeatureExtractor(
            feature_size=1,
            sampling_rate=self.frontend_cfg.sample_rate,
            padding_value=0.0,
            return_attention_mask=False
        )

        self.event_decoder = event_decoder

        self.spectrogram_conversion = Spectrogram(
            n_fft=self.frontend_cfg.n_fft,
            hop_length=self.frontend_cfg.hop_length,
            power=self.frontend_cfg.power)

        self.melscale_conversion = MelScale(
            n_mels=self.frontend_cfg.n_mels,
            sample_rate=self.frontend_cfg.sample_rate,
            n_stft=self.frontend_cfg.n_stft)

        self.dbscale_conversion = PowerToDB()

    def _process_waveforms(self, waveforms):
        max_length = int(int(self.frontend_cfg.sample_rate) * self.event_decoder.extracted_interval)

        waveform_batch = self.feature_extractor(
            waveforms,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_attention_mask=False
        )
        return waveform_batch

    def _compute_fbank_features(self, waveforms):
        fbank_features = [
            fbank(
                waveform.unsqueeze(0),
                htk_compat=True,
                sample_frequency=self.frontend_cfg.sampling_rate,
                use_energy=False,
                window_type='hanning',
                num_mel_bins=self.config.n_mels,
                dither=0.0,
                frame_shift=10,
                frame_length=32
            )
            for waveform in waveforms
        ]
        return torch.stack(fbank_features)



    def get_spectrogram(self, waveforms):
        spectrograms = self.spectrogram_conversion(waveforms)
        spectrograms = self.melscale_conversion(spectrograms)
        fbank_features = self.dbscale_conversion(spectrograms) # .permute(0, 1, 3, 2)  # batch, 128, 501 --> batch, 501, 128
        return fbank_features

    def pad(self, fbank_features):
        difference = self.target_length - fbank_features.shape[-1]

        if difference != 0:
            print(f"warning: generated spectrogram has not the same time shape ({fbank_features.shape[-1]}) as target_length ({self.target_length})")

        min_value = fbank_features.min()
        if self.target_length > fbank_features.shape[-1]:
            # padding = (0, 0, 0 , difference)
            padding = (0, difference)
            fbank_features = F.pad(fbank_features, padding, value=min_value.item())  # no difference!
        return fbank_features

    def __call__(self, batch):
        pass


class ValTransform(BaseTransform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def __call__(self, batch):
        """
        Main data pipeline for processing an input batch:
        1. Decode waveform
        2. Apply wave augmentation
        3. create specs and pad
        4. Apply spec augmentation
        5. Optionally inject no-call samples
        6. Normalize & resize
        7. Package results
        """

        waveform_batch = self.decode_batch(batch)

        batch_samples = waveform_batch["input_values"].unsqueeze(1)
        targets = self.get_targets(batch, waveform_batch)

        fbank_features = self.pad(self.get_spectrogram(batch_samples))
        fbank_features = (fbank_features - self.frontend_cfg.mean) / (self.frontend_cfg.std * 2)

        if self.frontend_cfg.resize:
            transform = transforms.Resize((self.frontend_cfg.resize, self.frontend_cfg.resize))
            fbank_features = transform(fbank_features)

        # --- Step 7: Package results ---
        try:
            if self.frontend_cfg.in_chans == 3:
                fbank_features = fbank_features.repeat(1, 3, 1, 1)

            result = {
                "audio": fbank_features,
                "label": targets.float(),
                "filepath": batch["filepath"],
            }
            return result

        except Exception:
            return {"audio": fbank_features}


    def get_targets(self, batch, waveform_batch):
        if "labels" in batch:
            targets = torch.tensor(batch["labels"])
        else:
            targets = torch.zeros(waveform_batch["input_values"].shape[0], 1)
        return targets


    def check(self, segment, soundscape):
        segment_name = segment.split('/')[-1]
        split = segment_name.split("_")
        start, end = split[-2], split[-1].split('.')[0]
        s = soundscape.split('/')[-1].split('.')[0]
        return segment_name == f"{s}_{start}_{end}.ogg"



    def decode_batch(self, batch):

        for i in range(len(batch['filepath'])):
            split = batch['filepath'][i].split("_")
            start, end = float(split[-2]), float(split[-1].split('.')[0])
            if end - start < 5:
                end = start + 5
            assert self.check(batch['filepath'][i], batch["soundscape_filepath"][i])
            batch['filepath'][i] = batch["soundscape_filepath"][i]
            difference = self.event_decoder.extracted_interval - 5 # dur
            batch["start_time"][i] = start  - (difference / 2)
            batch["end_time"][i] = end + (difference / 2)

        waveform_batch = self.event_decoder(batch)
        waveform_batch = [audio["array"] for audio in waveform_batch["audio"]]
        waveform_batch = self._process_waveforms(waveform_batch)
        return waveform_batch

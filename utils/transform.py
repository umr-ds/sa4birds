from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from omegaconf import DictConfig
from torchaudio.transforms import Spectrogram, MelScale
from torchvision import transforms
from transformers import SequenceFeatureExtractor, BatchFeature
from transformers.utils import PaddingStrategy

from utils.event_decoder import EventDecoder
from utils.power_to_db import PowerToDB
import torch.nn.functional as F


class DefaultFeatureExtractor(SequenceFeatureExtractor):
    """
    Default waveform feature extractor.

    This class wraps raw audio waveforms into a format compatible
    with transformer-based models (e.g., Hugging Face models).
    It handles padding, truncation, and optional attention mask
    generation.

    Inherits from:
       SequenceFeatureExtractor

    Attributes
    ----------
    model_input_names : list[str]
       Names of expected model inputs.
       Default: ["input_values", "attention_mask"]

    Parameters
    ----------
    feature_size : int, optional
       Feature dimension size. Default: 1 (raw waveform).
    sampling_rate : int, optional
       Sampling rate of input audio. Default: 32000.
    padding_value : float, optional
       Value used for padding shorter sequences. Default: 0.0.
    return_attention_mask : bool, optional
       Whether to return an attention mask by default.
       Default: False.
    **kwargs :
       Additional keyword arguments passed to the parent
       `SequenceFeatureExtractor`.

    Notes
    -----
    - Input waveforms can be NumPy arrays, Python lists,
     or batched lists of arrays.
    - Output tensors are converted to PyTorch tensors.
    """
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
        """
        Process and optionally pad/truncate input waveform(s).

        Parameters
        ----------
        waveform : array-like or list
         Raw audio waveform(s). Can be:
             - np.ndarray (single waveform)
             - List[float]
             - List[np.ndarray]
             - List[List[float]]
        padding : bool or str or PaddingStrategy, optional
         Padding strategy:
             - False: no padding
             - True/"longest": pad to longest in batch
             - "max_length": pad to max_length
        max_length : int, optional
         Maximum length for padding or truncation.
        truncation : bool, optional
         Whether to truncate sequences longer than max_length.
        return_attention_mask : bool, optional
         Whether to return attention mask.

        Returns
        -------
        BatchFeature
         Dictionary containing:
             - "input_values": torch.Tensor
             - "attention_mask" (optional)
        """
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
    """
    Base audio preprocessing pipeline.

    This class defines a common frontend processing pipeline for
    audio classification models. It handles:

        1. Waveform padding and truncation
        2. Spectrogram computation
        3. Mel-scale conversion
        4. Log (dB) scaling
        5. Optional spectrogram padding to fixed time length

    It is designed to be subclassed (e.g., TrainTransform,
    ValTransform) where the `__call__` method is implemented.

    Parameters
    ----------
    config : DictConfig
        Configuration object containing frontend parameters such as:
            - sample_rate
            - n_fft
            - hop_length
            - power
            - n_mels
            - n_stft
            - val_target_length
    train : bool, optional
        Whether the transform is used during training.
        Default: True.
    noisy_training : bool, optional
        Enables noise-related augmentations (to be implemented
        in subclasses). Default: False.
    event_decoder : EventDecoding, optional
        Decoder used to load and preprocess raw audio files.

    Notes
    -----
    - Uses Spectrogram → MelScale → PowerToDB pipeline.
    - Pads spectrograms to a fixed temporal dimension.
    - `__call__` must be implemented by subclasses.
    """
    def __init__(self,
                 config : DictConfig,
                 train: bool = True,
                 event_decoder: EventDecoder= EventDecoder()
                 ):
        self.train = train
        self.config = config

        # Frontend configuration (FFT, mel bins, etc.)
        self.frontend_cfg = config.frontend

        # Target time dimension of final spectrogram
        self.target_length = self.frontend_cfg.val_target_length

        # --------------------------------------------------
        # Waveform processing
        # --------------------------------------------------
        self.feature_extractor = DefaultFeatureExtractor(
            feature_size=1,
            sampling_rate=self.frontend_cfg.sample_rate,
            padding_value=0.0,
            return_attention_mask=False
        )

        self.event_decoder = event_decoder

        # --------------------------------------------------
        # Spectrogram computation
        # --------------------------------------------------
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
        """
        Pad and truncate raw waveforms to fixed duration.

        Parameters
        ----------
        waveforms : list or tensor
            Batch of raw waveform arrays.

        Returns
        -------
        dict
            Padded waveform batch as returned by DefaultFeatureExtractor.
        """
        max_length = int(int(self.frontend_cfg.sample_rate) * self.event_decoder.extracted_interval)

        waveform_batch = self.feature_extractor(
            waveforms,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_attention_mask=False
        )
        return waveform_batch


    def get_spectrogram(self, waveforms):
        """
        Convert waveforms to log-mel spectrograms.

        Processing pipeline:
            1. Compute power spectrogram
            2. Convert to mel scale
            3. Convert to decibel (log) scale

        Returns
        -------
        torch.Tensor
            Log-mel spectrogram tensor.
        """
        spectrograms = self.spectrogram_conversion(waveforms)
        spectrograms = self.melscale_conversion(spectrograms)
        fbank_features = self.dbscale_conversion(spectrograms) # .permute(0, 1, 3, 2)  # batch, 128, 501 --> batch, 501, 128
        return fbank_features

    def pad(self, fbank_features):
        """
        Pad spectrograms to fixed time dimension.

        Parameters
        ----------
        fbank_features : torch.Tensor
           Spectrogram tensor with time dimension at index -1.

        Returns
        -------
        torch.Tensor
           Padded spectrogram tensor.
        """
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
        """
        Apply transformation to a dataset batch.

        Must be implemented in subclasses.

        Parameters
        ----------
        batch : dict
            Input batch from dataset.

        Raises
        ------
        NotImplementedError
            Always raised in base class.
        """
        raise NotImplementedError(
            "BaseTransform is an abstract class. "
            "Implement __call__ in a subclass."
        )


class ValTransform(BaseTransform):
    """
    Validation data transformation pipeline.

    This class implements the evaluation-time preprocessing
    pipeline. It extends `BaseTransform` and performs:

        1. Segment-to-soundscape alignment
        2. Audio decoding via EventDecoding
        3. Waveform padding/truncation
        4. Spectrogram computation (log-mel)
        5. Feature normalization
        6. Optional resizing
        7. Final packaging for model input

    Notes
    -----
    - No data augmentation is applied.
    - Designed for deterministic validation/testing.
    - Assumes `soundscape_filepath` exists in batch.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def __call__(self, batch):
        """
        Process a validation batch.

        Steps
        -----
        1. Decode waveform from soundscape
        2. Generate log-mel spectrogram
        3. Pad to fixed length
        4. Normalize
        5. Optional resizing
        6. Package into final dict

        Parameters
        ----------
        batch : dict
            Dataset batch containing:
                - "filepath"
                - "soundscape_filepath"
                - "labels" (optional)

        Returns
        -------
        dict
            {
                "audio": Tensor (B, C, H, W),
                "label": Tensor (B, num_classes),
                "filepath": list[str]
            }
        """
        # ------------------------------------------
        # Step 1: Decode waveform
        # ------------------------------------------
        waveform_batch = self.decode_batch(batch)

        batch_samples = waveform_batch["input_values"].unsqueeze(1)
        targets = self.get_targets(batch, waveform_batch)

        # ------------------------------------------
        # Step 2–3: Spectrogram + Padding
        # ------------------------------------------
        fbank_features = self.pad(self.get_spectrogram(batch_samples))

        # ------------------------------------------
        # Step 4: Normalize
        # ------------------------------------------
        fbank_features = (fbank_features - self.frontend_cfg.mean) / (self.frontend_cfg.std * 2)

        # ------------------------------------------
        # Step 5: Optional resizing
        # ------------------------------------------
        if self.frontend_cfg.resize:
            transform = transforms.Resize((self.frontend_cfg.resize, self.frontend_cfg.resize))
            fbank_features = transform(fbank_features)

        # ------------------------------------------
        # Step 6: Package results
        # ------------------------------------------
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
        """
        Extract target labels from batch.

        Parameters
        ----------
        batch : dict
        waveform_batch : dict

        Returns
        -------
        torch.Tensor
            Target tensor of shape (B, num_classes).
        """
        if "labels" in batch:
            targets = torch.tensor(batch["labels"])
        else:
            targets = torch.zeros(waveform_batch["input_values"].shape[0], 1)
        return targets


    def check(self, segment, soundscape):
        """
        Verify that segment filename matches soundscape file.

        Used as a sanity check during decoding.

        Returns
        -------
        bool
        """
        segment_name = Path(segment).name
        split = segment_name.split("_")

        start, end = split[-2], split[-1].split('.')[0]
        s = Path(soundscape).stem

        return segment_name == f"{s}_{start}_{end}.ogg"



    def decode_batch(self, batch):
        """
        Align segment metadata with full soundscape and
        decode audio using EventDecoding.

        Steps
        -----
        1. Replace segment path with soundscape path
        2. Expand segment window to extracted_interval
        3. Decode audio
        4. Pad/truncate waveform

        Returns
        -------
        dict
           Output from `_process_waveforms`
        """
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




class ValTransformBeans(ValTransform):
    def decode_batch(self, batch):
        waveform_batch = self.event_decoder(batch)
        waveform_batch = [audio["array"] for audio in waveform_batch["audio"]]
        waveform_batch = self._process_waveforms(waveform_batch)
        return waveform_batch
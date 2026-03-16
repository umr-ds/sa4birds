import glob
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torchaudio.transforms import FrequencyMasking, TimeMasking

from utils.augmentation import Compose
from utils.transform import BaseTransform, DefaultFeatureExtractor


class TrainTransform(BaseTransform):
    """
    Training data transformation pipeline.

    Extends BaseTransform and applies:

      1. Waveform decoding
      2. Waveform augmentation
      3. Spectrogram computation + padding
      4. Spectrogram augmentation (SpecAugment, FilterAug, Mixup)
      5. Optional no-call sample injection
      6. Normalization
      7. Output packaging

    Designed for supervised multi-label training.

    Configuration Dependencies
    ---------------------------
    - config.frontend
    - config.augmentation.wave_aug
    - config.augmentation.spec_aug
    - config.event_decoder.train
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.train = True
        self.frontend_cfg = self.config.frontend
        self.augment_cfg = self.config.augmentation

        # Use training target length
        self.target_length = self.frontend_cfg.train_target_length

        # --------------------------------------------------
        # Collect no-call samples
        # --------------------------------------------------
        self.no_call_filepaths = []
        if self.augment_cfg.wave_aug is not None:
            if self.augment_cfg.wave_aug.no_call is not None:
                for p in self.augment_cfg.wave_aug.no_call.dirs:
                    self.no_call_filepaths += glob.glob(f"{p}/*.wav")

        # --------------------------------------------------
        # Waveform augmentations
        # --------------------------------------------------
        self.wave_augment = None
        if self.augment_cfg.wave_aug:
            print([v for k, v in self.augment_cfg.wave_aug.items() if
                    isinstance(v, DictConfig) and "_target_" in v])
            tfms = [hydra.utils.instantiate(v) for k, v in self.augment_cfg.wave_aug.items() if
                    isinstance(v, DictConfig) and "_target_" in v]
            self.wave_augment = Compose(transforms=tfms, output_type="object_dict")


        self.feature_extractor = DefaultFeatureExtractor(
            feature_size=1,
            sampling_rate=self.frontend_cfg.sample_rate,
            padding_value=0.0,
            return_attention_mask=False
        )

        # --------------------------------------------------
        # SpecAugment masking
        # --------------------------------------------------
        self.freqm = None
        self.timem = None
        if self.train:
            if self.augment_cfg.spec_aug.masking.freqm:
                self.freqm = FrequencyMasking(freq_mask_param=self.augment_cfg.spec_aug.masking.freqm)
            if self.augment_cfg.spec_aug.masking.timem:
                self.timem = TimeMasking(time_mask_param=self.augment_cfg.spec_aug.masking.timem)

        if self.train:
            self.filteraug = None
            if self.augment_cfg.spec_aug.filteraug:
                self.filteraug = hydra.utils.instantiate(self.augment_cfg.spec_aug.filteraug)


    def __call__(self, batch):
        """
        Full training preprocessing pipeline.
        """

        # --------------------------------------------------
        # 1. Decode waveform
        # --------------------------------------------------
        waveform_batch = self.decode_batch(batch)


        batch_samples = waveform_batch["input_values"].unsqueeze(1)

        targets = self.get_targets(batch, waveform_batch)

        # --------------------------------------------------
        # 2. Waveform augmentation
        # --------------------------------------------------
        batch_samples, targets = self.apply_wave_aug(batch_samples, targets)

        # --------------------------------------------------
        # 3. Spectrogram computation + padding
        # --------------------------------------------------
        fbank_features = self.pad(self.get_spectrogram(batch_samples))

        # --------------------------------------------------
        # 4. Spectrogram augmentation
        # --------------------------------------------------
        fbank_features, targets = self.apply_spec_aug(fbank_features, targets)

        # --------------------------------------------------
        # 5. Optional no-call injection
        # --------------------------------------------------
        if self.train and self.no_call_filepaths and random.random() < self.augment_cfg.wave_aug.no_call.p:
            nocall_batch = self.decode_batch({
                "filepath": random.sample(
                    self.no_call_filepaths,
                    k=self.augment_cfg.wave_aug.no_call.num_per_batch
                )
            })

            nocall_samples = nocall_batch["input_values"].unsqueeze(1)
            nocall_targets = torch.zeros(len(nocall_samples), targets.shape[1])

            # Augment no-call samples
            nocall_samples, nocall_targets = self.apply_wave_aug(nocall_samples, nocall_targets)

            nocall_fbank = self.pad(self.get_spectrogram(nocall_samples))


            nocall_fbank, nocall_targets = self.apply_spec_aug(nocall_fbank, nocall_targets)

            # Replace first N samples in batch
            n = len(nocall_fbank)
            fbank_features[:n] = nocall_fbank
            targets[:n] = nocall_targets

        # --------------------------------------------------
        # 6. Normalize
        # --------------------------------------------------
        fbank_features = (fbank_features - self.frontend_cfg.mean) / (self.frontend_cfg.std * 2)

        # --------------------------------------------------
        # 7. Package output
        # --------------------------------------------------
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

    def apply_spec_aug(self, fbank_features, targets):
        """
        Apply spectrogram-level augmentations.

        This method applies three possible augmentations:

            1. SpecAugment (frequency & time masking)
            2. Filter-based augmentation
            3. Mixup (feature + label mixing)

        Parameters
        ----------
        fbank_features : torch.Tensor
            Log-mel spectrogram tensor of shape (B, C, F, T).
        targets : torch.Tensor
            Multi-label targets of shape (B, num_classes).

        Returns
        -------
        tuple
            (fbank_features, targets)
                Augmented spectrogram features and corresponding targets.

        Notes
        -----
        - Each augmentation is applied conditionally based on
          probabilities defined in the configuration.
        - Mixup modifies both features and targets.
        - SpecAugment operates independently on each sample.
        """

        # --------------------------------------------------
        # 1. SpecAugment (Frequency + Time Masking)
        # --------------------------------------------------
        # Applied with probability self.augment_cfg.spec_aug.masking.p
        if self.train and np.random.uniform() < self.augment_cfg.spec_aug.masking.p:
            # Apply frequency masking per sample
            fbank_features = torch.stack([self.freqm(feature) for feature in fbank_features])
            # Apply time masking per sample
            fbank_features = torch.stack([self.timem(feature) for feature in fbank_features])

        # --------------------------------------------------
        # 2. Filter-based augmentation
        # --------------------------------------------------
        # Applies global spectrogram filtering (if enabled)
        if self.train and self.filteraug:
            fbank_features = self.filteraug(fbank_features)

        # --------------------------------------------------
        # 3. Mixup augmentation
        # --------------------------------------------------
        # Applied with probability self.augment_cfg.spec_aug.mix.p
        if self.train and random.random() < self.augment_cfg.spec_aug.mix.p:
            fbank_features, targets, _ = mixup_hard(fbank_features,
                                                    alpha=self.augment_cfg.spec_aug.mix.alpha,
                                                    beta=self.augment_cfg.spec_aug.mix.beta,
                                                    targets=targets)
        return fbank_features, targets

    def apply_wave_aug(self, batch_samples, targets):
        """
        Apply waveform-level augmentations.

        This method performs augmentations directly on raw waveform
        inputs before spectrogram computation.

        Parameters
        ----------
        batch_samples : torch.Tensor
            Input waveforms of shape (B, 1, T).
        targets : torch.Tensor
            Multi-label targets of shape (B, num_classes).

        Returns
        -------
        tuple
            (batch_samples, targets)
                Augmented waveforms and corresponding targets.

        Notes
        -----
        - Augmentations are only applied during training.
        - Targets are temporarily reshaped to match augmentation
          library expectations.
        - Augmentation pipeline is defined via Hydra config.
        - Cyclic rolling is declared but not implemented.
        """
        if self.train:
            # --------------------------------------------------
            # Optional cyclic rolling (currently not implemented)
            # --------------------------------------------------
            if self.augment_cfg.wave_aug.cyclic_rolling_start:
                raise NotImplementedError

            # --------------------------------------------------
            # Prepare targets for augmentation pipeline
            # Some augmentation frameworks expect shape (B, 1, 1, C)
            # --------------------------------------------------
            targets = targets.unsqueeze(1).unsqueeze(1)
            # --------------------------------------------------
            # Apply waveform augmentations
            # --------------------------------------------------
            aug_output = self.wave_augment(samples=batch_samples,
                                           targets=targets,
                                           sample_rate=self.frontend_cfg.sample_rate)
            batch_samples = aug_output["samples"]

            # Restore target shape back to (B, num_classes)
            targets = aug_output["targets"][0].squeeze(1).squeeze(1)

        return batch_samples, targets

    def get_targets(self, batch, waveform_batch):
        """
        Extract target labels from a batch.

        This method retrieves multi-label targets from the batch
        dictionary. If labels are not present (e.g., inference mode),
        a zero tensor is returned as a placeholder.

        Parameters
        ----------
        batch : dict
            Input batch containing:
                - "labels" (optional): multi-label targets
        waveform_batch : dict
            Processed waveform batch returned by `_process_waveforms`,
            used only to infer batch size when labels are missing.

        Returns
        -------
        torch.Tensor
            Target tensor of shape:
                - (B, num_classes) if labels exist
                - (B, 1) if labels are missing

        Notes
        -----
        - Labels are converted to a torch tensor.
        - If labels are absent (e.g., during inference),
          dummy zero targets are created.
        """
        if "labels" in batch:
            targets = torch.tensor(batch["labels"])
        else:
            targets = torch.zeros(waveform_batch["input_values"].shape[0], 1)
        return targets

    def decode_batch(self, batch):
        waveform_batch = self.event_decoder(batch)
        waveform_batch = [audio["array"] for audio in waveform_batch["audio"]]
        waveform_batch = self._process_waveforms(waveform_batch)
        return waveform_batch

@staticmethod
def mixup_hard(x, alpha, beta, targets=None, targets_pl=None):
    """
    Apply hard mixup augmentation to a batch.

    This function performs sample-level mixup by randomly pairing
    samples within a batch and linearly combining their inputs.
    Unlike standard mixup (which linearly mixes targets), this
    variant performs "hard" target merging:

        - For multi-label targets:
            targets are added and clamped to [0, 1]
        - For pseudo-labels (targets_pl):
            element-wise maximum is taken

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (B, C, H, W).
    alpha : float
        Lower bound for mixup weight sampling.
    beta : float
        Scaling factor controlling mixup strength.
        Final weight is sampled as:
            alpha + beta * U(0,1)
    targets : torch.Tensor, optional
        Multi-label targets of shape (B, num_classes).
    targets_pl : torch.Tensor, optional
        Optional pseudo-label targets.

    Returns
    -------
    tuple
        mixed_x : torch.Tensor
            Mixed inputs.
        mixed_t : torch.Tensor or None
            Mixed targets (if provided).
        mixed_t_pl : torch.Tensor or None
            Mixed pseudo-label targets (if provided).

    Notes
    -----
    - Mixup is performed within-batch using random permutation.
    - "Hard" mixing keeps labels binary instead of convex
      combinations.
    - Suitable for multi-label classification tasks.
    """
    batch_size = x.size(0)
    # --------------------------------------------------
    # Sample mixup weights per sample
    # --------------------------------------------------
    weights = alpha + beta * torch.rand(
        batch_size,
        device=x.device
    )

    # Random permutation of batch indices
    indices = torch.randperm(batch_size).to(x.device)
    # --------------------------------------------------
    # Mix inputs
    # --------------------------------------------------
    mixed_x = (1 - weights.view(-1, 1, 1, 1)) * x + weights.view(-1, 1, 1, 1) * x[indices]

    # --------------------------------------------------
    # Mix targets (hard merge)
    # --------------------------------------------------
    if targets is not None:
        mixed_t = targets + targets[indices]
        mixed_t = mixed_t.clamp(min=0, max=1)
    else:
        mixed_t = None

    # --------------------------------------------------
    # Mix pseudo-labels (optional)
    # --------------------------------------------------
    if targets_pl is not None:
        mixed_t_pl = torch.maximum(targets_pl, targets_pl[indices])
    else:
        mixed_t_pl = None

    return mixed_x, mixed_t, mixed_t_pl

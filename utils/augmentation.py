import math
import warnings
import random
from pathlib import Path
from typing import Optional, Union, List

import librosa
import torchaudio
from audiomentations.core.utils import convert_decibels_to_amplitude_ratio
from omegaconf import ListConfig
from torch import Tensor
from torch.distributions import Bernoulli
from torch_audiomentations.augmentations.colored_noise import _gen_noise
from torch_audiomentations.core.transforms_interface import EmptyPathException, ModeNotSupportedException, \
    MultichannelAudioNotSupportedException
from torch_audiomentations.utils.dsp import calculate_rms
from torch_audiomentations.utils.file import find_audio_files_in_paths
from torch_audiomentations.utils.io import AudioFile
from torch_audiomentations.utils.multichannel import is_multichannel
from torch_audiomentations.utils.object_dict import ObjectDict
import torch

class BaseCompose(torch.nn.Module):
    """This class can apply a sequence of transforms to waveforms."""

    def __init__(
        self,
        transforms: List[
            torch.nn.Module
        ],  # FIXME: do we really want to support regular nn.Module?
        shuffle: bool = False,
        p: float = 1.0,
        p_mode="per_batch",
        output_type: Optional[str] = None,
    ):
        """
        :param transforms: List of waveform transform instances
        :param shuffle: Should the order of transforms be shuffled?
        :param p: The probability of applying the Compose to the given batch.
        :param p_mode: Only "per_batch" is supported at the moment.
        :param output_type: This optional argument can be set to "tensor" or "dict".
        """
        super().__init__()
        self.p = p
        if p_mode != "per_batch":
            # TODO: Support per_example as well? And per_channel?
            raise ValueError(f'p_mode = "{p_mode}" is not supported')
        self.p_mode = p_mode
        self.shuffle = shuffle
        self.are_parameters_frozen = False

        if output_type is None:
            warnings.warn(
                f"Transforms now expect an `output_type` argument that currently defaults to 'tensor', "
                f"will default to 'dict' in v0.12, and will be removed in v0.13. Make sure to update "
                f"your code to something like:\n"
                f"  >>> augment = {self.__class__.__name__}(..., output_type='dict')\n"
                f"  >>> augmented_samples = augment(samples).samples",
                FutureWarning,
            )
            output_type = "tensor"

        elif output_type == "tensor":
            warnings.warn(
                f"`output_type` argument will default to 'dict' in v0.12, and will be removed in v0.13. "
                f"Make sure to update your code to something like:\n"
                f"your code to something like:\n"
                f"  >>> augment = {self.__class__.__name__}(..., output_type='dict')\n"
                f"  >>> augmented_samples = augment(samples).samples",
                DeprecationWarning,
            )

        self.output_type = output_type

        self.transforms = torch.nn.ModuleList(transforms)
        for tfm in self.transforms:
            tfm.output_type = "dict"

    def freeze_parameters(self):
        """
        Mark all parameters as frozen, i.e. do not randomize them for each call. This can be
        useful if you want to apply an effect chain with the exact same parameters to multiple
        sounds.
        """
        self.are_parameters_frozen = True
        for transform in self.transforms:
            transform.freeze_parameters()

    def unfreeze_parameters(self):
        """
        Unmark all parameters as frozen, i.e. let them be randomized for each call.
        """
        self.are_parameters_frozen = False
        for transform in self.transforms:
            transform.unfreeze_parameters()

    @property
    def supported_modes(self) -> set:
        """Return the intersection of supported modes of the transforms in the composition."""
        currently_supported_modes = {"per_batch", "per_example", "per_channel"}
        for transform in self.transforms:
            currently_supported_modes = currently_supported_modes.intersection(
                transform.supported_modes
            )
        return currently_supported_modes


class Compose(BaseCompose):
    def forward(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:

        inputs = ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )

        mix_keys = dict()
        if random.random() < self.p:
            transform_indexes = list(range(len(self.transforms)))
            if self.shuffle:
                random.shuffle(transform_indexes)

            for i in transform_indexes:
                tfm = self.transforms[i]
                if isinstance(tfm, (BaseWaveformTransform, BaseCompose)):
                    inputs = self.transforms[i](**inputs)
                    if 'mix_indices' in inputs:
                        mix_keys['mix_indices'] = inputs['mix_indices']
                        del inputs['mix_indices']

                else:
                    assert isinstance(tfm, torch.nn.Module)
                    inputs.samples = self.transforms[i](inputs.samples)

        if 'mix_indices' in mix_keys:
            inputs['mix_indices'] = mix_keys['mix_indices']

        return inputs.samples if self.output_type == "tensor" else inputs




class BaseWaveformTransform(torch.nn.Module):

    supported_modes = {"per_batch", "per_example", "per_channel"}

    supports_multichannel = True
    requires_sample_rate = True

    supports_target = True
    requires_target = False

    def __init__(
        self,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: Optional[str] = None,
        sample_rate: Optional[int] = None,
        target_rate: Optional[int] = None,
        output_type: Optional[str] = None,
    ):
        """

        :param mode:
            mode="per_channel" means each channel gets processed independently.
            mode="per_example" means each (multichannel) audio snippet gets processed
                independently, i.e. all channels in each audio snippet get processed with the
                same parameters.
            mode="per_batch" means all (multichannel) audio snippets in the batch get processed
                with the same parameters.
        :param p: The probability of the transform being applied to a batch/example/channel
            (see mode and p_mode). This number must be in the range [0.0, 1.0].
        :param p_mode: This optional argument can be set to "per_example" or "per_channel" if
            mode is set to "per_batch", or it can be set to "per_channel" if mode is set to
            "per_example". In the latter case, the transform is applied to the randomly selected
            examples, but the channels in those examples will be processed independently, i.e.
            with different parameters. Default value: Same as mode.
        :param sample_rate: sample_rate can be set either here or when
            calling the transform.
        :param target_rate: target_rate can be set either here or when
            calling the transform.
        :param output_type: This optional argument can be set to "tensor" or "dict".

        """
        super().__init__()
        assert 0.0 <= p <= 1.0
        self.mode = mode
        self._p = p
        self.p_mode = p_mode
        if self.p_mode is None:
            self.p_mode = self.mode
        self.sample_rate = sample_rate
        self.target_rate = target_rate

        if output_type is None:
            warnings.warn(
                f"Transforms now expect an `output_type` argument that currently defaults to 'tensor', "
                f"will default to 'dict' in v0.12, and will be removed in v0.13. Make sure to update "
                f"your code to something like:\n"
                f"  >>> augment = {self.__class__.__name__}(..., output_type='dict')\n"
                f"  >>> augmented_samples = augment(samples).samples",
                FutureWarning,
            )
            output_type = "tensor"

        elif output_type == "tensor":
            warnings.warn(
                f"`output_type` argument will default to 'dict' in v0.12, and will be removed in v0.13. "
                f"Make sure to update your code to something like:\n"
                f"your code to something like:\n"
                f"  >>> augment = {self.__class__.__name__}(..., output_type='dict')\n"
                f"  >>> augmented_samples = augment(samples).samples",
                DeprecationWarning,
            )

        self.output_type = output_type

        # Check validity of mode/p_mode combination
        if self.mode not in self.supported_modes:
            raise ModeNotSupportedException(
                "{} does not support mode {}".format(self.__class__.__name__, self.mode)
            )
        if self.p_mode == "per_batch":
            assert self.mode in ("per_batch", "per_example", "per_channel")
        elif self.p_mode == "per_example":
            assert self.mode in ("per_example", "per_channel")
        elif self.p_mode == "per_channel":
            assert self.mode == "per_channel"
        else:
            raise Exception("Unknown p_mode {}".format(self.p_mode))

        self.transform_parameters = {}
        self.are_parameters_frozen = False
        self.bernoulli_distribution = Bernoulli(self._p)

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, p):
        self._p = p
        # Update the Bernoulli distribution accordingly
        self.bernoulli_distribution = Bernoulli(self._p)

    def forward(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Union[Tensor, List[Tensor]]] = None,
        target_rate: Optional[int] = None,
        # TODO: add support for additional **kwargs (batch_size, ...)-shaped tensors
        # TODO: but do that only when we actually have a use case for it...
    ) -> ObjectDict:

        if not self.training:
            output = ObjectDict(
                samples=samples,
                sample_rate=sample_rate,
                targets=targets,
                target_rate=target_rate,
            )
            return output.samples if self.output_type == "tensor" else output

        if not isinstance(samples, Tensor) or len(samples.shape) != 3:
            raise RuntimeError(
                "torch-audiomentations expects three-dimensional input tensors, with"
                " dimension ordering like [batch_size, num_channels, num_samples]. If your"
                " audio is mono, you can use a shape like [batch_size, 1, num_samples]."
            )

        batch_size, num_channels, num_samples = samples.shape

        if batch_size * num_channels * num_samples == 0:
            warnings.warn(
                "An empty samples tensor was passed to {}".format(self.__class__.__name__)
            )
            output = ObjectDict(
                samples=samples,
                sample_rate=sample_rate,
                targets=targets,
                target_rate=target_rate,
            )
            return output.samples if self.output_type == "tensor" else output

        if is_multichannel(samples):
            if num_channels > num_samples:
                warnings.warn(
                    "Multichannel audio must have channels first, not channels last. In"
                    " other words, the shape must be (batch size, channels, samples), not"
                    " (batch_size, samples, channels)"
                )

            if not self.supports_multichannel:
                raise MultichannelAudioNotSupportedException(
                    "{} only supports mono audio, not multichannel audio".format(
                        self.__class__.__name__
                    )
                )

        sample_rate = sample_rate or self.sample_rate
        if sample_rate is None and self.is_sample_rate_required():
            raise RuntimeError("sample_rate is required")

        if targets is None and self.is_target_required():
            raise RuntimeError("targets is required")

        has_targets = targets is not None

        if has_targets and not self.supports_target:
            warnings.warn(f"Targets are not (yet) supported by {self.__class__.__name__}")

        if has_targets:
            if  isinstance(targets, List):
                to_check = targets[0]
            else:
                to_check = targets
                targets = [targets]

            if not isinstance(to_check, Tensor) or len(to_check.shape) != 4:

                raise RuntimeError(
                    "torch-audiomentations expects four-dimensional target tensors, with"
                    " dimension ordering like [batch_size, num_channels, num_frames, num_classes]."
                    " If your target is binary, you can use a shape like [batch_size, num_channels, num_frames, 1]."
                    " If your target is for the whole channel, you can use a shape like [batch_size, num_channels, 1, num_classes]."
                )

            (
                target_batch_size,
                target_num_channels,
                num_frames,
                num_classes,
            ) = to_check.shape

            if target_batch_size != batch_size:
                raise RuntimeError(
                    f"samples ({batch_size}) and target ({target_batch_size}) batch sizes must be equal."
                )
            if num_channels != target_num_channels:
                raise RuntimeError(
                    f"samples ({num_channels}) and target ({target_num_channels}) number of channels must be equal."
                )

            target_rate = target_rate or self.target_rate
            if target_rate is None:
                if num_frames > 1:
                    target_rate = round(sample_rate * num_frames / num_samples)
                    warnings.warn(
                        f"target_rate is required by {self.__class__.__name__}. "
                        f"It has been automatically inferred from targets shape to {target_rate}. "
                        f"If this is incorrect, you can pass it directly."
                    )
                else:
                    # corner case where num_frames == 1, meaning that the target is for the whole sample,
                    # not frame-based. we arbitrarily set target_rate to 0.
                    target_rate = 0

        if not self.are_parameters_frozen:

            if self.p_mode == "per_example":
                p_sample_size = batch_size

            elif self.p_mode == "per_channel":
                p_sample_size = batch_size * num_channels

            elif self.p_mode == "per_batch":
                p_sample_size = 1

            else:
                raise Exception("Invalid mode")

            self.transform_parameters = {
                "should_apply": self.bernoulli_distribution.sample(
                    sample_shape=(p_sample_size,)
                ).to(torch.bool)
            }

        if self.transform_parameters["should_apply"].any():

            cloned_samples = samples.clone()

            if has_targets:
                cloned_targets = [t.clone() for t in targets]
            else:
                cloned_targets = None
                selected_targets = None

            if self.p_mode == "per_channel":
                pass

            elif self.p_mode == "per_example":

                selected_samples = cloned_samples[
                    self.transform_parameters["should_apply"]
                ]

                if has_targets:
                    selected_targets = [t[self.transform_parameters["should_apply"]] for t in cloned_targets]


                if self.mode == "per_example":

                    if not self.are_parameters_frozen:
                        self.randomize_parameters(
                            samples=selected_samples,
                            sample_rate=sample_rate,
                            targets=selected_targets,
                            target_rate=target_rate,
                        )

                    perturbed: ObjectDict = self.apply_transform(
                        samples=selected_samples,
                        sample_rate=sample_rate,
                        targets=selected_targets,
                        target_rate=target_rate,
                    )

                    cloned_samples[
                        self.transform_parameters["should_apply"]
                    ] = perturbed.samples

                    if has_targets:
                        for i in range(len(cloned_targets)):
                            cloned_targets[i][self.transform_parameters["should_apply"]] = perturbed.targets[i]

                    output = ObjectDict(
                        samples=cloned_samples,
                        sample_rate=perturbed.sample_rate,
                        targets=cloned_targets,
                        target_rate=perturbed.target_rate,
                    )
                    return output.samples if self.output_type == "tensor" else output

                elif self.mode == "per_channel":

                    (
                        selected_batch_size,
                        selected_num_channels,
                        selected_num_samples,
                    ) = selected_samples.shape

                    assert selected_num_samples == num_samples

                    selected_samples = selected_samples.reshape(
                        selected_batch_size * selected_num_channels,
                        1,
                        selected_num_samples,
                    )

                    if has_targets:
                        selected_targets = selected_targets.reshape(
                            selected_batch_size * selected_num_channels,
                            1,
                            num_frames,
                            num_classes,
                        )

                    if not self.are_parameters_frozen:
                        self.randomize_parameters(
                            samples=selected_samples,
                            sample_rate=sample_rate,
                            targets=selected_targets,
                            target_rate=target_rate,
                        )

                    perturbed: ObjectDict = self.apply_transform(
                        selected_samples,
                        sample_rate=sample_rate,
                        targets=selected_targets,
                        target_rate=target_rate,
                    )

                    perturbed.samples = perturbed.samples.reshape(
                        selected_batch_size, selected_num_channels, selected_num_samples
                    )
                    cloned_samples[
                        self.transform_parameters["should_apply"]
                    ] = perturbed.samples

                    if has_targets:
                        perturbed.targets = perturbed.targets.reshape(
                            selected_batch_size,
                            selected_num_channels,
                            num_frames,
                            num_classes,
                        )
                        cloned_targets[
                            self.transform_parameters["should_apply"]
                        ] = perturbed.targets

                    output = ObjectDict(
                        samples=cloned_samples,
                        sample_rate=perturbed.sample_rate,
                        targets=cloned_targets,
                        target_rate=perturbed.target_rate,
                    )
                    return output.samples if self.output_type == "tensor" else output

                else:
                    raise Exception("Invalid mode/p_mode combination")

            elif self.p_mode == "per_batch":

                if self.mode == "per_batch":

                    cloned_samples = cloned_samples.reshape(
                        1, batch_size * num_channels, num_samples
                    )

                    if has_targets:
                        cloned_targets = cloned_targets.reshape(
                            1, batch_size * num_channels, num_frames, num_classes
                        )

                    if not self.are_parameters_frozen:
                        self.randomize_parameters(
                            samples=cloned_samples,
                            sample_rate=sample_rate,
                            targets=cloned_targets,
                            target_rate=target_rate,
                        )

                    perturbed: ObjectDict = self.apply_transform(
                        samples=cloned_samples,
                        sample_rate=sample_rate,
                        targets=cloned_targets,
                        target_rate=target_rate,
                    )
                    perturbed.samples = perturbed.samples.reshape(
                        batch_size, num_channels, num_samples
                    )

                    if has_targets:
                        perturbed.targets = perturbed.targets.reshape(
                            batch_size, num_channels, num_frames, num_classes
                        )

                    return (
                        perturbed.samples if self.output_type == "tensor" else perturbed
                    )

                elif self.mode == "per_example":

                    if not self.are_parameters_frozen:
                        self.randomize_parameters(
                            samples=cloned_samples,
                            sample_rate=sample_rate,
                            targets=cloned_targets,
                            target_rate=target_rate,
                        )

                    perturbed = self.apply_transform(
                        samples=cloned_samples,
                        sample_rate=sample_rate,
                        targets=cloned_targets,
                        target_rate=target_rate,
                    )

                    return (
                        perturbed.samples if self.output_type == "tensor" else perturbed
                    )

                elif self.mode == "per_channel":

                    cloned_samples = cloned_samples.reshape(
                        batch_size * num_channels, 1, num_samples
                    )

                    if has_targets:
                        cloned_targets = cloned_targets.reshape(
                            batch_size * num_channels, 1, num_frames, num_classes
                        )

                    if not self.are_parameters_frozen:
                        self.randomize_parameters(
                            samples=cloned_samples,
                            sample_rate=sample_rate,
                            targets=cloned_targets,
                            target_rate=target_rate,
                        )

                    perturbed: ObjectDict = self.apply_transform(
                        cloned_samples,
                        sample_rate,
                        targets=cloned_targets,
                        target_rate=target_rate,
                    )

                    perturbed.samples = perturbed.samples.reshape(
                        batch_size, num_channels, num_samples
                    )

                    if has_targets:
                        perturbed.targets = perturbed.targets.reshape(
                            batch_size, num_channels, num_frames, num_classes
                        )

                    return (
                        perturbed.samples if self.output_type == "tensor" else perturbed
                    )

                else:
                    raise Exception("Invalid mode")

            else:
                raise Exception("Invalid p_mode {}".format(self.p_mode))

        output = ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )
        return output.samples if self.output_type == "tensor" else output

    def _forward_unimplemented(self, *inputs) -> None:
        # Avoid IDE error message like "Class ... must implement all abstract methods"
        # See also https://github.com/python/mypy/issues/8795#issuecomment-691658758
        pass

    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        pass

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) :

        raise NotImplementedError()

    def serialize_parameters(self):
        """Return the parameters as a JSON-serializable dict."""
        raise NotImplementedError()
        # TODO: Clone the params and convert any tensors into json-serializable lists
        # return self.transform_parameters

    def freeze_parameters(self):
        """
        Mark all parameters as frozen, i.e. do not randomize them for each call. This can be
        useful if you want to apply an effect with the exact same parameters to multiple sounds.
        """
        self.are_parameters_frozen = True

    def unfreeze_parameters(self):
        """
        Unmark all parameters as frozen, i.e. let them be randomized for each call.
        """
        self.are_parameters_frozen = False

    def is_sample_rate_required(self) -> bool:
        return self.requires_sample_rate

    def is_target_required(self) -> bool:
        return self.requires_target


class Audio:
    """Audio IO with on-the-fly resampling

    Parameters
    ----------
    sample_rate: int
        Target sample rate.
    mono : int, optional
        Convert multi-channel to mono. Defaults to True.

    Usage
    -----
    >>> audio = Audio(sample_rate=16000)
    >>> samples = audio("/path/to/audio.wav")

    # on-the-fly resampling
    >>> original_sample_rate = 44100
    >>> two_seconds_stereo = torch.rand(2, 2 * original_sample_rate)
    >>> samples = audio({"samples": two_seconds_stereo, "sample_rate": original_sample_rate})
    >>> assert samples.shape[1] == 2 * 16000
    """

    @staticmethod
    def is_valid(file: AudioFile) -> bool:

        if isinstance(file, dict):

            if "samples" in file:

                samples = file["samples"]
                if len(samples.shape) != 2 or samples.shape[0] > samples.shape[1]:
                    raise ValueError(
                        "'samples' must be provided as a (channel, time) torch.Tensor."
                    )

                sample_rate = file.get("sample_rate", None)
                if sample_rate is None:
                    raise ValueError(
                        "'samples' must be provided with their 'sample_rate'."
                    )
                return True

            elif "audio" in file:
                return True

            else:
                # TODO improve error message
                raise ValueError("either 'audio' or 'samples' key must be provided.")

        return True

    @staticmethod
    def rms_normalize(samples: Tensor) -> Tensor:
        """Power-normalize samples

        Parameters
        ----------
        samples : (..., time) Tensor
            Single (or multichannel) samples or batch of samples

        Returns
        -------
        samples: (..., time) Tensor
            Power-normalized samples
        """
        rms = samples.square().mean(dim=-1, keepdim=True).sqrt()
        return samples / (rms + 1e-8)

    @staticmethod
    def get_audio_metadata(file_path) -> tuple:
        """Return (num_samples, sample_rate)."""
        info = torchaudio.info(file_path)
        # Deal with backwards-incompatible signature change.
        # See https://github.com/pytorch/audio/issues/903 for more information.
        if type(info) is tuple:
            si, ei = info
            num_samples = si.length
            sample_rate = si.rate
        else:
            num_samples = info.num_frames
            sample_rate = info.sample_rate
        return num_samples, sample_rate

    def get_num_samples(self, file: AudioFile) -> int:
        """Number of samples (in target sample rate)

        :param file: audio file

        """

        self.is_valid(file)

        if isinstance(file, dict):

            # file = {"samples": torch.Tensor, "sample_rate": int, [ "channel": int ]}
            if "samples" in file:
                num_samples = file["samples"].shape[1]
                sample_rate = file["sample_rate"]

            # file = {"audio": str or Path, [ "channel": int ]}
            else:
                num_samples, sample_rate = self.get_audio_metadata(file["audio"])

        #  file = str or Path
        else:
            num_samples, sample_rate = self.get_audio_metadata(file)

        return math.ceil(num_samples * self.sample_rate / sample_rate)

    def __init__(self, sample_rate: int, mono: bool = True):
        super().__init__()
        self.sample_rate = sample_rate
        self.mono = mono

    def downmix_and_resample(self, samples: Tensor, sample_rate: int) -> Tensor:
        """Downmix and resample

        Parameters
        ----------
        samples : (channel, time) Tensor
            Samples.
        sample_rate : int
            Original sample rate.

        Returns
        -------
        samples : (channel, time) Tensor
            Remixed and resampled samples
        """

        # downmix to mono
        if self.mono and samples.shape[0] > 1:
            samples = samples.mean(dim=0, keepdim=True)

        # resample
        if self.sample_rate != sample_rate:
            samples = samples.numpy()
            if self.mono:
                # librosa expects mono audio to be of shape (n,), but we have (1, n).
                samples = librosa.core.resample(
                    samples[0], orig_sr=sample_rate, target_sr=self.sample_rate
                )[None]
            else:
                samples = librosa.core.resample(
                    samples.T, orig_sr=sample_rate, target_sr=self.sample_rate
                ).T

            samples = torch.tensor(samples)

        return samples

    def __call__(
        self, file: AudioFile, sample_offset: int = 0, num_samples: int = None
    ) -> Tensor:
        """

        Parameters
        ----------
        file : AudioFile
            Audio file.
        sample_offset : int, optional
            Start loading at this `sample_offset` sample. Defaults ot 0.
        num_samples : int, optional
            Load that many samples. Defaults to load up to the end of the file.

        Returns
        -------
        samples : (time, channel) torch.Tensor
            Samples

        """

        self.is_valid(file)

        original_samples = None

        if isinstance(file, dict):

            # file = {"samples": torch.Tensor, "sample_rate": int, [ "channel": int ]}
            if "samples" in file:
                original_samples = file["samples"]
                original_sample_rate = file["sample_rate"]
                original_total_num_samples = original_samples.shape[1]
                channel = file.get("channel", None)

            # file = {"audio": str or Path, ["channel": int ]}
            else:
                audio_path = str(file["audio"])
                (
                    original_total_num_samples,
                    original_sample_rate,
                ) = self.get_audio_metadata(audio_path)
                channel = file.get("channel", None)

        #  file = str or Path
        else:
            audio_path = str(file)
            original_total_num_samples, original_sample_rate = self.get_audio_metadata(
                audio_path
            )
            channel = None

        original_sample_offset = round(
            sample_offset * original_sample_rate / self.sample_rate
        )
        if num_samples is None:
            original_num_samples = original_total_num_samples - original_sample_offset
        else:
            original_num_samples = round(
                num_samples * original_sample_rate / self.sample_rate
            )

        if original_sample_offset + original_num_samples > original_total_num_samples:
            original_sample_offset = original_total_num_samples - original_num_samples
            # raise ValueError() # rounding error i guess

        if original_samples is None:
            try:
                original_data, _ = torchaudio.load(
                    audio_path,
                    frame_offset=original_sample_offset,
                    num_frames=original_num_samples,
                )
            except TypeError:
                raise Exception(
                    "It looks like you are using an unsupported version of torchaudio."
                    " If you have 0.6 or older, please upgrade to a newer version."
                )

        else:
            original_data = original_samples[
                :,
                original_sample_offset : original_sample_offset + original_num_samples,
            ]

        if channel is not None:
            original_data = original_data[channel - 1 : channel, :]

        result = self.downmix_and_resample(original_data, original_sample_rate)

        if num_samples is not None:
            # If there is an off-by-one error in the length (e.g. due to resampling), fix it.
            if result.shape[-1] > num_samples:
                result = result[:, :num_samples]
            elif result.shape[-1] < num_samples:
                diff = num_samples - result.shape[-1]
                result = torch.nn.functional.pad(result, (0, diff))

        return result



class AddBackgroundNoise(BaseWaveformTransform):
    """
    Add background noise to the input audio.
    """

    supported_modes = {"per_batch", "per_example", "per_channel"}

    # Note: This transform has only partial support for multichannel audio. Noises that are not
    # mono get mixed down to mono before they are added to all channels in the input.
    supports_multichannel = True
    requires_sample_rate = True

    supports_target = True
    requires_target = False

    def __init__(
        self,
        background_paths: Union[List[Path], List[str], Path, str],
        min_snr_in_db: float = 3.0,
        max_snr_in_db: float = 30.0,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: str = None,
        sample_rate: int = None,
        target_rate: int = None,
        output_type: Optional[str] = None,
    ):
        """

        :param background_paths: Either a path to a folder with audio files or a list of paths
            to audio files.
        :param min_snr_in_db: minimum SNR in dB.
        :param max_snr_in_db: maximum SNR in dB.
        :param mode:
        :param p:
        :param p_mode:
        :param sample_rate:
        """

        super().__init__(
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
            output_type=output_type,
        )

        # TODO: check that one can read audio files
        if isinstance(background_paths, ListConfig):
            background_paths = list(background_paths)
        self.background_paths = find_audio_files_in_paths(background_paths)

        if sample_rate is not None:
            self.audio = Audio(sample_rate=sample_rate, mono=True)

        if len(self.background_paths) == 0:
            raise EmptyPathException("There are no supported audio files found.")

        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        if self.min_snr_in_db > self.max_snr_in_db:
            raise ValueError("min_snr_in_db must not be greater than max_snr_in_db")

    def random_background(self, audio: Audio, target_num_samples: int) -> torch.Tensor:
        pieces = []

        # TODO: support repeat short samples instead of concatenating from different files

        missing_num_samples = target_num_samples
        while missing_num_samples > 0:
            background_path = random.choice(self.background_paths)
            background_num_samples = audio.get_num_samples(background_path)

            # If the background sample is longer than what we need, extract the exact amount
            if background_num_samples >= missing_num_samples:
                sample_offset = random.randint(
                    0, background_num_samples - missing_num_samples
                )
                background_samples = audio(
                    background_path,
                    sample_offset=sample_offset,
                    num_samples=missing_num_samples,
                )
                pieces.append(background_samples)
                # background_samples matches missing_num_samples, break out of while loop
                break

            background_samples = audio(background_path)
            pieces.append(background_samples)
            missing_num_samples -= background_num_samples

        # the inner call to rms_normalize ensures concatenated pieces share the same RMS (1)
        # the outer call to rms_normalize ensures that the resulting background has an RMS of 1
        # (this simplifies "apply_transform" logic)
        return audio.rms_normalize(
            torch.cat([audio.rms_normalize(piece) for piece in pieces], dim=1)
        )

    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[List[Tensor]] = None,
        target_rate: Optional[int] = None,
    ):
        """

        :params samples: (batch_size, num_channels, num_samples)
        """

        batch_size, _, num_samples = samples.shape

        # (batch_size, num_samples) RMS-normalized background noise
        audio = self.audio if hasattr(self, "audio") else Audio(sample_rate, mono=True)
        self.transform_parameters["background"] = torch.stack(
            [self.random_background(audio, num_samples) for _ in range(batch_size)]
        )

        # (batch_size, ) SNRs
        if self.min_snr_in_db == self.max_snr_in_db:
            self.transform_parameters["snr_in_db"] = torch.full(
                size=(batch_size,),
                fill_value=self.min_snr_in_db,
                dtype=torch.float32,
                device=samples.device,
            )
        else:
            snr_distribution = torch.distributions.Uniform(
                low=torch.tensor(
                    self.min_snr_in_db, dtype=torch.float32, device=samples.device
                ),
                high=torch.tensor(
                    self.max_snr_in_db, dtype=torch.float32, device=samples.device
                ),
                validate_args=True,
            )
            self.transform_parameters["snr_in_db"] = snr_distribution.sample(
                sample_shape=(batch_size,)
            )

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[List[Tensor]] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        batch_size, num_channels, num_samples = samples.shape

        # (batch_size, num_samples)
        background = self.transform_parameters["background"].to(samples.device)

        # (batch_size, num_channels)
        background_rms = calculate_rms(samples) / (
            10 ** (self.transform_parameters["snr_in_db"].unsqueeze(dim=-1) / 20)
        )

        return ObjectDict(
            samples=samples
            + background_rms.unsqueeze(-1)
            * background.view(batch_size, 1, num_samples).expand(-1, num_channels, -1),
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )


def _choose_original_labels(target, background_target, snr):
    return target


def _make_union_labels(target, background_target, snr):
    return torch.maximum(target, background_target)



class MultilabelMix(BaseWaveformTransform):
    supported_modes = {"per_example", "per_channel"}

    supports_multichannel = True
    requires_sample_rate = False

    supports_target = True
    requires_target = False

    def __init__(
        self,
        min_snr_in_db: float = 0.0,
        max_snr_in_db: float = 5.0,
        mix_target: str = "union",
        max_samples: int = 1,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: str = None,
        sample_rate: int = None,
        target_rate: int = None,
        output_type: Optional[str] = None,
    ):
        super().__init__(
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
            output_type=output_type,
        )
        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        if self.min_snr_in_db > self.max_snr_in_db:
            raise ValueError("min_snr_in_db must not be greater than max_snr_in_db")

        self.mix_target = mix_target
        if mix_target == "original":
            self._mix_target = _choose_original_labels

        elif mix_target == "union":
            self._mix_target = _make_union_labels
        else:
            raise ValueError("mix_target must be one of 'original' or 'union'.")

        self.max_samples = max_samples

    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[List[Tensor]]  = None,
        target_rate: Optional[int] = None,
    ):

        batch_size, num_channels, num_samples = samples.shape

        snr_distribution = torch.distributions.Uniform(
            low=torch.tensor(
                self.min_snr_in_db,
                dtype=torch.float32,
                device=samples.device,
            ),
            high=torch.tensor(
                self.max_snr_in_db,
                dtype=torch.float32,
                device=samples.device,
            ),
            validate_args=True,
        )  # sample uniformly from this distribution (low and high values)

        # randomize SNRs
        self.transform_parameters["snr_in_db"] = snr_distribution.sample(
            sample_shape=(batch_size,)
        )

        # randomize number of samples to mix for the entire batch
        num_mixes = torch.randint(
            1, self.max_samples + 1, (1,), device=samples.device
        ).item()

        self.transform_parameters["num_mixes"] = num_mixes

        # Generate random indices with the constraint
        sample_indices = torch.empty((batch_size, num_mixes), dtype=torch.long)

        for i in range(batch_size):
            possible_indices = list(range(batch_size))

            if len(possible_indices) > 1:  # avoid error if only one sample is chosen
                possible_indices.remove(
                    i
                )  # Remove the current index to avoid self-mixing
                sample_indices[i] = torch.tensor(
                    [
                        possible_indices[
                            torch.randint(0, len(possible_indices), (1,)).item()
                        ]
                    ]
                )
            else:
                # If there's only one sample, we can set the index to a default value or skip
                sample_indices[i] = torch.tensor([0])
        self.transform_parameters["sample_indices"] = sample_indices

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[List[Tensor]] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:

        snr = self.transform_parameters["snr_in_db"]
        num_mixes = self.transform_parameters["num_mixes"]
        sample_indices = self.transform_parameters["sample_indices"]

        mixed_samples = samples.clone()
        if targets is not None:
            mixed_targets = [t.clone() for t in targets]
        else:
            mixed_targets = None

        batch_size, _, waveform_length = mixed_samples.shape

        for i in range(num_mixes):
            current_indices = sample_indices[:, i]
            background_samples = Audio.rms_normalize(samples[current_indices])

            idx = torch.randint(
                0, waveform_length, (batch_size,), device=background_samples.device
            )
            arange = (
                torch.arange(waveform_length, device=background_samples.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            rolled_indices = (arange + idx.unsqueeze(1)) % waveform_length
            background_samples = background_samples.squeeze(1)[
                torch.arange(batch_size).unsqueeze(1), rolled_indices
            ].unsqueeze(1)
            background_rms = calculate_rms(mixed_samples) / (
                10 ** (snr.unsqueeze(dim=-1) / 20)
            )

            mixed_samples += background_rms.unsqueeze(-1) * background_samples

            if mixed_targets is not None:
                background_targets = targets[0][current_indices]
                for i in range(len(mixed_targets)):
                    mixed_targets[i] = self._mix_target(mixed_targets[i], background_targets, snr)

        return ObjectDict(
            samples=mixed_samples,
            # mix_indices=sample_indices,
            sample_rate=sample_rate,
            targets=mixed_targets,
            target_rate=target_rate,
        )


class AddColoredNoise(BaseWaveformTransform):
    """
    Add colored noises to the input audio.
    """

    supported_modes = {"per_batch", "per_example", "per_channel"}

    supports_multichannel = True
    requires_sample_rate = True

    supports_target = True
    requires_target = False

    def __init__(
        self,
        min_snr_in_db: float = 3.0,
        max_snr_in_db: float = 30.0,
        min_f_decay: float = -2.0,
        max_f_decay: float = 2.0,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: str = None,
        sample_rate: int = None,
        target_rate: int = None,
        output_type: Optional[str] = None,
    ):
        """
        :param min_snr_in_db: minimum SNR in dB.
        :param max_snr_in_db: maximum SNR in dB.
        :param min_f_decay:
            defines the minimum frequency power decay (1/f**f_decay).
            Typical values are "white noise" (f_decay=0), "pink noise" (f_decay=1),
            "brown noise" (f_decay=2), "blue noise (f_decay=-1)" and "violet noise"
            (f_decay=-2)
        :param max_f_decay:
            defines the maximum power decay (1/f**f_decay) for non-white noises.
        :param mode:
        :param p:
        :param p_mode:
        :param sample_rate:
        :param target_rate:
        """

        super().__init__(
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
            output_type=output_type,
        )

        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        if self.min_snr_in_db > self.max_snr_in_db:
            raise ValueError("min_snr_in_db must not be greater than max_snr_in_db")

        self.min_f_decay = min_f_decay
        self.max_f_decay = max_f_decay
        if self.min_f_decay > self.max_f_decay:
            raise ValueError("min_f_decay must not be greater than max_f_decay")

    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        """
        :params selected_samples: (batch_size, num_channels, num_samples)
        """
        batch_size, _, num_samples = samples.shape

        # (batch_size, ) SNRs
        for param, mini, maxi in [
            ("snr_in_db", self.min_snr_in_db, self.max_snr_in_db),
            ("f_decay", self.min_f_decay, self.max_f_decay),
        ]:
            dist = torch.distributions.Uniform(
                low=torch.tensor(mini, dtype=torch.float32, device=samples.device),
                high=torch.tensor(maxi, dtype=torch.float32, device=samples.device),
                validate_args=True,
            )
            self.transform_parameters[param] = dist.sample(sample_shape=(batch_size,))

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:

        batch_size, num_channels, num_samples = samples.shape

        # (batch_size, num_samples)
        noise = torch.stack(
            [
                _gen_noise(
                    self.transform_parameters["f_decay"][i],
                    num_samples,
                    sample_rate,
                    samples.device,
                )
                for i in range(batch_size)
            ]
        )

        # (batch_size, num_channels)
        noise_rms = calculate_rms(samples) / (
            10 ** (self.transform_parameters["snr_in_db"].unsqueeze(dim=-1) / 20)
        )

        return ObjectDict(
            samples=samples
            + noise_rms.unsqueeze(-1)
            * noise.view(batch_size, 1, num_samples).expand(-1, num_channels, -1),
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )


class Gain(BaseWaveformTransform):
    """
    Multiply the audio by a random amplitude factor to reduce or increase the volume. This
    technique can help a model become somewhat invariant to the overall gain of the input audio.

    Warning: This transform can return samples outside the [-1, 1] range, which may lead to
    clipping or wrap distortion, depending on what you do with the audio in a later stage.
    See also https://en.wikipedia.org/wiki/Clipping_(audio)#Digital_clipping
    """

    supported_modes = {"per_batch", "per_example", "per_channel"}

    supports_multichannel = True
    requires_sample_rate = False

    supports_target = True
    requires_target = False

    def __init__(
        self,
        min_gain_in_db: float = -18.0,
        max_gain_in_db: float = 6.0,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: Optional[str] = None,
        sample_rate: Optional[int] = None,
        target_rate: Optional[int] = None,
        output_type: Optional[str] = None,
    ):
        super().__init__(
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
            output_type=output_type,
        )
        self.min_gain_in_db = min_gain_in_db
        self.max_gain_in_db = max_gain_in_db
        if self.min_gain_in_db >= self.max_gain_in_db:
            raise ValueError("max_gain_in_db must be higher than min_gain_in_db")

    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[List[Tensor]] = None,
        target_rate: Optional[int] = None,
    ):
        distribution = torch.distributions.Uniform(
            low=torch.tensor(
                self.min_gain_in_db, dtype=torch.float32, device=samples.device
            ),
            high=torch.tensor(
                self.max_gain_in_db, dtype=torch.float32, device=samples.device
            ),
            validate_args=True,
        )
        selected_batch_size = samples.size(0)
        self.transform_parameters["gain_factors"] = (
            convert_decibels_to_amplitude_ratio(
                distribution.sample(sample_shape=(selected_batch_size,))
            )
            .unsqueeze(1)
            .unsqueeze(1)
        )

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[List[Tensor]] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:

        return ObjectDict(
            samples=samples * self.transform_parameters["gain_factors"],
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )



class FilterAugment(torch.nn.Module):
    def __init__(self, p=1.0, db_range=(-6, 6), n_band=(3, 6), min_bw=6, filter_type="linear"):
        """
        Filter augmentation module for spectral features.

        Args:
            db_range (tuple): Range of gain in decibels.
            n_band (tuple): Range of number of frequency bands.
            min_bw (int): Minimum bandwidth per band.
            filter_type (str or float): "step", "linear", or a float as probability threshold.
        """
        super(FilterAugment, self).__init__()
        self.p = p
        self.db_range = db_range
        self.n_band = n_band
        self.min_bw = min_bw
        self.filter_type = filter_type

    def forward(self, features):
        """
        Apply frequency domain augmentation.

        Args:
            features (Tensor): Shape (B, F, T) - Batch, Frequency bins, Time

        Returns:
            Tensor: Filter-augmented features of shape (B, F, T)
        """
        if random.random() < self.p:
            features = features.squeeze(1)
            db_range = self.db_range
            n_band = list(self.n_band)
            min_bw = self.min_bw
            filter_type = self.filter_type

            batch_size, n_freq_bin, _ = features.shape
            n_freq_band = torch.randint(low=n_band[0], high=n_band[1], size=(1,)).item()

            if n_freq_band > 1:
                while n_freq_bin - n_freq_band * min_bw + 1 < 0:
                    min_bw -= 1

                band_bndry_freqs = torch.sort(torch.randint(0, n_freq_bin - n_freq_band * min_bw + 1,
                                                            (n_freq_band - 1,)))[0] + \
                                   torch.arange(1, n_freq_band) * min_bw
                band_bndry_freqs = torch.cat((torch.tensor([0]), band_bndry_freqs, torch.tensor([n_freq_bin]))).to(features.device)

                if filter_type == "step":
                    band_factors = torch.rand((batch_size, n_freq_band), device=features.device) \
                                   * (db_range[1] - db_range[0]) + db_range[0]
                    band_factors = 10 ** (band_factors / 20)

                    freq_filt = torch.ones((batch_size, n_freq_bin, 1), device=features.device)
                    for i in range(n_freq_band):
                        freq_filt[:, band_bndry_freqs[i]:band_bndry_freqs[i + 1], :] = \
                            band_factors[:, i].unsqueeze(-1).unsqueeze(-1)

                elif filter_type == "linear":
                    band_factors = torch.rand((batch_size, n_freq_band + 1), device=features.device) \
                                   * (db_range[1] - db_range[0]) + db_range[0]

                    freq_filt = torch.ones((batch_size, n_freq_bin, 1), device=features.device)
                    for i in range(n_freq_band):
                        for j in range(batch_size):
                            interp = torch.linspace(band_factors[j, i], band_factors[j, i + 1],
                                                    band_bndry_freqs[i + 1] - band_bndry_freqs[i], device=features.device)
                            freq_filt[j, band_bndry_freqs[i]:band_bndry_freqs[i + 1], 0] = interp

                    freq_filt = 10 ** (freq_filt / 20)

                return (features * freq_filt).unsqueeze(1)

            else:
                return features.unsqueeze(1)
        else:
            return features
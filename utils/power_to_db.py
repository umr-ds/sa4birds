import torch
from torch import nn
import warnings

class PowerToDB(nn.Module):
    """
    Convert a power (or magnitude) spectrogram to decibel (dB) scale.

    This module implements a PyTorch version of the common
    power-to-decibel transformation:

        10 * log10(S / ref)

    where:
        - S is a power spectrogram (or magnitude input),
        - ref is a reference value,
        - amin is a minimum threshold for numerical stability,
        - top_db optionally limits the dynamic range.

    Args:
        ref (float or callable, optional):
            Reference value for scaling.
            - If float: used directly.
            - If callable: computed dynamically from the input.
            Default: 1.0.
        amin (float, optional):
            Minimum threshold for input magnitude to avoid log(0).
            Must be strictly positive. Default: 1e-10.
        top_db (float or None, optional):
            If provided, clips the output to a maximum dynamic range
            of `top_db` below the peak value. Must be non-negative.
            Default: 80.0.

    Notes:
        - If complex input is provided, magnitude is taken and a
          warning is issued.
        - The output is in decibels (dB).
        - This is typically used after computing a power spectrogram.
    """
    def __init__(self, ref=1.0, amin=1e-10, top_db=80.0):
        super(PowerToDB, self).__init__()
        # Initialize parameters
        self.ref = ref
        self.amin = amin
        self.top_db = top_db

    def forward(self, S):
        """
        Apply power-to-decibel conversion.

        Args:
           S (Tensor or array-like):
               Input power or magnitude spectrogram.
               Can be real or complex.

        Returns:
           torch.Tensor:
               Spectrogram converted to decibel (dB) scale.
        """
        # Convert S to a PyTorch tensor if it is not already
        S = torch.as_tensor(S, dtype=torch.float32)

        if self.amin <= 0:
            raise ValueError("amin must be strictly positive")

        if torch.is_complex(S):
            warnings.warn(
                "power_to_db was called on complex input so phase "
                "information will be discarded. To suppress this warning, "
                "call power_to_db(S.abs()**2) instead.",
                stacklevel=2,
            )
            magnitude = S.abs()
        else:
            magnitude = S

        # Check if ref is a callable function or a scalar
        if callable(self.ref):
            ref_value = self.ref(magnitude)
        else:
            ref_value = torch.abs(torch.tensor(self.ref, dtype=S.dtype))

        # Compute the log spectrogram
        log_spec = 10.0 * torch.log10(
            torch.maximum(magnitude, torch.tensor(self.amin, device=magnitude.device))
        )
        log_spec -= 10.0 * torch.log10(
            torch.maximum(ref_value, torch.tensor(self.amin, device=magnitude.device))
        )

        # Optional dynamic range compression
        if self.top_db is not None:
            if self.top_db < 0:
                raise ValueError("top_db must be non-negative")

            # Clip values below (max - top_db)
            log_spec = torch.maximum(log_spec, log_spec.max() - self.top_db)

        return log_spec
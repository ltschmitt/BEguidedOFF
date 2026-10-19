"""BEguidedOFF — off-target base-editing search.

Two modes share one public factory:

* ``mode="extensive"`` (default) -- GPU brute-force search; requires CUDA and
  ``torch``/``numba``.
* ``mode="fast"`` -- CPU FM-index search; the lightweight filter used by the web
  application.  Requires only numpy/pandas/pysam plus the FM-index binding.

Importing this package never imports ``torch`` or ``numba``; the mode-specific
implementation is imported lazily when the factory is called.
"""
from beguidedoff.types import GuideInput
from beguidedoff import genome_converter

__all__ = ["beguidedoff", "beguidedofffast", "GuideInput", "genome_converter"]


def beguidedoff(*args, mode: str = "extensive", **kwargs):
    """Create an off-target finder.

    Args:
        mode: ``"extensive"`` (GPU brute force, default) or ``"fast"``
            (CPU FM-index).
        *args: Positional arguments forwarded to the selected implementation.
        **kwargs: Keyword arguments forwarded to the selected implementation.

    Returns:
        Either :class:`beguidedoff.finder.beguidedoff` (extensive) or
        :class:`beguidedoff.fast.finder.beguidedofffast` (fast).

    Raises:
        ValueError: If *mode* is not recognised.
    """
    mode = str(mode).lower()
    if mode == "extensive":
        from beguidedoff.finder import beguidedoff as _extensive

        return _extensive(*args, **kwargs)
    if mode == "fast":
        from beguidedoff.fast.finder import beguidedofffast as _fast

        return _fast(*args, **kwargs)
    raise ValueError(f"Unknown mode {mode!r}; expected 'extensive' or 'fast'.")


def beguidedofffast(*args, **kwargs):
    """Create a fast CPU finder.

    Shorthand for ``beguidedoff(mode="fast", ...)``.
    """
    from beguidedoff.fast.finder import beguidedofffast as _fast

    return _fast(*args, **kwargs)

"""Image processing shared by every imaging instrument and workflow.

Segmentation back-ends and the synthetic-microscopy renderer live in the core
because more than one instrument needs them: ``cell_counter`` segments a
field of view, ``cell_tracker`` segments every frame of a movie, and the
sorter-video workflow renders and segments a sorter's outlet. Keeping them
here is what lets the instruments stay independent of one another
(ARCHITECTURE.md: instruments never import each other).
"""

from biosim_lab.core.imaging.segmentation import (
    SegmentationResult,
    available_backends,
    segment,
)
from biosim_lab.core.imaging.synthetic import (
    SyntheticImageSpec,
    synthetic_field,
    synthetic_movie,
)

__all__ = [
    "SegmentationResult",
    "available_backends",
    "segment",
    "SyntheticImageSpec",
    "synthetic_field",
    "synthetic_movie",
]

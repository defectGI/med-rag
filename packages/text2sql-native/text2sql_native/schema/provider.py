"""SchemaProvider.

This is the seam that keeps the pipeline schema-agnostic. The default loads
from a file (see :mod:`.file_provider`). A caller can inject any subclass
instead, for example one that reads a live database, as long as it returns a
:class:`~text2sql.schema.models.Schema`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .models import Schema


class SchemaProvider(ABC):
    """Gives schema metadata to the pipeline.

    Subclasses return a full :class:`Schema`. The pipeline does not care where
    it comes from.
    """

    @abstractmethod
    def load(self) -> Schema:
        """Load and return the schema.

        Raises:
            SchemaError: If the schema cannot be built or is invalid.
        """
        raise NotImplementedError

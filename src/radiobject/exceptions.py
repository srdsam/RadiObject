"""Custom exception hierarchy for RadiObject."""


class RadiObjectError(Exception):
    """Base exception for all RadiObject errors."""


class ShapeError(RadiObjectError, ValueError):
    """Volume or collection shape mismatch."""


class AlignmentError(RadiObjectError, ValueError):
    """Index or subject alignment failure."""


class StorageError(RadiObjectError, RuntimeError):
    """TileDB storage or I/O failure."""


class ConfigurationError(RadiObjectError, ValueError):
    """Invalid configuration or credential error."""


class SchemaError(RadiObjectError, ValueError):
    """Column, schema, or metadata conflict."""


class IngestError(RadiObjectError, ValueError):
    """Image discovery or format detection failure."""


class ViewError(RadiObjectError, TypeError):
    """Attempted mutation on an immutable view."""

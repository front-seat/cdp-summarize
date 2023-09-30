import typing as t

from cdp_data.utils import connect_to_infrastructure
from gcsfs import GCSFileSystem


class CDPConnection:
    """
    Represents an active connection to CDP infrastructure.

    This is a bit awkward because FireO seems to assume one global connection:
    if you call `connect()` multiple times, you'll need to stop using your old
    connection instances.
    """

    # TODO REVISIT this. It's awkward.

    infrastructure_slug: str
    file_system: GCSFileSystem

    def __init__(self, infrastructure_slug: str, file_system: GCSFileSystem):
        self.infrastructure_slug = infrastructure_slug
        self.file_system = file_system

    @classmethod
    def connect(cls, infrastructure_slug: str) -> t.Self:
        """Connect to the given CDP infrastructure, returning a valid CDPConnection."""
        return cls(infrastructure_slug, connect_to_infrastructure(infrastructure_slug))

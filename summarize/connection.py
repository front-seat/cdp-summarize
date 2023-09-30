import typing as t
from inspect import getmembers

from cdp_data import CDPInstances
from cdp_data.utils import connect_to_infrastructure
from gcsfs import GCSFileSystem


def _get_available_cdp_instances() -> dict[str, str]:
    """
    Return a dictionary mapping a friendly name for the CLI
    (like "Seattle") to a CDP infrastructure slug (like
    "cdp-seattle-21723dcf").

    Try not to duplicate the CDPInstances class; rather, use it.
    """
    return {
        name: value
        for name, value in getmembers(CDPInstances)
        if isinstance(value, str) and not name.startswith("__")
    }


INSTANCES = _get_available_cdp_instances()
INSTANCE_NAMES = list(INSTANCES.keys())


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
    def for_slug(cls, infrastructure_slug: str) -> t.Self:
        """Connect to the given CDP slug, returning a valid CDPConnection."""
        return cls(infrastructure_slug, connect_to_infrastructure(infrastructure_slug))

    @classmethod
    def for_name(cls, infrastructure_name: str) -> t.Self:
        """Connect to the given CDP name, returning a valid CDPConnection."""
        return cls.for_slug(INSTANCES[infrastructure_name])

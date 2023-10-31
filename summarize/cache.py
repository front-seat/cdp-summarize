import abc
import json
import logging
import pathlib
import typing as t

logger = logging.getLogger(__name__)


class BaseCache(abc.ABC):
    """
    An abstract cache base class.

    Here's a dirty secret about this "cache": there's no ability to
    set an item expiration time. This is a deliberate choice to keep the
    code simple.
    """

    @abc.abstractmethod
    def __getitem__(self, key: str) -> dict:
        """Return the value associated with the given key."""
        ...

    @abc.abstractmethod
    def __setitem__(self, key: str, value: dict) -> None:
        """Associate the given value with the given key."""
        ...

    @abc.abstractmethod
    def __contains__(self, key: str) -> bool:
        """Return True if the given key is in the cache."""
        ...

    @abc.abstractmethod
    def __delitem__(self, key: str) -> None:
        """Remove the given key from the cache."""
        ...

    @abc.abstractmethod
    def __iter__(self) -> t.Iterator[str]:
        """Return an iterator over the keys in the cache."""
        ...

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the number of items in the cache."""
        ...

    @abc.abstractmethod
    def clear(self) -> None:
        """Remove all items from the cache."""
        ...

    @abc.abstractmethod
    def keys(self) -> t.Iterator[str]:
        """Return an iterator over the keys in the cache."""
        ...

    @abc.abstractmethod
    def values(self) -> t.Iterator[dict]:
        """Return an iterator over the values in the cache."""
        ...

    @abc.abstractmethod
    def items(self) -> t.Iterator[tuple[str, dict]]:
        """Return an iterator over the (key, value) pairs in the cache."""
        ...

    @abc.abstractmethod
    def get(self, key: str, default: dict | None = None) -> dict | None:
        """Return the value associated with the given key, or default."""
        ...

    @abc.abstractmethod
    def set(self, key: str, value: dict) -> None:
        """Associate the given value with the given key."""
        ...


class InMemoryCache(BaseCache):
    """A simple cache variant that stores content in a dict."""

    _data: dict[str, dict]

    def __init__(self) -> None:
        self._data = {}

    def __getitem__(self, key: str) -> dict:
        """Return the value associated with the given key."""
        return self._data[key]

    def __setitem__(self, key: str, value: dict) -> None:
        """Associate the given value with the given key."""
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        """Return True if the given key is in the cache."""
        return key in self._data

    def __delitem__(self, key: str) -> None:
        """Remove the given key from the cache."""
        del self._data[key]

    def __iter__(self) -> t.Iterator[str]:
        """Return an iterator over the keys in the cache."""
        return iter(self._data)

    def __len__(self) -> int:
        """Return the number of items in the cache."""
        return len(self._data)

    def clear(self) -> None:
        """Remove all items from the cache."""
        self._data.clear()

    def keys(self) -> t.Iterator[str]:
        """Return an iterator over the keys in the cache."""
        return iter(self._data)

    def values(self) -> t.Iterator[dict]:
        """Return an iterator over the values in the cache."""
        return iter(self._data.values())

    def items(self) -> t.Iterator[tuple[str, dict]]:
        """Return an iterator over the (key, value) pairs in the cache."""
        return iter(self._data.items())

    def get(self, key: str, default: dict | None = None) -> dict | None:
        """Return the value associated with the given key, or default."""
        return self._data.get(key, default)

    def set(self, key: str, value: dict) -> None:
        """Associate the given value with the given key."""
        self._data[key] = value


class FileSystemCache(BaseCache):
    """
    A cache that looks like a dictionary to the outside world, but stores
    its data in the filesystem so that it can persist across multiple
    invocations of the program.
    """

    _cache_dir: pathlib.Path

    def __init__(self, cache_dir: pathlib.Path):
        self._cache_dir = cache_dir.resolve()
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def __getitem__(self, key: str) -> dict:
        """Return the value associated with the given key."""
        path = self._cache_dir / f"{self.safe_key(key)}.txt"
        if not path.exists():
            raise KeyError(key)
        if not path.is_file():
            raise KeyError(key)
        text = path.read_text()
        return json.loads(text)

    def __setitem__(self, key: str, value: dict) -> None:
        """Associate the given value with the given key."""
        path = self._cache_dir / f"{self.safe_key(key)}.txt"
        text = json.dumps(value)
        path.write_text(text)

    def __contains__(self, key: str) -> bool:
        """Return True if the given key is in the cache."""
        path = self._cache_dir / f"{self.safe_key(key)}.txt"
        return path.exists()

    def __delitem__(self, key: str) -> None:
        """Remove the given key from the cache."""
        path = self._cache_dir / f"{self.safe_key(key)}.txt"
        path.unlink()

    def __iter__(self) -> t.Iterator[str]:
        """Return an iterator over the keys in the cache."""
        return (path.stem for path in self._cache_dir.iterdir())

    def __len__(self) -> int:
        """Return the number of items in the cache."""
        return len(list(self._cache_dir.iterdir()))

    def clear(self) -> None:
        """Remove all items from the cache."""
        for path in self._cache_dir.iterdir():
            path.unlink()

    def keys(self) -> t.Iterator[str]:
        """Return an iterator over the keys in the cache."""
        return iter(self)

    def values(self) -> t.Iterator[dict]:
        """Return an iterator over the values in the cache."""
        for path in self._cache_dir.iterdir():
            text = path.read_text()
            yield json.loads(text)

    def items(self) -> t.Iterator[tuple[str, dict]]:
        """Return an iterator over the (key, value) pairs in the cache."""
        for path in self._cache_dir.iterdir():
            text = path.read_text()
            yield path.stem, json.loads(text)

    def get(self, key: str, default: dict | None = None) -> dict | None:
        """Return the value associated with the given key, or default."""
        try:
            return self[key]
        except KeyError:
            return default

    def set(self, key: str, value: dict) -> None:
        """Associate the given value with the given key."""
        self[key] = value

    def safe_key(self, unsafe: str) -> str:
        """
        Return a safe key for the given string, suitable for use in a
        filesystem cache.
        """
        safe = unsafe.replace("/", "_")
        return safe

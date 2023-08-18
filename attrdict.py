"""UserDict wrapper class to enable accessing keys as attributes."""
from pathlib import Path
from typing import Any, Mapping, Optional, Union, Dict

# Common type hints
PathName = Union[str, Path]


class AttrDict(dict):
    """
    Dictionary wrapper class that:
    1) Enables accessing keys as properties.
    2) Treats None values as absent keys. Setting a key to None effectively unsets it (though the value may exist deep inside).
    3) Getting a non-existent key returns None instead of raising KeyError.
    4) Converts nested dictionaries to AttrDict objects. Only nested dictionaries are converted, not lists or tuples.
    5) Override _new() to change the type of nested dictionaries.
    6) Setting a key with dot separated segments (e.g. attr_dict.a.b.c = val) creates nested AttrDict objects.
    """

    def __init__(self, initialdata: Optional[Mapping] = None, **kwargs: Any):
        """Initializes a new AttrDict object.
        Args:
            initialdata: Mapping object to initialize the AttrDict object with. kwargs must be none if this is provided.
            kwargs: key-value pairs to initialize the AttrDict object with. initialdata must be none if this is provided.
        """
        if initialdata is not None:
            assert isinstance(initialdata, (Mapping, Dict)
                              ), f'initialdata should be a dictionary, not {type(initialdata)}'
            assert not kwargs
        elif kwargs:
            initialdata = kwargs
        if initialdata:
            super().__init__(self._recurse_init(initialdata))
        else:
            super().__init__()

    def update(self, other: Optional[Mapping] = None, **kwargs) -> None:
        """Same as dict.update except that it converts nested dictionaries to AttrDict objects."""
        if other is not None:
            assert not kwargs, 'Cannot pass both other and kwargs'
            _d = other
        else:
            _d = kwargs
        super().update(self._recurse_init(_d))

    def updated(self, other: Optional[Mapping] = None, **kwargs) -> 'AttrDict':
        """Same as update except that it returns self."""
        self.update(other, **kwargs)
        return self

    @classmethod
    def _new(cls, val: Optional[Mapping] = None) -> 'AttrDict':
        """Create and return a new object of this class.
        When you inherit, override this method if you want nested dictionaries to be of a different class.
        """
        return AttrDict(val)

    def _recurse_init(self, d: Mapping) -> 'AttrDict':
        """Convert all nested dict values to self._new() objects.

        Useful when deserializing a Json file.
        Also results in a deep copy of dictionary values.

        Returns:
            A AttrDict object of type self._attrdict_class()
        """
        new_d = self._new()
        for k, v in d.items():
            if v is None:  # None values are never set
                pass
            elif isinstance(v, Mapping):  # and not isinstance(v, self._attrdict_class())
                new_d[k] = self._new(v)
            else:
                new_d[k] = v
        return new_d

    def _get_val_(self, key: Any) -> Any:
        """Innermost function for getting value."""
        keys = key.split('.')
        if len(keys) > 1:
            sub_params = dict.get(self, keys[0])
            if not isinstance(sub_params, Mapping):
                return None
            else:
                return sub_params['.'.join(keys[1:])]
        else:
            val = dict.get(self, key)
            return val

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            val = self._get_val_(key)
            if val is None:
                return default
            return val
        except KeyError:
            return default

    def setdefault(self, k: str, default: Any) -> Any:
        """
        Like dict.setdefault except tha an existing set value of 'None' is considered unset in which case, the key's
        value will be set.
        """
        val = self.get(k)
        if val is None:
            self._set_val_(k, default)
            return self.get(k)
        else:
            return val

    def _set_val_(self, key: Any, val: Any) -> None:
        """
        Innermost function for setting value.
        Note: Setting a key to None effectively unsets it (though the key exists in the dictionary).
        """
        keys = key.split('.')
        if len(keys) > 1:
            my_type = type(self._new())
            p = dict.get(self, keys[0])
            if p is None:
                p = self._new()
                dict.__setitem__(self, keys[0], p)
            assert isinstance(p, my_type), f'Cannot assign to key {keys[0]} of type {type(p)}. Need type {my_type}.'
            p._set_val_('.'.join(keys[1:]), val)
        else:
            if isinstance(val, Mapping):
                val = self._recurse_init(val)
            dict.__setitem__(self, key, val)

    @staticmethod
    def _do_intercept(key: str) -> bool:
        """Return True if we manage this key, False if it's delegated to dict"""
        return key.startswith('__') and key.endswith('__')

    def __getattr__(self, key: str) -> Any:
        if self._do_intercept(key):
            return dict.__getattribute__(self, key)
        else:
            return self._get_val_(key)

    def __setattr__(self, key: Any, val: Any) -> None:
        if self._do_intercept(key):
            return dict.__setattr__(self, key, val)
        else:
            return self._set_val_(key, val)

    def __getitem__(self, key: Any) -> Any:
        if self._do_intercept(key):
            return dict.__getattribute__(self, key)
        else:
            return self._get_val_(key)

    def __setitem__(self, key: Any, val: Any) -> None:
        if self._do_intercept(key):
            return dict.__setattr__(self, key, val)
        else:
            return self._set_val_(key, val)


Params = AttrDict
NDict = AttrDict

"""Utility methods for pandagrader modules."""
import pprint
from pathlib import Path
from typing import Any, Mapping, Optional, Union, Dict, List, Tuple
import json
import gzip
import numpy as np
from ruamel.yaml import YAML

# Common type hints
PathName = Union[str, Path]


def save_params_to_json_file(params_dict: Any, filename: PathName, sort_keys: bool = False) -> None:
    """Save python data structure (mostly config) to json filename.

    params_dict is the data-structure to save, but it doesn't have to be a dictionary. It can be anything that
    is JSON serializable. This parameter should've been named 'data' instead.
    """
    with open(filename, 'w') as file_pointer:
        json.dump(params_dict, file_pointer, sort_keys=sort_keys, indent=4)
        file_pointer.write("\n")  # Add newline cause Py JSON does not


def load_params_from_json_file(filename: PathName) -> Any:
    """Load python data structure (mostly config) from json filename."""
    with open(filename, 'r') as file_pointer:
        data = json.load(file_pointer)
    return data


def save_params_to_yaml_file(data: Any, filename: PathName) -> None:
    """Save python data structure (mostly config) to yaml filename."""
    yaml = YAML(typ="safe", pure=True)
    with open(filename, 'wt') as file_pointer:
        yaml.dump(data, file_pointer)


def load_params_from_yaml_file(filename: PathName) -> Any:
    """Load python data structure (mostly config) from yaml filename."""
    yaml = YAML(typ="safe", pure=True)
    with open(filename, 'r') as file_pointer:
        data = yaml.load(file_pointer)
    return data


def load_params_from_text_file(filename: PathName) -> Any:
    """Load python data struct from yaml or json file depending on filename extension."""
    filename = Path(filename)
    if filename.suffix == '.json':
        return load_params_from_json_file(filename)
    else:
        return load_params_from_yaml_file(filename)


def save_params_to_text_file(data: Any, filename: PathName) -> None:
    """Save python data structure (mostly config) to a yaml or json file, depending on the filename extension."""
    filename = Path(filename)
    if filename.suffix == '.json':
        return save_params_to_json_file(data, filename)
    else:
        return save_params_to_yaml_file(data, filename)


def load_params_from_text_gz(filename: PathName) -> Any:
    """Load python data structure from a .json.gz or .yml.gz file."""
    filename = Path(filename)
    if filename.suffix == '.gz':
        with gzip.open(filename, 'rt', encoding='utf-8') as fp:
            if filename.suffixes[-2] == '.json':
                return json.load(fp)
            else:
                return YAML(typ="safe", pure=True).load(fp)
    else:
        return load_params_from_text_file(filename)


def save_params_to_text_gz(data: Any, filename: PathName) -> None:
    """Save python data structure to a .yml.gz, .yaml.gz or .json.gz file."""
    filename = Path(filename)
    if filename.suffix == '.gz':
        with gzip.open(filename, 'wt') as fp:
            filename = Path(filename.stem)
            if filename.suffix == '.json':
                json.dump(data, fp, indent=4)
                fp.write("\n")  # Add newline cause Py JSON does not
            else:
                yaml = YAML(typ="safe", pure=True)
                yaml.dump(data, fp)
    else:
        save_params_to_text_file(data, filename)


class Params(dict):
    """Convenience class that allows accessing all dictionary keys as properties (like Javascript objects).

    i.e. c['x'] == c.x. Allows both getting and setting of said properties.
    Also allows Javascript-like 'sealing' and 'freezing' of the object.
    The 'seal' feature comes in handy when you want to merge configs from different places but ensure
    that there's no overlap between them. For e.g. we have decided to keep command-line args distinct from
    config file parameters but want to merge them together in the code. This can be achieved as:
    Params(cli_params_dict).seal().updated(config_file_params)
    """

    def __init__(self, d: Optional[Mapping] = None, **kwargs: Any):
        """Convenience class that allows accessing all dictionary keys as properties (like Javascript objects).

        Args:
            d: Optional dictionary-like object to initialize the params. The initializer will descend all nested
                dictionary-like values and convert them all to objects of this class using self._new. Can be
                a dict, Params, HyperParams or any type that looks like a python Mapping.
            kwargs: kwargs that will be converted to key,value pairs. Can only be supplied if d is None.
        """
        if d is not None:
            assert isinstance(d, (Mapping, Dict)), f'Argument type is {type(d)}, should be a dictionary or subclass'
            assert not kwargs
        elif kwargs:
            d = kwargs
        if d:
            dict.__init__(self, self._dict2params(d))
        else:
            dict.__init__(self)
        dict.__setattr__(self, '_isFrozen', False)
        dict.__setattr__(self, '_isSealed', False)

    def update(self, dct: Optional[Mapping]) -> None:  # type: ignore
        """Update self.

        Same as dict.update except that it converts nested dicts into Params.
        """
        if dct is not None:
            dict.update(self, self._dict2params(dct))

    def updated(self, dct: Optional[Mapping]) -> 'Params':
        """Update the dictionary same as update except that it returns itself, thus enabling call chaining."""
        self.update(dct)
        return self

    @classmethod
    def _params_class(cls) -> Any:
        """
        Return this object's class. Override this method when you inherit this class.
        """
        return Params

    @classmethod
    def _new(cls, val: Optional[Mapping] = None) -> 'Params':
        """Create and return a new object of this class.

        When you inherit, override this method to return a new object of your class.
        """
        return Params(val)

    def _dict2params(self, d: Mapping) -> Any:  # can't annotate return type with self class
        """Convert all nested dict values to self._new() objects.

        Useful when deserializing a Json file.
        Also results in a deep copy of dictionary values.

        Returns:
            A Params object of type self._params_class()
        """
        new_d = self._new()
        for k, v in d.items():
            # if v is None:  # None values are never set
            #     pass
            if isinstance(v, Mapping):  # and not isinstance(v, self._params_class())
                new_d[k] = self._new(v)
            else:
                new_d[k] = v
        return new_d

    def to_dict(self) -> Dict:
        """
        Convert to a Dict, including nested Params.
        Also results in a deep copy of dictionary and list values.
        Returns:
            A dict object
        """
        new_d = dict()
        for k, v in self.items():
            if isinstance(v, Params):
                new_d[k] = v.to_dict()
            else:
                new_d[k] = v
        return new_d

    @classmethod
    def read_file(cls, path: PathName) -> 'Params':
        """Read parameters from a yaml or json file and return a new object."""
        return cls._new(load_params_from_text_file(path))

    def to_file(self, path: PathName) -> None:
        """Save params to a yaml or json file, based on filename."""
        save_params_to_text_file(self, path)

    def to_data_dict(self) -> dict:
        """Convert to a data-only dictionary. Raises error in case of non-data contents."""
        return json.loads(json.dumps(self))

    def print(self) -> None:
        """Pretty print self"""
        pprint.pprint(self)

    def _get_val_(self, key: Any, raise_if_not_exists: bool = False) -> Any:
        """Innermost function for getting value."""
        keys = key.split('.')
        if len(keys) > 1:
            sub_params = dict.get(self, keys[0])
            if not isinstance(sub_params, Mapping):
                if raise_if_not_exists:  # pylint: disable=no-else-raise
                    raise KeyError(f'key {keys[0]} does not exist')
                else:
                    return None
            else:
                return sub_params['.'.join(keys[1:])]
        else:
            # return dict.__getitem__(self, key)
            val = dict.get(self, key)
            if raise_if_not_exists and val is None:
                raise KeyError(f'key {key} does not exist')
            return val

    def get(self, key: Any, default: Any = None) -> Any:
        """Get the key's value if exists and non-None else return default."""
        try:
            val = self._get_val_(key)
            if val is None:
                return default
            return val
        except KeyError:
            return default

    def setdefault(self, k: str, default: Any) -> Any:  # type: ignore
        """
        Like dict.setdefault except:
        1) An existing set value of 'None' is considered unset. In such case it will set
           the value to default.
        Note: Setting a value to None effectively unsets it (though the key exists in the dictionary).

        Returns
        -------
            The set value
        """
        val = self.get(k)
        if val is None:
            self._set_val_(k, default)
            return self.get(k)
        else:
            return val

    def set_leaves(self,  # pylint: disable=dangerous-default-value
                   key: Optional[str] = None,
                   val: Any = {},
                   treat_false_as_none: bool = True) -> 'Params':
        """
        Set leaf values.

        Arguments:
            key: String or None. if None, then val must be a dictionary.
                If string, it can be a nested key separated by dots.
            val: A leaf value or a dictionary. If dictionary, then the method walks down
                the dictionary hierarchy and sets the leaf values.
            treat_false_as_none: If val is a dictionary and if self[key] is False, then go ahead and replace
                self[key] with val. In other words expand the configuration tree beyond key. This is the behaviour
                if self[key] is None.
        Note: Setting a key to None effectively unsets it (though the key exists in the dictionary).

        Returns:
            Self
        """
        if key is None:
            assert isinstance(val, Mapping), 'val must be an instance of Mapping if key is None'
            keys = []
        else:
            keys = key.split('.')

        if len(keys) > 1:
            p = self.get(keys[0])
            if p is None:
                p = self._new()
                self._set_val_(keys[0], p)
                p = self.get(keys[0])
            p.set_leaves('.'.join(keys[1:]), val)
        elif isinstance(val, Mapping):
            p = self if key is None else self.get(key)
            if p is None or (p is False and treat_false_as_none):
                self._set_val_(key, self._new(val))
            elif isinstance(p, Params):
                for k in val.keys():
                    p.set_leaves(k, val[k])
            else:
                raise ValueError(f'key ({p}) is not a dictionary. Cannot set value ({val})')
        else:
            self._set_val_(key, val)

        return self

    def _set_val_(self, key: Any, val: Any) -> None:
        """
        Set key to val except if object is frozen or key is a new key and the object was sealed.
        Note: Setting a key to None effectively unsets it (though the key exists in the dictionary).
        """
        # if val is None:
        #     return
        if self.is_frozen():
            raise Exception(f'Object is frozen, therefore key {key} cannot be modified')
        if self.is_sealed() and key not in dict.keys(self):
            raise Exception(f'Object is sealed, new key {key} cannot be added')

        keys = key.split('.')
        if len(keys) > 1:
            p = dict.get(self, keys[0])
            if p is None:
                p = self._new()
                dict.__setitem__(self, keys[0], p)
            assert isinstance(p, self._params_class()), f'Cannot assign to key {keys[0]} of type {type(keys[0])}. ' +\
                                                        f'Need type {self._params_class()}.'
            p._set_val_('.'.join(keys[1:]), val)  # pylint: disable=protected-access
        else:
            if isinstance(val, Mapping):  # and not isinstance(val, self._params_class()):
                val = self._dict2params(val)
            dict.__setitem__(self, key, val)

    @staticmethod
    def _intercept(key: str) -> bool:
        """Return True if we manage this key, False if it's delegated to dict"""
        return key.startswith('__') and key.endswith('__')

    def __getattr__(self, key: str) -> Any:
        if key.startswith('__') and key.endswith('__'):
            return dict.__getattribute__(self, key)
        else:
            return self._get_val_(key)

    def __setattr__(self, key: Any, val: Any) -> None:
        if key.startswith('__') and key.endswith('__'):
            return dict.__setattr__(self, key, val)
        else:
            return self._set_val_(key, val)

    def __getitem__(self, key: Any) -> Any:
        if key.startswith('__') and key.endswith('__'):
            return dict.__getattribute__(self, key)
        else:
            return self._get_val_(key)

    def __setitem__(self, key: Any, val: Any) -> None:
        if key.startswith('__') and key.endswith('__'):
            return dict.__setattr__(self, key, val)
        else:
            return self._set_val_(key, val)

    def __eq__(self, __o: object) -> bool:
        keys = set(self.keys())
        if set(__o.keys()) != keys:
            return False
        for key in keys:
            if self[key] != __o[key]:
                return False
        return True

    def is_frozen(self) -> bool:
        """
        Return True if the object is frozen. False otherwise.
        Note: At the time of unpickling, the dictionary properties are set before the
        _isFrozen and _isSealed attributes are set (via. __setstate__). Therefore during unpickling this method is
        invoked (by _set_val_) even before the _isFrozen attribute is set. Therfore we
        need to use getattr instead of dict.__getattribute__.
        """
        try:
            return dict.__getattribute__(self, '_isFrozen')
        except AttributeError:
            return False

    def is_sealed(self) -> bool:
        """
        Return True if the object is sealed. False otherwise.
        Note: At the time of unpickling, the dictionary properties are set before the
        _isFrozen and _isSealed attributes are set (via. __setstate__). Therefore during unpickling this method is
        invoked (by _set_val_) even before the _isSealed attribute is set. Therfore we
        need to use getattr instead of dict.__getattribute__.
        """
        try:
            return dict.__getattribute__(self, '_isSealed')
        except AttributeError:
            return False

    def freeze(self) -> 'Params':
        """Freeze the object. No further changes will be permitted."""
        dict.__setattr__(self, '_isFrozen', True)
        for k, v in self.items():
            if isinstance(v, Params) and self._intercept(k):
                v.freeze()
        return self

    def seal(self) -> 'Params':
        """Seal the object. You can add new keys but can't modify them."""
        dict.__setattr__(self, '_isSealed', True)
        for k, v in self.items():
            if isinstance(v, Params) and self._intercept(k):
                v.seal()
        return self


class NDict(dict):
    """
    A dictionary who's string keys are also accessible via. dot notation i.e., properties interface.
    This implies that string keys are properties are string keys. Unlike Bunch this class also allows you to
    set string keys via the properties interface (i.e. dot notation).
    Non-string keys are accessed using the regular dictionary interfaces (square brackets, .get, .set etc.)
    """

    def __init__(self, initialdata: Any = None, **kwargs: Any) -> None:
        """Initializes a new dict.

        Args:
            initialdata: A dictionary of keys and values to initialize the NDict. kwargs must not be
                provided if this is.
            kwargs: keys and values to populate the dictionary with. initialdata must not be provided if this is.
        """
        if initialdata is not None:
            assert not kwargs
            dict.__init__(self, initialdata)
        else:
            dict.__init__(self, **kwargs)

    def updated(self, dct: Any = None, **kwargs: Any) -> 'NDict':
        """Update the dictionary same as update except that it returns itself, thus enabling call chaining."""
        if dct is not None:
            self.update(dct, **kwargs)
        else:
            self.update(**kwargs)
        return self

    def __getattr__(self, key: str) -> Any:
        if isinstance(key, str) and key.startswith('__') and key.endswith('__'):
            return dict.__getattribute__(self, key)
        else:
            return self._get_val_(key)

    def __setattr__(self, key: Any, val: Any) -> None:
        if isinstance(key, str) and isinstance(key, str) and key.startswith('__') and key.endswith('__'):
            return dict.__setattr__(self, key, val)
        else:
            return self._set_val_(key, val)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str) and isinstance(key, str) and key.startswith('__') and key.endswith('__'):
            return dict.__getattribute__(self, key)
        else:
            return self._get_val_(key)

    def __setitem__(self, key: Any, val: Any) -> None:
        if isinstance(key, str) and key.startswith('__') and key.endswith('__'):
            return dict.__setattr__(self, key, val)
        else:
            return self._set_val_(key, val)

    def _set_val_(self, key: Any, val: Any) -> None:
        dict.__setitem__(self, key, val)

    def _get_val_(self, key: Any) -> Any:
        return dict.get(self, key)


def to_ndict(o: Any) -> Any:
    """Descend down a collection structure converting every dictionary to an NDict.
    Iterables get converted to lists as a side effect.
    """
    if isinstance(o, (List, Tuple)):
        return [to_ndict(item) for item in o]
    elif isinstance(o, (Mapping, Dict)):
        return NDict({k: to_ndict(v) for k, v in o.items()})
    else:
        return o


class Params2(Params):
    """
    Extends Params to recognize / allow lists type containers (in addition to dictionary type) at the pre-leaf level.
    This enables indexing into (for reading only) what would've been a leaf level list in the Params data-structure.
    e.g. params['classes.1.examples.[0]'] will return the first element of the list at 'classes.1.examples'.
    """
    @classmethod
    def _new(cls, val: Optional[Mapping] = None) -> 'Params2':
        """Create and return a new object of this class.

        When you inherit, override this method to return a new object of your class.
        """
        return Params2(val)

    def _get_val_(self, key: Any, raise_if_not_exists: bool = False) -> Any:
        """Innermost function for getting value."""
        keys = key.split('.')
        if len(keys) > 1:
            listCont = False
            if keys[1].startswith('['):
                listCont = True
                assert len(keys) == 2, KeyError('index key (e.g. [0]) can only appear last')
                id = int(keys[1][1:-1])
            sub_params = dict.get(self, keys[0])
            if (not listCont and not isinstance(sub_params, Mapping)) or (
                    # listCont and not isinstance(sub_params, [List, np.ndarray])
                    listCont and (np.ndim(sub_params) != 1)
                    ):
                if raise_if_not_exists:  # pylint: disable=no-else-raise
                    raise KeyError(
                        f'key {keys[0]} {"does not exist" if sub_params is None else "is the wrong type of container"}')
                else:
                    return None
            elif not listCont:
                return sub_params['.'.join(keys[1:])]
            else:  # listCont
                if len(sub_params) > id:
                    return sub_params[id]
                elif raise_if_not_exists:
                    raise KeyError(f'key {keys[0]}.{key[1]} does not exist')
                else:
                    return None
        else:
            # return dict.__getitem__(self, key)
            val = dict.get(self, key)
            if raise_if_not_exists and val is None:
                raise KeyError(f'key {key} does not exist')
            return val

    def _set_val_(self, key: Any, val: Any) -> None:
        """
        Set key to val except if object is frozen or key is a new key and the object was sealed.
        Note: Setting a key to None effectively unsets it (though the key exists in the dictionary).
        """
        # if val is None:
        #     return
        if self.is_frozen():
            raise Exception(f'Object is frozen, therefore key {key} cannot be modified')
        if self.is_sealed() and key not in dict.keys(self):
            raise Exception(f'Object is sealed, new key {key} cannot be added')

        keys = key.split('.')
        if len(keys) > 1:
            listCont = False
            if keys[1].startswith('['):
                listCont = True
                assert len(keys) == 2, KeyError('index key (e.g. [0]) can only appear last')
                id = int(keys[1][1:-1])
            p = dict.get(self, keys[0])
            if p is None:
                p = self._new() if not listCont else []
                dict.__setitem__(self, keys[0], p)
            elif not listCont:
                assert isinstance(p, self._params_class()), f'Cannot assign to key {keys[0]} of type {type(keys[0])}. ' +\
                                                            f'Need type {self._params_class()}.'
                p._set_val_('.'.join(keys[1:]), val)  # pylint: disable=protected-access
            else:
                # assert isinstance(p, List), f'Cannot assign to key {keys[0]} of type {type(keys[0])}. Need type list'
                assert np.ndim(p) == 1, f'Cannot assign to key {keys[0]} of type {type(keys[0])}. Need type list'
                p[id] = val
        else:
            if isinstance(val, Mapping):  # and not isinstance(val, self._params_class()):
                val = self._dict2params(val)
            dict.__setitem__(self, key, val)

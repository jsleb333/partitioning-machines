import re
from inspect import signature


def camel_to_snake(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


def filter_signature(cls):
    sig = {}
    for base_cls in cls.mro():
        if base_cls is not object:
            sig |= signature(base_cls.__init__).parameters

    def init_cls(*args, **kwargs):
        for kw in list(kwargs):
            if kw not in sig:
                del kwargs[kw]
        return cls(*args, **kwargs)

    return init_cls


def get_default_kwargs(cls):
    default_kwargs = {}
    for base_cls in cls.mro():
        if base_cls is object: break
        sig = signature(base_cls.__init__).parameters
        default_kwargs |= {k:v.default for k, v in sig.items()}
    default_kwargs.pop('self', None)
    default_kwargs.pop('kwargs', None)
    return default_kwargs


class Mock:
    def __getattr__(self, attr):
        return self

    def __call__(self, *args, **kwds):
        return self

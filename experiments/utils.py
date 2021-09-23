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


class Mock:
    def __getattr__(self, attr):
        return self

    def __call__(self, *args, **kwds):
        return self

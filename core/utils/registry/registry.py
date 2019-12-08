from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# DEEP_MODEL = "deep_model"
# TRAINER_HOOK = "trainer_hook"
# TRAINER_CLASS = "trainer_class"
# LOSS_FUNCTION = "loss_function"

# KNOWN_CATEGORIES = [
#     DEEP_MODEL, TRAINER_HOOK, TRAINER_CLASS, LOSS_FUNCTION
# ]


class _Registry(object):
    def __init__(self, category):
        self._category = category

        self._obj_map = {}

    def _do_register(self, name, obj):
        assert (
                name not in self._obj_map
        ), "An object named '{}' was already registered in '{}' registry!".format(name, self._category)
        self._obj_map[name] = obj

    def register(self, obj=None):
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class):
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        name = obj.__name__
        self._do_register(name, obj)

    def get(self, name):
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError("No object named '{}' found in '{}' registry!".format(name, self._category))
        return ret


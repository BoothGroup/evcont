"""Pickle persistence shared by continuation solver objects."""

import pickle


class EVContPersistenceMixin:
    """Save and restore a continuation object's Python state."""

    def _persistence_state(self):
        return self.__dict__

    def save(self, filename):
        with open(filename, "wb") as handle:
            pickle.dump(self._persistence_state(), handle, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as handle:
            state = pickle.load(handle)
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        restore = getattr(obj, "_restore_persistence_state", None)
        if restore is not None:
            restore()
        return obj

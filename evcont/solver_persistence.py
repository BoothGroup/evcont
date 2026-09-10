"""Pickle persistence shared by continuation solver objects."""

import os
import pickle
import tempfile


class EVContPersistenceMixin:
    """Save and restore a continuation object's Python state."""

    def _persistence_state(self):
        return self.__dict__

    def save(self, filename):
        target = os.path.abspath(os.fspath(filename))
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=os.path.dirname(target),
                prefix=f".{os.path.basename(target)}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = handle.name
                pickle.dump(
                    self._persistence_state(), handle, pickle.HIGHEST_PROTOCOL
                )
            os.replace(temporary, target)
        except Exception:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
            raise

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

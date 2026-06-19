class DictAsObject:
    """Convert a dictionary to an object with attribute access."""
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, DictAsObject(v))
            else:
                setattr(self, k, v)

    def __getattr__(self, name):
        return None

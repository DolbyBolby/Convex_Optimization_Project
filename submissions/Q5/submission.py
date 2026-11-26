from controllers import CascadeController


class GCSPolicy(CascadeController):
    def __init__(self, spec):
        super().__init__(gravity=spec["gravity"], spec=spec)
        self.spec = spec


def make_policy(spec):
    return GCSPolicy(spec)

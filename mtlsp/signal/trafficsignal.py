class Signal:
    def __init__(self):
        """Initialize the Signal class.
        """
        self.id = None,
        self._recent_observation = None

    def _get_observation(self):
        pass

if __name__ == "main":
    signal1 = Signal()
    signal_observation = signal1._get_observation()

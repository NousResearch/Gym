from types import SimpleNamespace


class NeMoGymResponseCreateParamsNonStreaming:
    @classmethod
    def model_validate(cls, value):
        return SimpleNamespace(**value)

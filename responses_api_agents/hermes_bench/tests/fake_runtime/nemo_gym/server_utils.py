class ServerClient:
    @classmethod
    def model_construct(cls, **values):
        instance = cls()
        instance.__dict__.update(values)
        return instance

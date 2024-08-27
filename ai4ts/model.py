class Forecast(object):
    def __init__(self, data, name, model=None):
        self.data = data
        self.name = name
        self.model = model

    def __len__(self):
        return len(self.data)

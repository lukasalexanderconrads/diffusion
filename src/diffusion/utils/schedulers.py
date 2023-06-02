from abc import ABC

class BaseScheduler(ABC):
    def __init__(self, **kwargs):
        self.scheduled_variable = kwargs.get('scheduled_variable')

    def update_scheduled_variable(self, model, epoch):
        value = self.get_scheduled_variable_value(epoch)
        setattr(model, self.scheduled_variable, value)

    def get_scheduled_variable_value(self, epoch):
        raise NotImplementedError('cannot use BaseScheduler as a scheduler')

class ConstantScheduler(BaseScheduler):
    """
    keeps the scheduled variable constant
    :param kwargs
        scheduled_variable: str, name of variable of model class to be scheduled
        value: type of scheduled_variable, the value of the scheduled variable
    """
    def __init__(self, **kwargs):
        super(ConstantScheduler, self).__init__(**kwargs)
        self.value = kwargs.get('value')

    def get_scheduled_variable_value(self, epoch):
        return self.value

class HalvingScheduler(BaseScheduler):
    """
    keeps the scheduled variable constant
    :param kwargs
        scheduled_variable: str, name of variable of model class to be scheduled
        value: type of scheduled_variable, the value of the scheduled variable
    """
    def __init__(self, **kwargs):
        super(HalvingScheduler, self).__init__(**kwargs)
        self.value = kwargs.get('start_value')
        assert self.value is not None
        self.halve_after_epochs = kwargs.get('halve_after_epochs', [])

    def get_scheduled_variable_value(self, epoch):
        if epoch in self.halve_after_epochs:
            self.value /= 2.0
        return self.value
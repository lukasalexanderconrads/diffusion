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
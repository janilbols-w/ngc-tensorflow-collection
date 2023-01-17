from tensorflow.keras import callbacks

def init():
  pass

def rank():
  return 0

def local_rank():
  return 0

def size():
  return 1

def DistributedGradientTape(tape):
  return tape

def DistributedOptimizer(opt):
  return opt

class DummyCallback(callbacks.Callback):
  def __init__(self):
    pass

class DummyCallbackFactory:
  def BroadcastGlobalVariablesCallback(self, rank):
    return DummyCallback()

callbacks = DummyCallbackFactory()

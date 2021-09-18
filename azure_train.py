from azureml.core import Dataset, Environment, ScriptRunConfig, Workspace
from azureml.train.hypderdrive import RandomParameterSampling, TruncationSelectionPolicy, HyperDriveConfig, PrimaryMetricGoal
from azureml.train.hyperdrive import choice

def azure_train(config):
    ws = Workspace.from_config()

    datastore = ws.get_default_datastore()
from azureml.core import Dataset, Environment, Experiment, ScriptRunConfig, Workspace
from azureml.train.hyperdrive import RandomParameterSampling, TruncationSelectionPolicy, HyperDriveConfig, PrimaryMetricGoal
from azureml.train.hyperdrive import choice

def azure_train(config):
    ws = Workspace.from_config()

    datastore = ws.get_default_datastore()
    dataset = Dataset.File.from_files(path=(datastore, 'datasets'))
    experiment = Experiment(workspace=ws, name="htn-clothing-captioning")
    env = Environment.get(ws, name='AzureML-tensorflow-2.4-ubuntu18.04-py37-cuda11-gpu')

    script_config = ScriptRunConfig(
        source_directory='./',
        script='train.py',
        compute_target='htn-compute',
        arguments=[
            '--dataset_base_path', dataset.as_named_input('input').as_mount(),
        ],
        environment=env
    )

    ps = RandomParameterSampling(
        {
            '--hdc_img_size': choice(*[(224,224),(512,512)])
        }
    )

    policy = TruncationSelectionPolicy(evaluation_interval=2,
                                       truncation_percentage=25,
                                       delay_evaluation=10)

    hdc = HyperDriveConfig(run_config=script_config,
                           hyperparameter_sampling=ps,
                           policy=policy,
                           primary_metric_name='val_loss',
                           primary_metric_goal=PrimaryMetricGoal.MINIMIZE,
                           max_total_runs=10,
                           max_concurrent_runs=10)

    run = experiment.submit(config=hdc)
    aml_url = run.get_portal_url()
    print("Submitted to Hyperdrive")
    print(aml_url)

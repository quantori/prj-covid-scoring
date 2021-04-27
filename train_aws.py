import argparse
import yaml
import os
from sagemaker.pytorch import PyTorch
AWS_DEFAULT_ROLE = "arn:aws:iam::340815445214:role/ml-team_sagemaker_execute"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', metavar='FILE', type=str, required=True)
    parser.add_argument('-s', '--script', metavar='FILE', type=str, default=None)
    parser.add_argument('-d', '--dependencies', nargs='*', default=None)
    parser.add_argument('--instance-type', type=str, default='local',
                        help="Possible values: local, ml.p2.xlarge, ml.p3.2xlarge")
    parser.add_argument('--instance-count', type=int, default=1,
                        help='Usualy you need only one instance')
    parser.add_argument('--role', type=str, default=AWS_DEFAULT_ROLE)
    parser.add_argument('--wandb_api_key', type=str, default=None)
    args = parser.parse_args()

    assert os.path.exists(args.config) and os.path.isfile(args.config), "Unable to find config file"
    with open(args.config, 'rt') as stream:
        config = yaml.safe_load(stream)
    assert args.script or config['script'], "Train script is not specified"
    hyperparameters = config['hyperparameters']
    hyperparameters['wandb_api_key'] = args.wandb_api_key
    inputs = config['inputs']
    pytorch_estimator = PyTorch(entry_point=args.script or config['script'],
                                instance_type=args.instance_type,
                                dependencies=args.dependencies or config.get('dependencies', []),
                                instance_count=args.instance_count,
                                role=args.role,
                                framework_version='1.8',
                                py_version='py36',
                                #output_path="file:///mnt/d/quantori_projects/prj-covid-scoring/",
                                hyperparameters=hyperparameters)

    pytorch_estimator.fit(inputs)
















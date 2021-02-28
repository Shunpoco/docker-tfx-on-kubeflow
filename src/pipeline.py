import os
import urllib
from typing import Text

import absl
import tensorflow as tf
import tensorflow_model_analysis as tfma

import tfx
from tfx.components import CsvExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import Pusher
from tfx.components import ResolverNode
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.dsl.components.base import executor_spec
from tfx.components.trainer.executor import GenericExecutor

from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.kubeflow.runner import kubeflow_dag_runner

from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing
from tfx.utils.dsl_utils import external_input

from kfp import onprem

_pipeline_name = 'chicago_taxi3'
_persistent_volume_claim = 'data'
_persistent_volume = 'pvc-b59868a7-0d99-4cdd-9a28-cc14bf87ab46'
_persistent_volume_mount = '/home/jovyan/data'

_input_base = os.path.join(_persistent_volume_mount, 'docker-tfx-on-kubeflow/src')
_output_base = os.path.join(_persistent_volume_mount, 'pipelines')
_tfx_root = os.path.join(_persistent_volume_mount, 'tfx')
_pipeline_root = os.path.join(_tfx_root, _pipeline_name)

_data_root = os.path.join(_input_base, 'data')

_transform_module_file = os.path.join(_input_base, 'taxi_transform.py')

_train_module_file = os.path.join(_input_base, 'taxi_train.py')

_serving_model_dir = os.path.join(_output_base, _pipeline_name, 'serving_model')

def _create_pipeline(
    pipeline_name: Text, pipeline_root: Text, data_root: Text, 
    transform_module_file: Text, train_module_file: Text, serving_model_dir: Text, 
    direct_num_workers: int,
) -> pipeline.Pipeline:
    
    # Component 1: Data Ingestion
    example_gen = CsvExampleGen(input_base=data_root)
    
    # Component 2: Statistics
    statistics_gen = StatisticsGen(
        examples=example_gen.outputs['examples'],
    )
    
    # Component 3: Schema
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=False,
    )
    
    # Component 4: Data Validator
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'],
    )
    
    # Component 5: Transform (Feature Engineering)
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=transform_module_file,
    )
    
    # Component 6: Trainer
    trainer = Trainer(
        module_file=train_module_file,
        custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
        examples=transform.outputs['transformed_examples'],
        schema=schema_gen.outputs['schema'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=trainer_pb2.TrainArgs(num_steps=10000),
        eval_args=trainer_pb2.EvalArgs(num_steps=5000),
    )

    # Component 7: Evaluate
    model_resolver = ResolverNode(
        instance_name='latest_blessed_model_resolver',
        resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing),
    )

    eval_config = tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(label_key='tips')
        ],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(class_name='ExampleCount'),
                    tfma.MetricConfig(class_name='BinaryAccuracy',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value': 0.5}),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value': -1e-10}))),
                ],
            ),
        ],
        slicing_specs=[
            tfma.SlicingSpec(),
            tfma.SlicingSpec(feature_keys=['trip_start_hour']),
        ],
    )

    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config,
    )

    # Component 8: Push
    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=_serving_model_dir,
            ),
        ),
    )

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=[
            example_gen, statistics_gen, schema_gen,
            example_validator, transform, trainer,
            model_resolver, evaluator, pusher
        ],
        beam_pipeline_args=[f'--direct_num_workers={direct_num_workers}'],
        enable_cache=True,
    )

if __name__ == '__main__':
    tfx_image = 'tensorflow/tfx:0.27.0'

    metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()
    metadata_config.mysql_db_service_host.value = 'mysql.kubeflow'
    metadata_config.mysql_db_service_port.value = "3306"
    metadata_config.mysql_db_name.value = "metadb"
    metadata_config.mysql_db_user.value = "root"
    metadata_config.mysql_db_password.value = ""
    metadata_config.grpc_config.grpc_service_host.value = '10.100.244.39' # Cluster IP of metadata-grpc-service
    metadata_config.grpc_config.grpc_service_port.value = '8080'

    runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        tfx_image=tfx_image,
        pipeline_operator_funcs=([
            onprem.mount_pvc(
                _persistent_volume_claim,
                _persistent_volume,
                _persistent_volume_mount,
            ),
        ]),
        kubeflow_metadata_config=metadata_config,
    )

    kubeflow_dag_runner.KubeflowDagRunner(config=runner_config).run(
        _create_pipeline(
            pipeline_name=_pipeline_name,
            pipeline_root=_pipeline_root,
            data_root=_data_root,
            transform_module_file=_transform_module_file,
            train_module_file=_train_module_file,
            serving_model_dir=_serving_model_dir,
            direct_num_workers=0,
        ),
    )
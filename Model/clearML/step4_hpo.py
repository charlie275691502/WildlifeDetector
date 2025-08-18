from clearml import Task, Dataset
from clearml.automation import HyperParameterOptimizer
from clearml.automation import UniformIntegerParameterRange, DiscreteParameterRange
import logging
import time
import json
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the HPO task
task = Task.init(
    project_name="GarbageClassifier",
    task_name='HPO',
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=False
)

# Connect parameters
args = {
    'dataset_task_id': '',
    'base_train_task_id': '8b3f72f435704677abe4e27323d3eba3',  # Will be set from pipeline
    'num_trials': 8,  # Reduced from 10 to 3 trials
    'time_limit_minutes': 1000,  # Reduced from 60 to 5 minutes
    'run_as_service': False,
    'test_queue': 'helloworld',  # Queue for test tasks
    'processed_dataset_id': '99e286d358754697a37ad75c279a6f0a',  # Will be set from pipeline
    'batch_size': 32,  # Default batch size
    'learning_rate': 0.001,  # Default learning rate
    'neural_count': 128  # Default weight decay
}
args = task.connect(args)
logger.info(f"Connected parameters: {args}")

# Execute the task remotely
task.execute_remotely()

# Get the actual training model task
try:
    BASE_TRAIN_TASK_ID = args['base_train_task_id']
    logger.info(f"Using base training task ID: {BASE_TRAIN_TASK_ID}")
except Exception as e:
    logger.error(f"Failed to get base training task ID: {e}")
    raise

# Create the HPO task
hpo_task = HyperParameterOptimizer(
    base_task_id=BASE_TRAIN_TASK_ID,
    hyper_parameters=[
        UniformIntegerParameterRange('neural_count', min_value=128, max_value=256, step_size=128),  # Reduced range
        UniformIntegerParameterRange('batch_size', min_value=16, max_value=32, step_size=16),  # Reduced range
        DiscreteParameterRange('learning_rate', values=[0.00006, 0.0005, 0.001]),  # Reduced range
    ],
    objective_metric_title='validation',
    objective_metric_series='accuracy',
    objective_metric_sign='max',
    max_number_of_concurrent_tasks=2,
    optimization_time_limit=args['time_limit_minutes'] * 60,
    compute_time_limit=None,
    total_max_jobs=args['num_trials'],
    min_iteration_per_job=1,
    max_iteration_per_job=20,
    pool_period_min=1.0,  # Reduced from 2.0 to 1.0 to check more frequently
    execution_queue=args['test_queue'],
    save_top_k_tasks_only=2,  # Reduced from 5 to 2
    parameter_override={
        'processed_dataset_id': args['dataset_task_id'],
        'General/processed_dataset_id': args['dataset_task_id'],
        'test_queue': args['test_queue'],
        'General/test_queue': args['test_queue'],
        'batch_size': args['batch_size'],
        'General/batch_size': args['batch_size'],
        'learning_rate': args['learning_rate'],
        'General/learning_rate': args['learning_rate'],
        'neural_count': args['neural_count'],
        'General/neural_count': args['neural_count']
    }
)

# Start the HPO task
logger.info("Starting HPO task...")
hpo_task.start()

# Wait for optimization to complete
logger.info(f"Waiting for optimization to complete (time limit: {args['time_limit_minutes']} minutes)...")
hpo_task.wait()

# Get the top performing experiments
try:
    top_exp = hpo_task.get_top_experiments(top_k=1)  # Get only the best experiment
    if top_exp:
        best_exp = top_exp[0]
        logger.info(f"Best experiment: {best_exp.id}")
        
        # Get the best parameters and accuracy
        best_params = best_exp.get_parameters()
        logger.info(best_params)
        metrics = best_exp.get_last_scalar_metrics()
        best_accuracy = metrics['validation']['accuracy'] if metrics and 'validation' in metrics and 'accuracy' in metrics['validation'] else None
        
        # Log detailed information about the best experiment
        logger.info("Best experiment parameters:")
        logger.info(f"  - batch_size: {best_params.get('batch_size')}")
        logger.info(f"  - learning_rate: {best_params.get('learning_rate')}")
        logger.info(f"  - neural_count: {best_params.get('neural_count')}")
        logger.info(f"Best validation accuracy: {best_accuracy}")
        
        # Save best parameters and accuracy
        best_results = {
            'parameters': best_params,
            'accuracy': best_accuracy
        }
        
        # Save to a temporary file
        temp_file = 'best_parameters.json'
        with open(temp_file, 'w') as f:
            json.dump(best_results, f, indent=4)
        
        # Upload as artifact
        task.upload_artifact('best_parameters', temp_file)
        logger.info(f"Saved best parameters with accuracy: {best_accuracy}")
        
        # Also save as task parameters for easier access
        task.set_parameter('best_parameters', best_params)
        task.set_parameter('best_accuracy', best_accuracy)
        
        logger.info("Best parameters saved as both artifact and task parameters")
    else:
        logger.warning("No experiments completed yet. This might be normal if the optimization just started.")
except Exception as e:
    logger.error(f"Failed to get top experiments: {e}")
    raise

# Make sure background optimization stopped
hpo_task.stop()
logger.info("Optimizer stopped")

print('We are done, good bye')
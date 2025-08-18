from clearml import Task
from clearml.automation import PipelineController


def pre_execute_callback_example(a_pipeline, a_node, current_param_override):
    # type (PipelineController, PipelineController.Node, dict) -> bool
    print(
        "Cloning Task id={} with parameters: {}".format(
            a_node.base_task_id, current_param_override
        )
    )
    # if we want to skip this node (and subtree of this node) we return False
    # return True to continue DAG execution
    return True


def post_execute_callback_example(a_pipeline, a_node):
    # type (PipelineController, PipelineController.Node) -> None
    print("Completed Task id={}".format(a_node.executed))
    # if we need the actual executed Task: Task.get_task(task_id=a_node.executed)
    return


def run_pipeline():
    # Connecting ClearML with the current pipeline,
    # from here on everything is logged automatically
    pipe = PipelineController(
        name="Train model", project="GarbageClassifier", version="0.0.1", add_pipeline_tags=False
    )

    pipe.add_parameter(
        "url",
        "dataset_url",
    )

    pipe.set_default_execution_queue("helloworld")

    pipe.add_step(
        name="stage_train",
        base_task_project="GarbageClassifier",
        base_task_name="Train model",
        parameter_override={"General/dataset_task_id": "80b62f1dc03d4c61b299d63ed4f61a41"},
        cache_executed_step=True,
    )

    # for debugging purposes use local jobs
    pipe.start_locally()

    # Starting the pipeline (in the background)
    # pipe.start(queue="pipeline")  # already set pipeline queue

    print("done")

if __name__ == '__main__':
    run_pipeline()
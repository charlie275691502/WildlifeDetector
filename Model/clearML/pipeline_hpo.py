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
        name="HPO", project="GarbageClassifier", version="0.0.1", add_pipeline_tags=False
    )

    pipe.add_parameter(
        "url",
        "dataset_url",
    )

    pipe.set_default_execution_queue("helloworld")

    pipe.add_step(
        name="stage_hpo",
        base_task_project="GarbageClassifier",
        base_task_name="HPO",
        parameter_override={
                "General/dataset_task_id": "26a16aa50c9c42a09a26af57fcebf215",
                "General/base_train_task_id": "7216ba1c555b4fe0b95fbb2288869db0"},
        cache_executed_step=True,
    )

    # for debugging purposes use local jobs
    pipe.start_locally()

    # Starting the pipeline (in the background)
    # pipe.start(queue="pipeline")  # already set pipeline queue

    print("done")

if __name__ == '__main__':
    run_pipeline()
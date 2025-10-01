Start Worker:
clearml-agent daemon --queue helloworld --services-mode

Compile Task:
python Model/clearML/task/yolo_preprocessing.py

Run Pipeline:
python Model/clearML/pipeline/pipeline_preprocessing.py

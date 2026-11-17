"""Task 3: rating prediction with timestamps. Same pipeline as task 1 (see task1.py),
with models that also learn how ratings change over time.

Run:  python task3.py
"""
from task1 import predict, train

if __name__ == "__main__":
    bundle = train("tasks_3_4_train.csv", use_time=True)
    predict(bundle, "task_3_test.csv", "task_3_predictions.csv", use_time=True)

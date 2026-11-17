"""Task 4: which-item prediction with timestamps. Same pipeline as task 2 (see task2.py),
plus clues about when each book was being read and what the user read around then.

Run:  python task4.py
"""
from task2 import predict, train

if __name__ == "__main__":
    bundle = train("tasks_3_4_train.csv", use_time=True)
    predict(bundle, "task_4_test.csv", "task_4_predictions.csv", use_time=True)

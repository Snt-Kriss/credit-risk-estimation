import json
def test_recall_threshhold():
    with open("metrics.json", 'r') as f:
        metrics= json.load(f)

    recall= metrics.get("recall", 0)
    assert recall >= 0.85, f"Recall is too low: {recall:.2f}"
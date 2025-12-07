import json
from ukaiforum_inspect.dataset import samples
from inspect_ai.dataset import json_dataset

def create_json_dataset(path: str = "dataset.json"):
    """
    Given an in-memory dataset of Inspect Samples, create a JSON dataset and save it to a file. Then, reload the dataset from the file using Inspect's json_dataset function.
    """

    samples_dict = [sample.model_dump(exclude_none=True) for sample in samples]
    json.dump(samples_dict, open(path, "w"))
    return json_dataset(path)

if __name__ == "__main__":
    create_json_dataset(path="dataset.json")
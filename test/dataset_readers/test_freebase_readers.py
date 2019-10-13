from datasets.dataset_readers.freebase_readers import OpenKEDatasetReader
from allennlp.data.fields import ArrayField
from pathlib import Path

if __name__ == "__main__":
    test_data_path = Path(
        '/Users/dhruv/UnsyncedDocuments/IESL/kb_completion/datasets/.data/test'
    )
    reader = OpenKEDatasetReader(all_datadir=test_data_path)
    instances = reader.read()
    print(instances)

    for instance in instances:
        print(instance)

        for name, field in instance.items():
            if isinstance(field, ArrayField):
                print(name, field.array)

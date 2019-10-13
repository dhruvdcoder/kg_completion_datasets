from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.fields import ArrayField, LabelField
from typing import Iterable, Callable, Optional, Tuple
from .constants import Mode
from .types import PathT, NegativeSamplerProtocol
from ..file_readers.openke import TrainIdReader, EntityIdReader
from ..sampling.simple_samplers import UniformNegativeSampler
from pathlib import Path
from .lazy_iterators import LazyIteratorWithSingleNegativeSampling
import numpy as np

SampleT = Tuple[int, int, int]


@DatasetReader.register("openke-dataset")
class OpenKEDatasetReader(DatasetReader):
    """OpenKE style KB completion dataset reader
    """
    urls = {
        'FB15K237':
        'https://drive.google.com/uc?id=1R1c-hfPSxUfQoHY5i_H0MVpXOyErVFjf'
    }

    def __init__(self,
                 dataset_name: str = 'FB15K237',
                 all_datadir: PathT = Path('.data'),
                 mode: str = Mode.train,
                 number_negative_samples: int = 1):
        self.dataset_name = dataset_name
        self.all_datadir = Path(all_datadir)
        self.mode = mode
        self.number_negative_samples = number_negative_samples

        if self.mode == Mode.train:
            self.file_reader = TrainIdReader(self.all_datadir / dataset_name)
        else:
            raise ValueError
        self.negative_sampler = UniformNegativeSampler()

    def generate_replacement_index(self):
        return np.random.choice([0, 2])

    def _read(self) -> Iterable[Instance]:
        """Read positive samples

        Arguments:

            dirname: Could be parent directory containing all dataset
                or the directory containing a particular dataset
        """

        return self.file_reader()

    def samples_to_instance(self, positive_sample: SampleT,
                            negative_sample: SampleT) -> Instance:
        pos_head = ArrayField(
            np.array(positive_sample[0], dtype=np.int), dtype=np.int)
        pos_relation = ArrayField(
            np.array(positive_sample[1], dtype=np.int), dtype=np.int)
        pos_tail = ArrayField(
            np.array(positive_sample[2], dtype=np.int), dtype=np.int)
        neg_head = ArrayField(
            np.array(negative_sample[0], dtype=np.int), dtype=np.int)
        neg_relation = ArrayField(
            np.array(negative_sample[1], dtype=np.int), dtype=np.int)
        neg_tail = ArrayField(
            np.array(negative_sample[2], dtype=np.int), dtype=np.int)
        label = LabelField(
            1, skip_indexing=True)  # first one is always the pos sample
        fields = {
            "p_h": pos_head,
            "p_r": pos_relation,
            "p_t": pos_tail,
            "n_h": neg_head,
            "n_r": neg_relation,
            "n_t": neg_tail,
            "label": label
        }

        return Instance(fields)

    def single_negative_sampler_generator(self):
        return lambda pos: self.negative_sampler.sample(pos,
                                                        self.generate_replacement_index())

    def read(self) -> Iterable[Instance]:
        """Lazyly return instances by negatively sampling"""
        positive_samples = self._read()
        all_entities = list(
            EntityIdReader(self.all_datadir / self.dataset_name)())
        self.negative_sampler.entities = all_entities
        self.negative_sampler.positives = positive_samples

        return LazyIteratorWithSingleNegativeSampling(
            self.single_negative_sampler_generator(), positive_samples,
            self.samples_to_instance)


if __name__ == "__main__":
    test_data_path = Path(
        '/Users/dhruv/UnsyncedDocuments/IESL/kb_completion/datasets/.data/test'
    )
    OpenKEDatasetReader(all_datadir=test_data_path)
    instances = OpenKEDatasetReader.read()

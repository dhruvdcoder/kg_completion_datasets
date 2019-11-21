from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.fields import ArrayField, LabelField
from allennlp.common.params import Params
from typing import Iterable, Callable, Optional, Tuple
from .constants import Mode
from .types import PathT, NegativeSamplerProtocol
from ..file_readers.openke import TrainIdReader, EntityIdReader, ValIdReader
from ..sampling.simple_samplers import UniformNegativeSampler
from pathlib import Path
from .lazy_iterators import (LazyIteratorWithNegativeSampling,
                             LazyIteratorWithSequentialNegativeSampling)
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

    lazy_iter = LazyIteratorWithNegativeSampling

    def __init__(self,
                 dataset_name: str = 'FB15K237',
                 all_datadir: PathT = Path('.data'),
                 mode: str = Mode.train,
                 number_negative_samples: int = 1):
        super().__init__()
        self.dataset_name = dataset_name
        self.all_datadir = Path(all_datadir)
        self.mode = mode
        self.number_negative_samples = number_negative_samples

        if self.mode == Mode.train:
            self.file_reader = TrainIdReader(self.all_datadir / dataset_name)
        elif self.mode == Mode.validate:
            self.file_reader = ValIdReader(self.all_datadir / dataset_name)
        self.negative_sampler = UniformNegativeSampler()

    def generate_replacement_index(self):
        return np.random.choice([0, 1])  # (eh,et,r)

    def replacement_index_generator(self):
        for i in range(self.number_negative_samples):
            yield self.generate_replacement_index()

    def _read(self, filename=None) -> Iterable[Instance]:
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
            np.array(positive_sample[2], dtype=np.int), dtype=np.int)
        pos_tail = ArrayField(
            np.array(positive_sample[1], dtype=np.int), dtype=np.int)
        neg_head = ArrayField(
            np.array(negative_sample[0], dtype=np.int), dtype=np.int)
        neg_relation = ArrayField(
            np.array(negative_sample[2], dtype=np.int), dtype=np.int)
        neg_tail = ArrayField(
            np.array(negative_sample[1], dtype=np.int), dtype=np.int)
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

    def negative_sampler_generator(self):
        return lambda pos: self.negative_sampler(
            pos, N=self.number_negative_samples,
            replacement_index=self.replacement_index_generator())

    def read(self, filename=None) -> Iterable[Instance]:
        """Lazyly return instances by negatively sampling"""
        positive_samples = set(self._read())
        all_entities = list(
            EntityIdReader(self.all_datadir / self.dataset_name)())
        self.negative_sampler.entities = all_entities
        self.negative_sampler.positives = positive_samples

        return self.lazy_iter(self.negative_sampler_generator(),
                              positive_samples, self.samples_to_instance)


@DatasetReader.register("openke-classification-dataset")
class OpenKEClassificationDatasetReader(OpenKEDatasetReader):
    lazy_iter = LazyIteratorWithSequentialNegativeSampling

    def samples_to_instance(self, sample: SampleT, label: int) -> Instance:
        head = ArrayField(np.array(sample[0], dtype=np.int), dtype=np.int)
        relation = ArrayField(np.array(sample[2], dtype=np.int), dtype=np.int)
        tail = ArrayField(np.array(sample[1], dtype=np.int), dtype=np.int)
        label_f = LabelField(label, skip_indexing=True)
        fields = {'h': head, 't': tail, 'r': relation, 'label': label_f}

        return Instance(fields)


if __name__ == "__main__":
    test_data_path = Path(
        '/Users/dhruv/UnsyncedDocuments/IESL/kb_completion/datasets/.data/test'
    )
    OpenKEDatasetReader(all_datadir=test_data_path)
    instances = OpenKEDatasetReader.read()

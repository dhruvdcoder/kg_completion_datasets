from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.instance import Instance
from pathlib import Path
from .types import PathT
from ..file_readers.openke import SamplesIdReader, JustNumberReader

from typing import Iterable, Optional, List, Tuple
import numpy as np
from allennlp.data.fields import ArrayField
import pickle
import itertools
import logging
logger = logging.getLogger(__name__)


@DatasetReader.register('classification-validation-dataset')
class ClassificationValidationDatasetReader(DatasetReader):
    """ Expects file of the form:
        number of samples(int)
        head_id(int) tail_id(int) relation_id(int) label(0 or 1)
        ...
        ...
    """

    def __init__(self,
                 dataset_name: Optional[str] = None,
                 all_datadir: PathT = Path('.data'),
                 validation_file: str = 'classification_dev.txt'):

        if dataset_name is None:
            raise ValueError("provide dataset_name")
        self.dataset_name = dataset_name
        self.all_datadir = Path(all_datadir)
        self.validation_file = validation_file
        self.file_reader = SamplesIdReader()

    def sample_to_instance(self,
                           sample: Tuple[int, int, int, int]) -> Instance:
        head = ArrayField(np.array(sample[0], dtype=np.int), dtype=np.int)
        tail = ArrayField(np.array(sample[1], dtype=np.int), dtype=np.int)
        relation = ArrayField(np.array(sample[2], dtype=np.int), dtype=np.int)
        label = ArrayField(np.array(sample[3], dtype=np.int), dtype=np.int)

        return Instance({'h': head, 't': tail, 'r': relation, 'label': label})

    def _read(self, filename=None) -> Iterable[Tuple]:
        return self.file_reader.read(
            self.all_datadir / self.dataset_name / self.validation_file)

    def read(self, filename=None) -> Iterable[Instance]:
        instances = [self.sample_to_instance(i) for i in self._read(filename)]

        return instances

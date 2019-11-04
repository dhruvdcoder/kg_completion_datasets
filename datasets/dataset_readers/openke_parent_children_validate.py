from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.instance import Instance
from pathlib import Path
from .types import PathT
from ..file_readers.openke import SamplesIdReader, JustNumberReader

from typing import Iterable, Optional, List, Tuple
import numpy as np
from allennlp.data.fields import ArrayField
from logging import logger
import pickle
import itertools


@DatasetReader.register(
    'openke-single-relation-parent-childrem-validation-dataset')
class OpenKESingleRelationParentChildrenValidationDatasetReader(DatasetReader):
    """Reads oracle/world files and creates a validation set where each
    sample is node, parents, children
    """

    def __init__(self,
                 dataset_name: str = None,
                 all_datadir: PathT = Path('.data'),
                 validation_file: str = None,
                 all_true_files: Optional[List[str]] = None):

        if dataset_name is None:
            raise ValueError("dataset name cannot be None")
        super().__init__()
        self.dataset_name = dataset_name
        self.all_datadir = Path(all_datadir)
        self.all_true_files = all_true_files
        self.validation_file = validation_file

    def cache_filename(self):
        filename = '_'.join([self.validation_file] +
                            self.all_true_files) + '.pkl'

        return self.all_datadir / self.dataset_name / filename

    def generate_samples(self, filename):
        yield from SamplesIdReader().read(filename)

    def generate_entities(self, filename):
        for sample_tuple in SamplesIdReader().read(filename):
            yield sample_tuple[0]

    def num_entities(self, filename):
        return JustNumberReader().read(filename)

    def _read(self, filename=None) -> Iterable[Tuple]:
        # check cache
        cache_file = self.cache_filename()

        if cache_file.exists() and cache_file.is_file():
            with open(cache_file, 'rb') as cf:
                logger.info(
                    "Loading parents children validation data from cache at {}"
                    .format(cache_file))
                samples = pickle.load(cf)

            return samples
        # read all true
        parents = {}
        children = {}

        for sample in itertools.chain(*(self.generate_samples(
                self.all_datadir / self.dataset_name / filename))):
            samples_children = children.get(sample[0], set())
            samples_children.add(sample[1])
            children[sample[0]] = samples_children

            samples_parents = parents.get(sample[0], set())
            samples_parents.add(sample[1])
            parents[sample[0]] = samples_parents

        samples = [(entity, list(parents[entity]), list(children[entity]))
                   for entity in self.generate_entities()]
        # cache
        # save to cache
        with open(cache_file, 'wb') as cf:
            logger.info("Writing parents children validation cache at {}".
                        format(cache_file))
            pickle.dump(samples, cf)

        return samples

    def samples_to_instance(
            self, sample: Tuple[int, List[int], List[int]]) -> Instance:
        node = ArrayField(np.array(sample[0], dtype=np.int), dtype=np.int)
        parents = ArrayField(np.array(sample[1], dtype=np.int), dtype=np.int)
        children = ArrayField(np.array(sample[1], dtype=np.int), dtype=np.int)
        fields = {'node': node, 'gt_parent': parents, 'gt_child': children}

        return fields

    def read(self, filename=None) -> Iterable[Instance]:
        instances = [self.sample_to_instance(i) for i in self._read()]

        return instances

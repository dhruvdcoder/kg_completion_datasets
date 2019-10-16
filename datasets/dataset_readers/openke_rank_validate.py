from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.instance import Instance
from pathlib import Path
from .types import PathT
from ..file_readers.openke import (RankValIdReader, RankValidationSampleT,
                                   RankValidationSampleHeadAndTailT,
                                   get_relation_from_val_sample)
from typing import Iterable
import numpy as np
from allennlp.data.fields import ArrayField


@DatasetReader.register('openke-rank-validation-dataset')
class OpenKERankValidationDatasetReader(DatasetReader):
    def __init__(self,
                 dataset_name: str = 'FB15K237',
                 all_datadir: PathT = Path('.data')):
        super().__init__()
        self.dataset_name = dataset_name
        self.all_datadir = Path(all_datadir)
        self.file_reader = RankValIdReader(
            self.all_datadir / self.dataset_name)

    def _read(self, filename=None) -> Iterable[Instance]:
        return self.file_reader()

    def sample_to_instance(
            self,
            complete_sample: RankValidationSampleHeadAndTailT) -> Instance:

        head_replaced, tail_replaced = complete_sample

        relation = get_relation_from_val_sample(complete_sample)

        head_replacement_relation_field = ArrayField(
            np.array(relation, dtype=np.int), dtype=np.int)

        head_replacement_tail_field = ArrayField(
            np.array(head_replaced[0], dtype=np.int), dtype=np.int)

        head_replacement_entities_field = ArrayField(
            np.array(head_replaced[2], dtype=np.int), dtype=np.int)

        tail_replacement_relation_field = ArrayField(
            np.array(relation, dtype=np.int), dtype=np.int)

        tail_replacement_head_field = ArrayField(
            np.array(tail_replaced[0], dtype=np.int), dtype=np.int)

        tail_replacement_entities_field = ArrayField(
            np.array(tail_replaced[2], dtype=np.int), dtype=np.int)

        fields = {
            "hr_t": head_replacement_tail_field,
            "hr_r": head_replacement_relation_field,
            "hr_e": head_replacement_entities_field,
            "tr_h": tail_replacement_head_field,
            "tr_r": tail_replacement_relation_field,
            "tr_e": tail_replacement_entities_field
        }

        return Instance(fields)

    def read(self, filename=None) -> Iterable[Instance]:
        instances = self._read()

        return instances

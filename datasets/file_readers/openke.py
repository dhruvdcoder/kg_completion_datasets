from .file_reader import FileReader
from pathlib import Path
from typing import Union, Iterable, Any, List, Tuple


class JustNumberReader(FileReader):
    """Reads just the number in the first line of the file.

    Sometimes just this is enough. For instance when we just
    want to read the ids of entities but not the exact entities"""

    def read(self, filename: Path) -> Iterable[int]:
        with open(filename) as f:
            try:
                for line in f:
                    v = int(line)

                    break
            except Exception as e:
                raise IOError(
                    "File should have an integer entry in first line but is {}"
                    .format(line)) from e

        return range(v)


@FileReader.register('entity-id-reader')
class EntityIdReader(JustNumberReader):
    filename = 'entity2id.txt'


class SamplesIdReader(FileReader):
    """ Reads samples from Openke files assuming the following structure

    numsamples (int)
    head_entity_id relation_id tail_entity_id
    ...            ...         ...
    """
    filename = 'override_this'

    def read(self, filename: Path) -> Iterable[Tuple[int, int, int]]:

        with open(filename) as f:
            # read number of samples
            try:
                for line in f:
                    num_samples = int(line)

                    break
            except Exception as e:
                raise IOError(
                    "Format of first line in file {} not as expected".format(
                        filename)) from e
            # read the actual samples
            samples = []

            for i, line in enumerate(f):
                samples.append(tuple(int(idx) for idx in line.split()))

            if len(samples) != num_samples:
                raise IOError(
                    "Number of samples in the file {} "
                    "does not match the number given in the first line".format(
                        filename))

            return samples


@FileReader.register('train-id-reader')
class TrainIdReader(SamplesIdReader):
    filename = 'train2id.txt'


@FileReader.register('val-id-reader')
class ValIdReader(SamplesIdReader):
    filename = 'valid2id.txt'

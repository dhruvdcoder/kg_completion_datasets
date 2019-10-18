from typing import Iterable, Iterator, Any, Callable
from .types import PathT, NegativeSamplerProtocol
from allennlp.data.instance import Instance


class LazyIteratorWithNegativeSampling(Iterable):
    """ Every call to __iter__ will dynamically generate
    the list of samples.

    .. note:: iter() is called once when a statement like
    the following is encountered:
        for element in iterable:
            ...
    This will return an iterator to the for by calling __iter__
    method.
    """

    def __init__(self, negative_sampler: Callable[[Any], Any],
                 positive_samples: Iterable[Any],
                 samples_to_instance: Callable[[Any, Any], Instance]):
        self.negative_sampler = negative_sampler
        self.positive_samples = positive_samples
        self.samples_to_instance = samples_to_instance

    def __iter__(self):
        for positive_sample in self.positive_samples:
            for negative_sample in self.negative_sampler(positive_sample):
                yield self.samples_to_instance(positive_sample,
                                               negative_sample)


class LazyIteratorWithSingleNegativeSampling(LazyIteratorWithNegativeSampling):
    """ Every call to __iter__ will dynamically generate
    the list of samples.

    .. note:: iter() is called once when a statement like
    the following is encountered:
        for element in iterable:
            ...
    This will return an iterator to the for by calling __iter__
    method.
    """

    def __iter__(self):
        for positive_sample in self.positive_samples:
            negative_sample = self.negative_sampler(positive_sample)
            yield self.samples_to_instance(positive_sample, negative_sample)

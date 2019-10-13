from allennlp.common.registrable import Registrable
from typing import Any, Iterable, Tuple, Union, List
from ..types import NegativeSamplerProtocol


class NegtiveSampler(Registrable, NegativeSamplerProtocol):
    def generate_one_negative_sample(self, sample: Tuple,
                                     replacement_index: int, **kwargs) -> Any:
        raise NotImplementedError

    def __call__(self,
                 positive_sample: Tuple,
                 N: int = 1,
                 replacement_index: Union[List[int], int] = 0,
                 **kwargs) -> Iterable[Any]:

        if isinstance(replacement_index, list):
            if len(replacement_index) != N:
                raise ValueError
        else:
            replacement_index = [replacement_index] * N

        for i, ri in enumerate(replacement_index):
            yield self.generate_one_negative_sample(positive_sample, ri,
                                                    **kwargs)

    def sample(self, positive_sample: Tuple, replacement_index: int,
               **kwargs) -> Any:

        return list(
            self.__call__(
                positive_sample,
                N=1,
                replacement_index=replacement_index,
            ))[0]

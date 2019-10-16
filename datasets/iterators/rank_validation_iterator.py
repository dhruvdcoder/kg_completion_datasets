from allennlp.data.iterators import BasicIterator, DataIterator


@DataIterator.register('single-sample-rank-validation-iterator')
class SingleSampleRankValidationIterator(BasicIterator):
    """ The only job of this special iterator is to make sure
    that the batch size is 1. Because the validation logic depends on
    it"""

    def __init__(self, **kwargs):
        batch_size = kwargs.pop('batch_size', 1)
        super().__init__(batch_size=batch_size, **kwargs)

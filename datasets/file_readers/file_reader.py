from allennlp.common.registrable import Registrable
from pathlib import Path
from typing import Union, Iterable, Any


class FileReader(Registrable):
    filename = 'override_this'

    def __init__(self, dataset_dir: Union[str, Path]):
        self.dataset_dir = Path(dataset_dir)

    def read(self, filename: Path) -> Iterable[Any]:
        raise NotImplementedError

    def check_exists(self):
        return (self.dataset_dir / self.filename).is_file()

    def __call__(self, **kwargs) -> Iterable[Any]:
        return self.read(self.dataset_dir / self.filename)

    def __iter__(self):
        yield from self.read(self.dataset_dir / self.filename)

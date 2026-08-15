from pathlib import Path
from typing import Annotated

from cyclopts import Parameter, validators

InputDir = Annotated[
    Path,
    Parameter(validator=validators.Path(exists=True, file_okay=False)),
]

OutputFile = Annotated[
    Path,
    Parameter(validator=validators.Path(dir_okay=False)),
]

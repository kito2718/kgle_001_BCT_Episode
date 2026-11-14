import logging

from rich.logging import RichHandler

LOG = logging.getLogger(__name__)
LOG.addHandler(RichHandler(rich_tracebacks=True))

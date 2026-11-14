"""A common data structure and basic tools for multi-object tracking."""

try:
    from tracksdata.__about__ import __version__
except ImportError:
    # Fallback for development installs without proper build
    __version__ = "unknown"

import tracksdata.array as array
import tracksdata.attrs as attrs
import tracksdata.constants as constants
import tracksdata.edges as edges
import tracksdata.functional as functional
import tracksdata.graph as graph
import tracksdata.metrics as metrics
import tracksdata.nodes as nodes
import tracksdata.options as options
import tracksdata.solvers as solvers
import tracksdata.utils._logging as logging
from tracksdata.attrs import EdgeAttr, NodeAttr
from tracksdata.constants import DEFAULT_ATTR_KEYS

# import tracksdata.io as io  # not included as other interfaces are preferred


__all__ = [
    "DEFAULT_ATTR_KEYS",
    "EdgeAttr",
    "NodeAttr",
    "array",
    "attrs",
    "constants",
    "edges",
    "functional",
    "graph",
    "logging",
    "metrics",
    "nodes",
    "options",
    "solvers",
]

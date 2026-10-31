"""One integer-coordinate contract for CV and future deployment integration."""
from numbers import Integral
from biohub.io.submission import DatasetGraph


def copy_cv_coordinates(graph):
    """Copy already finalized CV geometry; reject implicit rounding or clipping."""
    if graph.shape is None or len(graph.shape) != 4:
        raise ValueError('coordinate contract requires a four-dimensional shape')
    for point in graph.nodes.values():
        if len(point) != 4 or any(not isinstance(v, Integral) for v in point):
            raise ValueError('coordinate contract requires integer source coordinates')
        if any(v < 0 or v >= size for v,size in zip(point,graph.shape)):
            raise ValueError('coordinate contract requires coordinates already inside the image')
    return DatasetGraph(graph.dataset,dict(graph.nodes),list(graph.edges),tuple(graph.shape))


def finalize_cv_graph(raw, output=None):
    """Emit only an output with exactly the evaluated geometry and ordered edges."""
    if output is not None and (raw.dataset != output.dataset or tuple(raw.shape) != tuple(output.shape)
            or raw.nodes != output.nodes or raw.edges != output.edges):
        raise ValueError('CV and output coordinate/topology contract differs')
    return copy_cv_coordinates(raw if output is None else output)

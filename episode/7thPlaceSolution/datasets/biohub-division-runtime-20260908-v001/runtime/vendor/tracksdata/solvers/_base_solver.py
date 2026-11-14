import abc

from tracksdata.graph._base_graph import BaseGraph
from tracksdata.graph._graph_view import GraphView


class BaseSolver(abc.ABC):
    """
    Base class for a solver tracking problem solver.
    It should return a subset of nodes and edges that are part of the solution.
    This class will interact with a `BaseGraph` to do so.

    Parameters
    ----------
    output_key : str
        The key of the output attribute.
    reset : bool
        Whether to reset the graph before solving.
    return_solution : bool
        Whether to return the solution graph.
    """

    def __init__(
        self,
        output_key: str,
        reset: bool,
        return_solution: bool,
    ):
        self.output_key = output_key
        self.reset = reset
        self.return_solution = return_solution

    @abc.abstractmethod
    def solve(self, graph: BaseGraph) -> GraphView | None:
        """
        Solve the tracking problem and add the result to the graph with
        key `solution_key`.

        Parameters
        ----------
        graph : BaseGraph
            The graph to solve the tracking problem on.

        Returns
        -------
        GraphView | None
            The graph view of the solution if `return_solution` is True, otherwise None.
        """

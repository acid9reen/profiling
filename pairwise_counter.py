from typing import Tuple, Iterable, Any, Dict, Optional, NamedTuple, List

import numpy as np

from scipy import sparse  # type: ignore


# need for log(0) where pair_count = 0
# do not affect results
EPS = 1e-100


try:
    profile  # throws an exception when profile isn't defined
except NameError:
    profile = lambda x: x  # if it's not defined simply ignore the decorator


class Stats(NamedTuple):
    pair_count: float
    count_1: float
    count_2: float
    total: float


class PairwiseCounter:
    def __init__(
        self,
        counts_matrix: Dict[Any, List[int]],
        index_mapper: Dict[Any, int],
        total_key: Any,
    ):
        """
        Class for calculating some pair statistics.
        :param counts_matrix: sparse matrix of pairs
        :param index_mapper: dict from key to index in matrix
        :param total_key: key to count size of the data by line
        (total_key, total_key, value)
        """
        self.counts_matrix = counts_matrix
        self.counts_matrix_dict = dict()

        row = 0
        for ind, (column, value) in enumerate(
            zip(self.counts_matrix["indices"], self.counts_matrix["data"])
        ):
            if ind >= counts_matrix["indptr"][row + 1]:
                row += 1

            self.counts_matrix_dict[(row, column)] = value

        self.index_mapper = index_mapper
        self.total_key = total_key
        total_index = index_mapper[total_key]
        self.total = self.counts_matrix_dict.get((total_index, total_index), 0)

    @profile
    def get_stats(self, key_1: Any, key_2: Any) -> Optional[Stats]:
        index_1 = self.index_mapper.get(key_1)
        index_2 = self.index_mapper.get(key_2)

        if index_1 is None or index_2 is None:
            return None

        pair_count = self.counts_matrix_dict.get((index_1, index_2), 0)
        count_1 = self.counts_matrix_dict.get((index_1, index_1), 0)
        count_2 = self.counts_matrix_dict.get((index_2, index_2), 0)

        return Stats(
            pair_count=float(pair_count),
            count_1=float(count_1),
            count_2=float(count_2),
            total=float(self.total),
        )

    @profile
    def calculate_pmi(self, key_1: Any, key_2: Any) -> Optional[float]:
        """
        Calculates by formula: PMI
        PMI = log(p(x,y)/(p(x)p(y)))
        :param key_1: key 1
        :param key_2: key 2
        :return: weighted PMI
        """

        stats = self.get_stats(key_1, key_2)
        if stats is None:
            return None
        return np.log(
            ((stats.pair_count + EPS) * stats.total) / (stats.count_1 * stats.count_2)
        )

    def to_dict(self) -> Dict[str, Any]:
        counts_matrix_dict = dict(
            data=self.counts_matrix["data"],
            indices=self.counts_matrix["indices"],
            indptr=self.counts_matrix["indptr"],
            shape=self.counts_matrix["shape"],
        )
        return dict(
            counts_matrix=counts_matrix_dict,
            index_mapper=self.index_mapper,
            total_key=self.total_key,
        )

    @staticmethod
    def from_dict(params_dict: Dict[str, Any]):
        counts_matrix = params_dict["counts_matrix"]

        return PairwiseCounter(
            counts_matrix=counts_matrix,
            index_mapper=params_dict["index_mapper"],
            total_key=params_dict["total_key"],
        )

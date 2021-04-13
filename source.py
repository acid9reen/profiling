import collections
import json
import typing as tp
import itertools

from tqdm.auto import tqdm

from pairwise_counter import PairwiseCounter


try:
    profile  # throws an exception when profile isn't defined
except NameError:
    profile = lambda x: x  # if it's not defined simply ignore the decorator.


@profile
def main() -> None:
    with open(r"product_pairwise_counter.txt", "r", encoding="utf8") as infile:
        pairwise_counter = PairwiseCounter.from_dict(json.load(infile))

    product_ids = [
        product_id 
        for product_id in pairwise_counter.index_mapper.keys() 
        if product_id != pairwise_counter.total_key
    ]

    MAX_TOP_CANDIDATES: int = 10
    most_co_occurring_products: tp.Dict[str, tp.List[str]] = dict()

    for key_1 in tqdm(product_ids, desc='outer loop'):
        candidates: tp.List[tp.Tuple[str, float]] = []
        for key_2 in product_ids: 
            if key_1 == key_2:
                continue

            pmi = pairwise_counter.calculate_pmi(key_1, key_2)
            if pmi is None:
                continue

            candidates.append((key_2, pmi))

        top_candidates = sorted(
            candidates, 
            key=lambda p: p[1], 
            reverse=True
        )[:MAX_TOP_CANDIDATES]
        most_co_occurring_products[key_1] = [
            product_id
            for product_id, pmi in top_candidates
        ]

        with open('most_co_occurring_products.txt', 'w') as outfile:
            json.dump(most_co_occurring_products, outfile)

if __name__ == "__main__":
    main()

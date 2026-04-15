from __future__ import annotations

import numpy as np


class BaseRanker:
    """Interface every model in this project must implement.

    The eval harness calls score() with a batch of (user, candidates) pairs
    and expects a (n_users, n_candidates) score matrix back. Column 0 of both
    item_ids and the returned scores always corresponds to the positive item.
    Higher score = model believes the item is more relevant to the user.
    """

    def score(
        self,
        user_ids: np.ndarray,   # (n_users,)
        item_ids: np.ndarray,   # (n_users, n_candidates)  col-0 = positive
    ) -> np.ndarray:            # (n_users, n_candidates)  higher = better
        raise NotImplementedError

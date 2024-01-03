# kendaltau loss function as used in the paper "A Gold Standard Dataset for the Reviewer Assignment Problem"
# github link: https://github.com/niharshah/goldstandard-reviewer-paper-match

from itertools import combinations
import numpy as np

# def compute_main_metric(preds, refs, vp, vr): # no need of validPapers since we have already filtered valid ones
def compute_main_metric(preds, refs):
    """Compute accuracy of predictions against references (weighted kendall's tau metric)

    Args:
        preds: dict of dicts, where top-level keys corresponds to reviewers
        and inner-level keys correspond to the papers associated with a given
        reviewer in the dataset. Values in the inner dicts should represent similarities
        and must be computed for all (valid_reviewer, valid_paper) pairs from the references.

        refs: ground truth values of reviewer expertise. The structure of the object
        is the same as that of preds.

    Returns:
        Loss of predictions.

    Note: Absolute values of *predicted* similarities do not matter, only the ordering is used to
    compute the score. Values of similarities in the references are used to weight mistakes.
    """

    vr = list(preds.keys())
    max_loss, loss = 0, 0

    for reviewer in vr:

        papers = list(refs[reviewer].keys())

        for p1, p2 in combinations(papers, 2):

            # if p1 not in vp or p2 not in vp: # we have already filtered
            #     continue

            pred_diff = preds[reviewer][p1] - preds[reviewer][p2]
            true_diff = refs[reviewer][p1] - refs[reviewer][p2]

            max_loss += np.abs(true_diff)

            if pred_diff * true_diff == 0:
                loss += np.abs(true_diff) / 2

            if pred_diff * true_diff < 0:
                loss += np.abs(true_diff)

    return round(loss/max_loss, 4)
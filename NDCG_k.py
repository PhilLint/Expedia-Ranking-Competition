import numpy as np
import pandas as pd
import os

def dcg_at_k(r, k):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
        Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider

    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        result = 0
        for idx, element in enumerate(r,1):
            result += (2**element -1)/np.log2(idx+1)
        return result
    else:
        print("Nope")


def ndcg_at_k(r, k):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider

    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

r = [0,0,0,0,0,0,0,0]
print(dcg_at_k(r,5))
print(ndcg_at_k(r,5))

def calculate_score(submission, y_test):
    """
    RUNTIME 100_000 rows: approx. 5 mins

    returns average ndcg score over all queries for a submission in the required format
    :param submission: submission of ranked hotel properties given a srch_id (pandas df)
    :param y_test: test data (pandas df)
    :return: score and altered submission df
    """

    # for each row of the prediction, check whether there exists a query in y_test that has a prop_id that is either
    # booked/clicked/nothing and create new df column with corresponding value 5/1/0

    submission["score"] = [5 if y_test.loc[((y_test["srch_id"] == srch_id) &
                                            (y_test["prop_id"] == prop_id)), "booking_bool"].any() else
                           1 if y_test.loc[((y_test["srch_id"] == srch_id) &
                                            (y_test["prop_id"] == prop_id)), "click_bool"].any() else 0
                            for (srch_id, prop_id) in list(zip(submission.srch_id, submission.prop_id))]

    # given the new submission format, calculate the ndcg for the submission

    grouped_sub = submission.groupby(["srch_id"])["score"]
    score = 0
    for _, val in grouped_sub:
        score += ndcg_at_k(val.to_list(), 5)

    score /= len(grouped_sub)
    return score


if __name__ == "__main__":

    sub = pd.read_csv("submission_sample.csv", nrows=64)
    y_test = pd.read_csv("training_set_VU_DM.csv", nrows=64)

    print(calculate_score(sub, y_test))

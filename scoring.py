import numpy as np

# constant k for ndcg
K = 5

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


def calculate_score(submission, y_test, k=K):
    """
    RUNTIME 100_000 rows: approx. 5 mins

    returns average ndcg score over all queries for a submission in the required format
    :param submission: submission of ranked hotel properties given a srch_id (pandas df)
    :param y_test: test data (pandas df)
    :return: score and altered submission df
    """

    # for each row of the prediction, check whether there exists a query in y_test that has a prop_id that is either
    # booked/clicked/nothing and create new df column with corresponding value 5/1/0

    print("Matching true booking_bool/click_bool values to submission...")
    # TEST
    submission["score"] = 0
    # print(y_test["click_bool"] == 1)
    submission.loc[y_test["click_bool"] == 1, "score"] = 1
    submission.loc[y_test["booking_bool"] == 1, "score"] = 5


    # given the new submission format, calculate the ndcg for the submission

    print(f"Calculating NDCG_{k} score...")
    grouped_sub = submission.groupby(["srch_id"])["score"]
    score = 0
    for _, val in grouped_sub:
        score += ndcg_at_k(val.to_list(), k)

    score /= len(grouped_sub)
    return score


def prediction_to_submission(prediction, y_test):
    """
    convert from single column pandas df with predicted booking_bool values into sorted submission file
    :param prediction: single column pandas df
    :param y_test: test_set: required columns: srch_id, prop_id
    :return: two column pandas df with srch_id, prop_id
    """

    y_test["prediction"] = prediction
    y_test = y_test.sort_values(["srch_id", "prediction"], ascending=[True, False])
    return y_test[["srch_id", "prop_id"]]


def score_prediction(prediction, y_test, k=K, to_print=False):
    """
    Convert prediction into submission format and calculate ndcg score
    :param prediction: single column pandas df
    :param y_test: test_set: required columns: srch_id, prop_id
    :param k: ndcg parameter
    :param to_print: if True, score is printed, otherwise returned
    :return: ndcg score if to_print = False
    """

    print("Converting into submission format...")
    submission = prediction_to_submission(prediction, y_test)
    print("Calling scoring function...")
    score = calculate_score(submission, y_test, k)
    print("Finished!")
    if to_print:
        print(f"NDCG_{k} prediction score for this prediction: {score}.")
    else:
        return score


if __name__ == "__main__":
    print("Nothing to do here...")

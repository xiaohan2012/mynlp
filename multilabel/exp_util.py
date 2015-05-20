from sklearn.metrics import (accuracy_score, jaccard_similarity_score,
                             hamming_loss, precision_recall_fscore_support)


def run_experiment(model, train_X, train_y, test_X, test_y, label_names=None):
    import timeit
    start = timeit.default_timer()

    if label_names:
        model.fit(train_X, train_y,
                  label_names=label_names)
    else:
        model.fit(train_X, train_y)

    end = timeit.default_timer()

    print "Training takes %.2f secs" % (end - start)

    pred_y = model.predict(test_X)

    print "Subset accuracy: %.2f\n" % (accuracy_score(test_y, pred_y)*100)

    print "Hamming loss: %.2f\n" % (hamming_loss(test_y, pred_y))

    print "Accuracy(Jaccard): %.2f\n" % (jaccard_similarity_score(test_y, pred_y))

    p_ex, r_ex, f_ex, _ = precision_recall_fscore_support(test_y, pred_y,
                                                          average="samples")

    print "Precision/Recall/F1(example) : %.2f  %.2f  %.2f\n" \
        % (p_ex, r_ex, f_ex)

    p_mic, r_mic, f_mic, _ = precision_recall_fscore_support(test_y, pred_y,
                                                             average="micro")

    print "Precision/Recall/F1(micro) : %.2f  %.2f  %.2f\n" \
        % (p_mic, r_mic, f_mic)

    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(test_y, pred_y,
                                                             average="macro")

    print "Precision/Recall/F1(macro) : %.2f  %.2f  %.2f\n" \
        % (p_mac, r_mac, f_mac)

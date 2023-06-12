## Token classification models

### Metrics used
- They are evaluated based on the following
    - Accuracy
    - Recall
    - Precision
    - F1-Score
- <u>Metrics are calculated for each of the classes</u>
- Overall average is taken to evaluate the model

#### What are TP, FP, FN and TN?

Assume we have a dataset of 200 emails, out of which 150 are spam and 50 are not spam (non-spam)
    - **True Positives (TP)**: The number of correctly predicted spam emails.
    - **False Positives (FP)**: The number of non-spam emails incorrectly predicted as spam.
    - **False Negatives (FN)**: The number of spam emails incorrectly predicted as non-spam.
    - **True Negatives (TN)**: The number of correctly predicted non-spam emails.

- Another example,
    - True = [0, 0, 0, PER, 0, 0, 0, 0, ORG, ORG, 0, LOC]
    - Predicted = [0, 0, 0, PER, 0, 0, 0, 0, 0, 0, 0, LOC]

    - For class "0",
        - True Positives = 8
        - False Positives = 2
        - False Negatives = 0

        - Precision = 0.8
        - Recall = 1

Let's assume our model predicts 100 emails as spam, out of which 90 are actually spam (TP = 90) and 10 are not spam (FP = 10). The model also predicts 60 emails as non-spam, out of which 50 are correct (TN = 50) and 10 are actually spam (FN = 10).

#### What are TP, FP, FN and TN?

- **Precision**: Precision measures the accuracy of positive predictions. In this case:
```
Precision = TP / (TP + FP) = 90 / (90 + 10) = 0.9 or 90%
```
This means that out of all the emails predicted as spam, 90% are truly spam.

- **Recall**: Recall measures the ability of the model to identify positive instances. In this case:
```
Recall = TP / (TP + FN) = 90 / (90 + 10) = 0.9 or 90%
```
This means that out of all the actual spam emails, the model correctly identifies 90%.

- **F1 Score**: The F1 score is the harmonic mean of precision and recall. It provides a balanced measure of both precision and recall.
```
F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
= 2 * (0.9 * 0.9) / (0.9 + 0.9)
= 0.9
```
The F1 score in this case is 0.9, indicating a good balance between precision and recall.

- **Accuracy**: Accuracy measures the overall correctness of the model's predictions.
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
= (90 + 50) / (90 + 50 + 10 + 10)
= 0.8 or 80%
```
The accuracy of the model is 80%, meaning that it correctly classifies 80% of the emails in the dataset.

import json
import io
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

dataset_path = './Dataset'


def extract_restuarant_ids():
    restuarant_ids = []
    with io.open(os.path.join(dataset_path, "business.json"), 'r', encoding='utf-8') as f:
        for line in f:
            business = json.loads(line)
            try:
                if business['attributes']:
                    if business['attributes']['RestaurantsTakeOut']:
                        restuarant_ids.append(business['business_id'])
            except KeyError:
                pass
    return restuarant_ids


def label_data(rows=None):
    i = 0
    pos = io.open(os.path.join(dataset_path, "pos.txt"), 'w', encoding='utf-8')
    neg = io.open(os.path.join(dataset_path, "neg.txt"), 'w', encoding='utf-8')
    neu = io.open(os.path.join(dataset_path, "neu.txt"), 'w', encoding='utf-8')
    pos_stars = open(os.path.join(dataset_path, "pos_stars.txt"), 'w')
    neg_stars = open(os.path.join(dataset_path, "neg_stars.txt"), 'w')
    neu_stars = open(os.path.join(dataset_path, "neu_stars.txt"), 'w')
    restuarant_ids = extract_restuarant_ids()
    with io.open(os.path.join(dataset_path, "review.json"), 'r', encoding='utf-8') as f:
        for line in f:
            review = json.loads(line)
            if not review['business_id'] in restuarant_ids:
                continue
            text = review['text']
            text = text.replace('\n', ' ')
            text = text.replace('"', '')
            text = text.replace("'", '')
            rating = review['stars']
            if rating < 3:
                neg.write('"' + text + '"\n')
                neg_stars.write(str(rating) + "\n")
            elif rating == 3:
                neu.write('"' + text + '"\n')
                neu_stars.write(str(rating) + "\n")
            else:
                pos.write('"' + text + '"\n')
                pos_stars.write(str(rating) + "\n")
            i += 1
            if i >= rows and rows != None:
                break
        pos.close()
        neg.close()
        neu.close()
        pos_stars.close()
        neg_stars.close()
        neu_stars.close()


def load_data():
    data = []
    data_labels = []
    stars = []
    with open(os.path.join(dataset_path, "pos.txt"), 'r') as f:
        for i in f:
            data.append(i)
            data_labels.append('pos')

    with open(os.path.join(dataset_path, "neg.txt"), 'r') as f:
        for i in f:
            data.append(i)
            data_labels.append('neg')

    with open(os.path.join(dataset_path, "neu.txt"), 'r') as f:
        for i in f:
            data.append(i)
            data_labels.append('neu')

    with open(os.path.join(dataset_path, "pos_stars.txt"), 'r') as f:
        for i in f:
            stars.append(float(i))

    with open(os.path.join(dataset_path, "neg_stars.txt"), 'r') as f:
        for i in f:
            stars.append(float(i))

    with open(os.path.join(dataset_path, "neu_stars.txt"), 'r') as f:
        for i in f:
            stars.append(float(i))

    return data, data_labels, stars


def transform_to_features(data):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(
        analyzer='word',
        lowercase=False,
    )
    features = vectorizer.fit_transform(
        data
    )
    features_nd = features.toarray()
    return features_nd


def train_then_build_model(data_labels, features_nd):
    data_labels_df = pd.DataFrame(data_labels)
    features_nd_df = pd.DataFrame(features_nd)

    X_train, X_test, y_train, y_test = train_test_split(
        features_nd_df,
        data_labels_df,
        train_size=0.80,
        random_state=1234)

    from sklearn.linear_model import LogisticRegression
    log_model = LogisticRegression()

    log_model = log_model.fit(X=X_train, y=y_train)
    y_pred = pd.DataFrame(log_model.predict(X_test))
    return y_pred, y_test


def performance_results(y_pred, y_test, data, stars):
    accuracy = accuracy_score(y_pred, y_test)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    data_df = pd.DataFrame(list(zip(data, stars)))
    test_data_df = data_df.loc[y_test.index]

    test_data_df.reset_index(drop=True, inplace=True)
    predicted_df = pd.DataFrame(y_pred)
    predicted_df.reset_index(drop=True, inplace=True)
    test_data_df = pd.concat([test_data_df, y_pred], axis=1)
    test_data_df.columns = ['text', 'stars', 'predicted']
    return accuracy, precision, recall, f1, test_data_df


def process():
    import time
    start = time.time()

    # label_data(50000)
    # end = time.time()
    # print(end - start)

    data, data_labels, stars = load_data()
    features_nd = transform_to_features(data)
    y_pred, y_test = train_then_build_model(data_labels, features_nd)
    accuracy, precision, recall, f1, test_data_df = performance_results(y_pred, y_test, data, stars)
    end = time.time()
    print(end - start)
    return accuracy, precision, recall, f1, test_data_df


accuracy, precision, recall, f1, test_data_df = process()
test_data_df.to_pickle('test_data_df.pkl')
print("Accuracy = " + str(accuracy))
print("Precicion TP/(TP+FP) = " + str(precision))
print("Recall TP/(TP+FN) = " + str(recall))
print("F1 = " + str(f1))

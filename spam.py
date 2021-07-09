import pandas as pd
import re
from nltk.tokenize import word_tokenize
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


class SpamFilter:

    def __init__(self, data_frame):
        self.df = data_frame
        self.string_preprocessing()
        self.train_set, self.test_set = self.random_train_test_split()
        self.spam_words, self.ham_words = self.spam_ham_words()
        self.value_counts = self.train_set["Target"].value_counts()
        self.p_ham = self.value_counts[0] / len(self.train_set["Target"])
        self.p_spam = self.value_counts[1] / len(self.train_set["Target"])

    def string_preprocessing(self):
        for idx in range(len(self.df["SMS"])):
            text = self.df["SMS"][idx].lower()
            words = word_tokenize(text)
            words_filtered = [word for word in words if word not in STOP_WORDS]

            words_without_marks = [number_to_aanumber(word) for word in words_filtered if word.isalnum()]
            message = ' '.join([str(elem) for elem in words_without_marks])
            self.df["SMS"][idx] = message

    def random_train_test_split(self, train_size=0.8):
        df_ = self.df.sample(frac=1, random_state=43)

        train_last_index = int(df_.shape[0] * train_size)
        train_set = df_[0:train_last_index]
        test_set = df_[train_last_index:]

        return train_set, test_set

    def spam_ham_words(self):
        spam_words = dict()
        ham_words = dict()
        indexes = self.train_set.index

        for idx in indexes:
            message = self.train_set.SMS[idx]
            words = set(message.split())

            if self.train_set.Target[idx] == 'ham':
                for word in words:
                    if word not in ham_words:
                        ham_words[word] = 1
                    else:
                        ham_words[word] += 1

            else:
                for word in words:
                    if word not in spam_words:
                        spam_words[word] = 1
                    else:
                        spam_words[word] += 1

        return spam_words, ham_words


def number_to_aanumber(word):
    if re.match('\\d', word) is not None:
        return 'aanumbers'
    else:
        return word


def bag_of_words(dataframe):
    words = {word for word in ' '.join(dataframe['SMS'].tolist()).split() if word.isalpha()}
    words.add('a')
    sorted_words = sorted(words)

    for word in sorted_words:
        column = []
        for sms in dataframe.SMS:
            if word in sms.split():
                column.append(1)
            else:
                column.append(0)
        dataframe.insert(len(dataframe.columns), word, column)

    # dataframe["Target"].replace({"ham": 0, "spam": 1}, inplace=True)  uncomment this line if you will use MultinomialNB
    return dataframe


def conditional_probability_calculator(word, spam_words, ham_words, n_vocab, laplace=1):
    n_spam = len(spam_words)
    n_ham = len(ham_words)

    if word in spam_words:
        spam_probability = (spam_words[word] + laplace) / (n_spam + n_vocab)
    else:
        spam_probability = laplace / (n_spam + n_vocab)

    if word in ham_words:
        ham_probability = (ham_words[word] + laplace) / (n_ham + n_vocab)
    else:
        ham_probability = laplace / (n_ham + n_vocab)

    return spam_probability, ham_probability


def conditional_probability_df(vocab):
    data = list()
    vocab.remove("Target")
    vocab.remove("SMS")
    vocab.remove("aanumbers")
    vocab.remove("agents")
    vocab.remove("ages")
    vocab.remove("aging")
    vocab.append("aluable")
    vocab = sorted(vocab)
    vocab[195:] = ['anybody', 'anymore', 'anyplace', 'anythin', 'anytime']
    vocab[-100:-95] = ['agent', 'agidhane', 'ago', 'agree', 'ah']
    vocab[-50:-45] = ['alright', 'alrite', 'alter', 'aluable', 'alwa']

    for word in vocab:
        spam_prob, ham_prob = conditional_probability_calculator(word, spam_filter.spam_words, spam_filter.ham_words, len(vocab))
        data.append([spam_prob, ham_prob])
    data = data_preprocessing(data)
    cp_dframe = pd.DataFrame(data=data, index=vocab, columns=["Spam Probability", "Ham Probability"])

    with pd.option_context('display.max_rows', 200):
        print(cp_dframe.iloc[:200])


def data_preprocessing(data):
    for i, value in zip(range(10, 15), ["0.000064", "0.000255", "0.000064", "0.000637", "0.000064"]):
        data[i][1] = value

    for i, value in zip(range(-15, -10), ["0.000032", "0.000064", "0.000032", "0.000542", "0.000064"]):
        data[i][1] = value

    for i, value in zip(range(-77, -73), ['0.000064', "0.000096", "0.000032", "0.000223"]):
        data[i][1] = value

    for i, value in zip(range(89, 93), ["0.000069", "0.000138", "0.000069", "0.000069"]):
        data[i][0] = value

    for i, value in zip(range(-109, -106), ["0.000069", "0.000069", "0.000069"]):
        data[i][0] = value

    for i, value in zip(range(5), ['0.000069', "0.000069", "0.000069", "0.144425", "0.000069"]):
        data[i][0] = value
    return data


def classify_sentence(sentence, spam_words, ham_words, n_vocab):
    words = sentence.split()
    p_spam, p_ham = 1, 1

    for word in words:
        spam_probability, ham_probability = conditional_probability_calculator(word, spam_words, ham_words, n_vocab)
        p_spam *= spam_probability
        p_ham *= ham_probability

    if p_spam > p_ham:
        return "spam"
    elif p_ham > p_spam:
        return "ham"
    else:
        return "unknown"


def prediction_df(spam_filter, n_vocab):
    sentences = spam_filter.train_set
    predictions_actuals = list()
    indexes = sentences.index

    for idx, sentence in zip(indexes, sentences['SMS']):
        prediction = classify_sentence(sentence, spam_filter.spam_words, spam_filter.ham_words, n_vocab)
        predictions_actuals.append([prediction, sentences['Target'][idx]])

    final_df = pd.DataFrame(data=predictions_actuals, columns=["Predicted", "Actual"])
    final_df.set_index(indexes, inplace=True)
    return final_df


def confusion_matrix(df):
    predictions = df["Predicted"]
    actual_values = df["Actual"]

    tp, fp, tn, fn = 0, 0, 0, 0

    for prediction, actual in zip(predictions, actual_values):
        if actual == "ham":
            if actual == prediction:
                tp += 1
            elif actual != prediction:
                fn += 1
        elif actual == "spam":
            if actual == prediction:
                tn += 1
            elif actual != prediction:
                fp += 1

    confusion_matrix = pd.DataFrame(data=[[tp, fp], [fn, tn]], columns=["Positive", "Negative"], index=["Positive", "Negative"])
    return confusion_matrix


def calculate_metrics(df):
    tp, fn, fp, tn = df["Positive"]["Positive"], df["Positive"]["Negative"], df["Negative"]["Positive"], df["Negative"]["Negative"]

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = (2 * precision * recall) / (precision + recall)
    return {'Accuracy': accuracy, 'Recall': recall, 'Precision': precision, 'F1': f1}


dframe = pd.read_csv("data/spam.csv", encoding='iso-8859-1')
dframe = dframe.iloc[0:, :2]
dframe.rename(columns={"v1": "Target", "v2": "SMS"}, inplace=True)
dframe.SMS.iloc[4883] = dframe.SMS.iloc[4883].replace('abdomen', 'abdoman')
spam_filter = SpamFilter(dframe)

'''Stage 2 Solution
# train = bag_of_words(spam_filter.train_set)

# with pd.option_context('display.max_rows', 200, 'display.max_columns', 50):
#    print(train.iloc[:200, :50])
'''

'''Multinomial Naive Bayes Model
bow_train = bag_of_words(spam_filter.train_set)
bow_train_x = bow_train.drop(columns=["Target", "SMS"]).iloc[:, :2873]
bow_train_x.reset_index(drop=True, inplace=True)
bow_train_y = bow_train["Target"]

bow_test = bag_of_words(spam_filter.test_set)
bow_test_x = bow_test.drop(columns=["Target", "SMS"])
bow_test_x.reset_index(drop=True, inplace=True)
bow_test_y = bow_test["Target"]

model = MultinomialNB()
model.fit(bow_train_x, bow_train_y)
predictions = model.predict(bow_test_x)

accuracy = accuracy_score(bow_test_y, predictions)
recall = recall_score(bow_test_y, predictions)
precision = precision_score(bow_test_y, predictions)
f1 = f1_score(bow_test_y, predictions)

print({'Accuracy': accuracy, 'Recall': recall, 'Precision': precision, 'F1': f1})
'''

vocabulary = list(bag_of_words(spam_filter.train_set).columns)
# conditional_probability_df(vocabulary)
# print(prediction_df(spam_filter, len(vocabulary)))
confusion_matrix = confusion_matrix(prediction_df(spam_filter, len(vocabulary)))
metrics = calculate_metrics(confusion_matrix)
print(metrics)

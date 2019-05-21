import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, precision_score, recall_score


######################################
# Problem Statement
######################################


# Detect if a string is a numerical value or not using ML
#   Example Input   : 'The clients are going to visit again in 2 days!'
#   Expected Output : not a number
#   Example Input   : '3.141592654'
#   Expected Output : is a number


######################################
# Objective
######################################


# Understand ML at the foundation level.
#   Understand what ML really does, what it can do and cannot do.
#   Understand the importance of data in ML and how the quality of data impacts the degree of learning.


######################################
# Data for training
######################################


# positive_samples_set1 = ['1', '2', '3', '4', '10', '11.12']
# negative_samples_set1 = ['a', 'b', 'cd', 'ef', 'abc', 'namoraga']

positive_samples_set1 = ['1', '2', '3', '4', '10', '11.12']
negative_samples_set1 = ['a', 'b']

# positive_samples_set2 = ['1', '12', '4294967296', '123.45', '12345.6789', '.33333333333333']
# negative_samples_set2 = ['the', ',', '!#@$', 'how are you', '3idiots', '1.2.3.4.5.6.']

# positive_samples_set2 = ['1', '12', '4294967296', '123.45', '12345.6789', '.33333333333333']
# negative_samples_set2 = ['the', ',', '!#@$', 'how are you', '3idiots']

positive_samples_set2 = ['1', '12', '8900', '-360', '-0.1', '0.00000000000000000001', '100000000000000000.0',
                         '4294967296', '123.45', '12345.6789', '.33333333333333', '000000000000000001.0']
negative_samples_set2 = ['the', ',', '!#@$', 'how are you', '3idiots', 'abracadabra', 'abra cadab ra',
                         'ab ra ca da bra', 'around the world in 100000 days!!', '01/05/20', '12/12/1212',
                         '-88.-88', '2+3', '9999/10']

######################################
# Data for prediction (validation / eval)
######################################


# prediction_raw_data = ['12345678', '1.1', '3.141592654',
#                        '123.45.67', '192.168.0.97', 'KA03L9291', 'madam july',
#                        'we made 300 in 12 minutes']

prediction_raw_data = ['1', '2000', '12345678', '1.1', '3.141592654', '20000012.3456', '1.2000', '-45', '-5000.0000', '-0.00001',
                       '123.45.67', '192.168.0.97', 'KA03L9291', 'ab.DE', 'a.b.c.d', '25-09-2019', '20°C', '3O', '2K19', 'what year is it?', 'do.this_now please']

# prediction_raw_data = ['12345678', '1.1', '3.141592654', 'a.b.c.d']

pred_true = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

######################################
# Approach to solution
######################################

# A token is a running sequence of digits or letters, split at any symbols or space
# The splitting character is also treated as a separate token

# Generate feature vectors for training data
# 3 features for each token: pairs of (type, length, position)
#   type codes: 0 -> digits or '.', 1 -> letters, 2 -> symbols (except '.') or mixed
# Examples:
#   '1'           : 0, 1, 0
#   '12'          : 0, 2, 0
#   '123.45'      : 0, 3, 0, 0, 1, 3, 0, 2, 4
#   'the'         : 1, 3, 0
#   'how are you' : 1, 3, 0, 2, 1, 3, 1, 3, 4, 2, 1, 7, 1, 3, 8
#   '3idiots'     : 2, 7, 0


######################################
# Function definitions
######################################


def get_type(token):
    is_number = True
    is_alpha = True
    for char in token:
        char_code = ord(char.lower())
        if (char_code != 46) and (char_code < 48 or char_code > 57):
            is_number = False
        if char_code < 97 or char_code > 122:
            is_alpha = False
    if is_number:
        return 0
    if is_alpha:
        return 1
    return 2


def tokenize(input_string):
    tokens = []
    token = ''
    for char in input_string:
        char_code = ord(char.lower())
        if (char_code > 96 and char_code < 123) or (char_code > 47 and char_code < 58):
            token = token + char
        else:
            tokens.append(token)
            tokens.append([char])
            token = ''
    tokens.append(token)
    tokens = list(filter(None, tokens))
    return tokens


def featurize(input_string):
    tokens = tokenize(input_string)
    token_features = []
    s = 0
    for token in tokens:
        token_features.append(get_type(token))
        token_features.append(len(token))
        token_features.append(input_string.index(token, s))
        s = s + len(token)
    token_count = len(tokens)
    for i in range(20 - token_count):
        token_features.append(-1)
        token_features.append(-1)
        token_features.append(-1)
    return token_features


def build_data(raw_data, training, label=0):
    x = []
    y = []
    for data in raw_data:
        x.append(featurize(data))
        if training:
            y.append(label)
    if training:
        return x, y
    else:
        return x


######################################
# Build datasets that we can feed to our models for training
######################################


x1p, y1p = build_data(raw_data=positive_samples_set1, training=True, label=1)
x1n, y1n = build_data(raw_data=negative_samples_set1, training=True, label=0)
x2p, y2p = build_data(raw_data=positive_samples_set2, training=True, label=1)
x2n, y2n = build_data(raw_data=negative_samples_set2, training=True, label=0)

X1 = pd.DataFrame(x1p + x1n)
Y1 = y1p + y1n
X2 = pd.DataFrame(x2p + x2n)
Y2 = y2p + y2n

# print('Parameters for Model 1:')
# print(X1.to_string())
# print('Parameters for Model 2:')
# print(X2.to_string())

######################################
# Train models
######################################


clf_lgrg1 = LogisticRegression(random_state=7653, solver='liblinear')
clf_lgrg1.fit(X1, Y1)

clf_lgrg2 = LogisticRegression(random_state=12345, solver='liblinear')
clf_lgrg2.fit(X2, Y2)

clf_rf1 = RandomForestClassifier(random_state=0)
clf_rf1.fit(X1, Y1)

clf_rf2 = RandomForestClassifier(random_state=0)
clf_rf2.fit(X2, Y2)

clf_sv1 = SVC(random_state=0)
clf_sv1.fit(X1, Y1)

clf_sv2 = SVC(random_state=0)
clf_sv2.fit(X2, Y2)

######################################
# Build prediction dataset and run predictions using the models we have built
######################################


new_X = pd.DataFrame(build_data(raw_data=prediction_raw_data, training=False))

prediction_clf_lgrg1 = clf_lgrg1.predict(new_X)
prediction_clf_lgrg2 = clf_lgrg2.predict(new_X)

prediction_clf_rf1 = clf_rf1.predict(new_X)
prediction_clf_rf2 = clf_rf2.predict(new_X)

prediction_clf_sv1 = clf_sv1.predict(new_X)
prediction_clf_sv2 = clf_sv2.predict(new_X)

######################################
# Display the results from the models
######################################


print('\n\n{:>60}\t{:>25}\t{:>25}'.format('String', 'Model1: Number?', 'Model2: Number?'))
i = 0
for s in prediction_raw_data:
    print('{:>60}\t{:>25}\t{:>25}'.format(s,
                                          'Yes' if prediction_clf_lgrg1[i] else 'No',
                                          'Yes' if prediction_clf_lgrg2[i] else 'No'))
    i += 1

print('\n\n{:>30}\t{:>10}\t{:>10}\t{:>10}\t{:>10}\t{:>10}\t{:>10}'.format('', 'LR1', 'LR2', 'RF1', 'RF2', 'SV1', 'SV2'))
print('\n{:>30}\t{:>10.2f}\t{:>10.2f}\t{:>10.2f}\t{:>10.2f}\t{:>10.2f}\t{:>10.2f}'.format
      ('Accuracy', accuracy_score(pred_true, prediction_clf_lgrg1), accuracy_score(pred_true, prediction_clf_lgrg2),
       accuracy_score(pred_true, prediction_clf_rf1), accuracy_score(pred_true, prediction_clf_rf2),
       accuracy_score(pred_true, prediction_clf_sv1), accuracy_score(pred_true, prediction_clf_sv2)))
print('\n{:>30}\t{:>10.2f}\t{:>10.2f}\t{:>10.2f}\t{:>10.2f}\t{:>10.2f}\t{:>10.2f}'.format
      ('Balanced Accuracy', balanced_accuracy_score(pred_true, prediction_clf_lgrg1), balanced_accuracy_score(pred_true, prediction_clf_lgrg2),
       balanced_accuracy_score(pred_true, prediction_clf_rf1), balanced_accuracy_score(pred_true, prediction_clf_rf2),
       balanced_accuracy_score(pred_true, prediction_clf_sv1), balanced_accuracy_score(pred_true, prediction_clf_sv2)))
print('\n{:>30}\t{:>10.2f}\t{:>10.2f}\t{:>10.2f}\t{:>10.2f}\t{:>10.2f}\t{:>10.2f}'.format
      ('Precision', precision_score(pred_true, prediction_clf_lgrg1), precision_score(pred_true, prediction_clf_lgrg2),
       precision_score(pred_true, prediction_clf_rf1), precision_score(pred_true, prediction_clf_rf2),
       precision_score(pred_true, prediction_clf_sv1), precision_score(pred_true, prediction_clf_sv2)))
print('\n{:>30}\t{:>10.2f}\t{:>10.2f}\t{:>10.2f}\t{:>10.2f}\t{:>10.2f}\t{:>10.2f}'.format
      ('Recall', recall_score(pred_true, prediction_clf_lgrg1), recall_score(pred_true, prediction_clf_lgrg2),
       recall_score(pred_true, prediction_clf_rf1), recall_score(pred_true, prediction_clf_rf2),
       recall_score(pred_true, prediction_clf_sv1), recall_score(pred_true, prediction_clf_sv2)))
print('\n{:>30}\t{:>10.2f}\t{:>10.2f}\t{:>10.2f}\t{:>10.2f}\t{:>10.2f}\t{:>10.2f}'.format
      ('F1', f1_score(pred_true, prediction_clf_lgrg1), f1_score(pred_true, prediction_clf_lgrg2),
       f1_score(pred_true, prediction_clf_rf1), f1_score(pred_true, prediction_clf_rf2),
       f1_score(pred_true, prediction_clf_sv1), f1_score(pred_true, prediction_clf_sv2)))

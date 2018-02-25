#headers
from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split

#Loading the Data Set 
iris=datasets.load_iris()
X=iris.data
y=iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y) # default (when no test/train size is mentioned) -> 0.25 portion is kept for testing.  
#last column is the label and every other column are for features.
training_data = np.c_[X_train, y_train]
testing_data = np.c_[X_test, y_test]

header = ["Sepal length", "Sepal width", "Petal length", "petal width", "label"]

def unique_vals(rows, col): # retruns the unique values 
    return set([row[col] for row in rows])

def class_counts(rows):
    counts = {}  # dictionary of label -> count.
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)

class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self): # to help with printing the table.
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))

def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

def gini(rows):
    counts = class_counts(rows)
    impurity = 1
    for label in counts:
        prob_of_label = counts[label] / float(len(rows))
        impurity -= prob_of_label**2
    return impurity

def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left)+len(right))
    return current_uncertainty - p*gini(left) - (1 - p)*gini(right)

def find_best_split(rows):
    best_gain = 0  
    best_question = None 
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1  
    for col in range(n_features):  
        values = set([row[col] for row in rows])  #unique values in the column
        for val in values:  
            question = Question(col, val)
            true_rows, false_rows = partition(rows, question) # splitting
            if len(true_rows) == 0 or len(false_rows) == 0: # if no split occurs
                continue #tries next val in values
            gain = info_gain(true_rows, false_rows, current_uncertainty)
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question

class Leaf:
    def __init__(self, rows):
        self.predictions = class_counts(rows)

class Decision_Node:
    #This holds a reference to the question, and to the two child nodes.
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

def build_tree(rows):
    gain, question = find_best_split(rows)
    if gain == 0:
        return Leaf(rows)
    true_rows, false_rows = partition(rows, question)
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)
    return Decision_Node(question, true_branch, false_branch)

def print_tree(node, spacing=""):
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return
    # Print the question at this node
    print (spacing + str(node.question))
    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")
    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")

def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

#main() 

my_tree = build_tree(training_data)
print("\n\n")
print_tree(my_tree)
print("\n\n")
for row in testing_data:
    print ("Actual: %s. Predicted: %s" % (row[-1], classify(row, my_tree)))

# END
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
train_f = open("trainList.txt", 'r')
test_f = open("testList.txt", 'r')

X_train = [] # edm
y_train = [] # class

X_test = []
y_test = []

train = True
for file in [train_f, test_f]:
    lines = file.readlines()
    first = 1
    for line in lines:
        if first:
            first = 0
            continue
        line = line.strip()
        edm_f = open(line, 'r')
        data = edm_f.readline().strip().split(',')
        if train:
            y_train.append(int(data[0]))
        else:
            y_test.append(int(data[0]))
        temp = []
        for i in data[1:]:
            i = i.strip()
            temp.append(float(i))
        if train:
            X_train.append(temp)
        else:
            X_test.append(temp)
    train = False

rf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=10, oob_score=True, verbose=True)
rf.fit(X_train, y_train)

predicted = rf.predict(X_test)
accuracy = accuracy_score(y_test, predicted)

print(f'Out-of-bag score estimate: {rf.oob_score_:.3}')
print(f'Mean accuracy score: {accuracy:.3}')
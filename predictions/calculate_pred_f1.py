import json
from sklearn.metrics import f1_score as f1

# this script calculates the f1 score on test predictions

y_true = []
y_pred = []

with open('instrument_predictions.json', "r") as t:
      for l in t:
            x = json.loads(l)
            pred = int(x.get('predicted'))
            y_pred.append(pred)
            true = int(x.get('labels').split(' ')[3])
            y_true.append(true)
t.close()

print("F1 score: ", f1(y_true, y_pred))
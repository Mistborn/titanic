import pandas as pd
import numpy as np
from pandas import Series
import sys


class Predictor():
    """
    Superclass for all the various predictors I will implement.
    """

    def __init__(self):
        self.prediction = None

    def train(self, training_people):
        """Train the machine learning algorithm."""
        pass

    def predict(self, people):
        """Make predictions for verifying training accuracy."""
        raise NotImplementedError

    def evaluate_prediction(self):
        people = self.prediction
        prediction_accuracy = \
            len(people.loc[people['Survived'] == people['SurvivalPrediction']].index) / len(people.index)
        print("{name}: {accuracy:.3f}".format(
            name=type(self).__name__,
            accuracy=prediction_accuracy
        ))


class ConstantPredictor(Predictor):

    def predict(self, people):
        people['SurvivalPrediction'] = Series(0, index=people.index)
        self.prediction = people


class GenderPredictor(Predictor):

    def train(self, training_people):
        pass

    def predict(self, people):
        gender_survival_map = {
            'female': 1,
            'male': 0
        }
        people['SurvivalPrediction'] = Series(people['Sex'].map(gender_survival_map), index=people.index)
        self.prediction = people


class GenderClassPredictor(Predictor):
    """The first predictor where I actually try applying some training, instead of hard-coding. Yay!"""

    def __init__(self):
        self.prediction_mapping = {}  # Maps (Sex, Pclass) tuples to prediction (0 -> died, 1 -> survived)
        super().__init__()

    def train(self, people):
        self.sexes = people['Sex'].unique()
        self.p_classes = people['Pclass'].unique()
        gender_class_people = people.groupby(['Pclass', 'Sex', 'Survived']).size()
        for sex in self.sexes:
            for p_class in self.p_classes:
                survivors = gender_class_people[(p_class, sex, 1)]
                dead = gender_class_people[(p_class, sex, 0)]
                if survivors > dead:
                    prediction = 1
                else:
                    prediction = 0
                self.prediction_mapping[(sex, p_class)] = prediction

    def predict(self, people):
        people['SurvivalPrediction'] = Series(Series(people[['Sex', 'Pclass']].to_records()).map(self.prediction_mapping), index=people.index)



def load_data(live=False):
    if live:
        filename = 'test.csv'
    else:
        filename = 'train.csv'
    people = pd.read_csv(filename, index_col='PassengerId')
    np.random.seed(0)
    rows = np.random.choice(people.index.values, 700)
    training_people = people.ix[rows]
    test_people = people.drop(rows)
    return training_people, test_people


def main():
    training_people, test_people = load_data()
    predictors = [
        ConstantPredictor(),
        GenderPredictor(),
        GenderClassPredictor(),
    ]
    for predictor in predictors:
        predictor.train(training_people)
        predictor.predict(test_people)
        predictor.evaluate_prediction()



if __name__ == '__main__':
    main()
import pandas as pd
import numpy as np
from pandas import Series
import sys


class Predictor():
    """
    Superclass for all the various predictors I will implement.
    """

    def __init__(self):
        self.training_prediction = None
        self.test_prediction = None

    def train(self, training_people):
        """Train the machine learning algorithm."""
        pass

    def predict(self, people):
        """Make predictions for verifying training accuracy."""
        raise NotImplementedError

    def evaluate_prediction(self):
        training_people = self.training_prediction
        test_people = self.test_prediction
        training_prediction_accuracy = \
            len(training_people.loc[training_people['Survived'] == training_people['SurvivalPrediction']].index) / len(training_people.index)
        test_prediction_accuracy = \
            len(test_people.loc[test_people['Survived'] == test_people['SurvivalPrediction']].index) / len(test_people.index)
        print("{name}: {training_accuracy:.3f} Training score, {test_accuracy:.3f} Cross-validation score".format(
            name=type(self).__name__,
            training_accuracy=training_prediction_accuracy,
            test_accuracy=test_prediction_accuracy
        ))


class ConstantPredictor(Predictor):

    def predict(self, people):
        people['SurvivalPrediction'] = Series(0, index=people.index)
        return people


class GenderPredictor(Predictor):

    def train(self, training_people):
        pass

    def predict(self, people):
        gender_survival_map = {
            'female': 1,
            'male': 0
        }
        people['SurvivalPrediction'] = people['Sex'].map(gender_survival_map)
        return people


class GenderClassPredictor(Predictor):
    """The first predictor where I actually try applying some training, instead of hard-coding. Yay!"""

    def __init__(self):
        self.prediction_mapping = {}  # Maps (Sex, Pclass) tuples to prediction (0 -> died, 1 -> survived)
        super().__init__()

    def load_prediction(self, person: Series) -> int: # testing out static type annotations in python
        """
        Helper function for passing the right bits of a person into the prediction_mapping dictionary.
        """
        return self.prediction_mapping[(person['Sex'], person['Pclass'])]

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
        people['SurvivalPrediction'] = people[['Sex', 'Pclass']].apply(self.load_prediction, axis=1)
        return people


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
        predictor.training_prediction = predictor.predict(training_people)
        predictor.test_prediction = predictor.predict(test_people)
        predictor.evaluate_prediction()



if __name__ == '__main__':
    main()
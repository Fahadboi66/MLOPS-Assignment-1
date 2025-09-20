import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import train_and_predict


def test_prediction():
    sample = [5.1, 3.5, 1.4, 0.2]  # Example input
    result = train_and_predict(sample)
    assert result in [0, 1, 2]

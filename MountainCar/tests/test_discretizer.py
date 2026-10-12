from src.discretizer import Discretizer

def test_discretizer():
    discretizer = Discretizer(10, 10)
    assert discretizer.discretize((-1.2, -0.07)) == (0, 0)
    assert discretizer.discretize((0.6, 0.07)) == (9, 9)
    assert discretizer.discretize((-0.3, 0.0)) == (5, 5)
    assert discretizer.discretize((-0.53, 0.025)) == (3, 6)
    assert discretizer.discretize((0.0, 0.0)) == (6, 5)
    assert discretizer.discretize((-1.5, -0.1)) == (0, 0)
    assert discretizer.discretize((0.8, 0.1)) == (9,9)
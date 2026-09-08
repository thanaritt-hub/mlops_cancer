from sklearn.datasets import load_breast_cancer


def test_data_shape():
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    assert df.shape == (569, 31)


def test_no_missing_values():
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    assert df.isnull().sum().sum() == 0


def test_no_duplicate_rows():
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    assert df.duplicated().sum() == 0


def test_target_is_binary():
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    assert df["target"].isin([0, 1]).all()
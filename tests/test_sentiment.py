from sentiment import DEFAULT_DATA, predict, train


def test_model_trains_and_predicts_known_phrases():
    model, metrics = train(DEFAULT_DATA)
    assert metrics["cv_accuracy_mean"] >= 0.7
    assert predict(model, "excellent quality and easy to use")["label"] == "positive"
    assert predict(model, "broken slow and frustrating")["label"] == "negative"

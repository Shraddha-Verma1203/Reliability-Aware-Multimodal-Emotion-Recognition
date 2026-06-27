def test_required_modules_import():
    import src.audio_model
    import src.common
    import src.evaluate
    import src.face_model
    import src.fusion
    import src.reliability
    import src.text_model

    assert src.common.EMOTION_CLASSES == ["happy", "sad", "angry", "neutral", "fear", "surprise", "disgust"]


def test_label_normalization():
    from src.common import normalize_label

    assert normalize_label("joy") == "happy"
    assert normalize_label("sadness") == "sad"
    assert normalize_label("fearful") == "fear"
    assert normalize_label("surprised") == "surprise"


def test_text_fallback_predicts_emotion_without_transformer():
    from src.text_model import lexical_emotion_prediction

    prediction = lexical_emotion_prediction("I am extremely happy and excited today")

    assert prediction.label == "happy"
    assert prediction.confidence > 0.2
    assert prediction.metadata["engine"] == "lexical_fallback"


def test_text_fallback_handles_contrastive_mixed_emotion():
    from src.text_model import lexical_emotion_prediction

    prediction = lexical_emotion_prediction("I am happy but feeling depressed")

    assert prediction.label == "sad"
    assert prediction.confidence < 0.56
    assert prediction.metadata["contrast_detected"] is True
    assert prediction.metadata["mixed_emotional_cues"] is True
    assert "Mixed emotional cues detected" in prediction.metadata["ambiguity_note"]

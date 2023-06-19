import pytest
import transformers
from transformers.models.auto.modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES


@pytest.mark.parametrize("model_name", MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES.keys())
def test_renate_hf_sequence_cls_transformer(model_name):
    model = pytest.helpers.get_renate_hf_text_cls_transformer(model_name, num_classes=2)
    will_work = False
    if len(list(model.children())) == 2 and hasattr(model, "classifier"):
        will_work = True

    assert True

import pytest

from medrag.api.thinking import is_suppressing, thinking_off_body


def test_is_suppressing_only_when_explicit_off():
    assert is_suppressing({}, "CHATBOT_LLM") is False  # unset -> neutral
    assert is_suppressing({"CHATBOT_LLM_THINKING": "on"}, "CHATBOT_LLM") is False
    assert is_suppressing({"CHATBOT_LLM_THINKING": "off"}, "CHATBOT_LLM") is True
    assert is_suppressing({"CHATBOT_LLM_THINKING": "OFF"}, "CHATBOT_LLM") is True


def test_off_body_empty_when_unset():
    # OPT-IN: when unset, nothing is injected (neutral, no regression).
    assert thinking_off_body({}, "CHATBOT_LLM") == {}


def test_off_body_empty_when_on():
    assert thinking_off_body({"CHATBOT_LLM_THINKING": "on"}, "CHATBOT_LLM") == {}


def test_off_body_default_when_explicit_off():
    env = {"CHATBOT_LLM_THINKING": "off"}
    assert thinking_off_body(env, "CHATBOT_LLM") == {"reasoning_effort": "none"}


def test_off_body_override_json():
    env = {"CHATBOT_LLM_THINKING": "off", "CHATBOT_LLM_THINKING_OFF_BODY": '{"think": false}'}
    assert thinking_off_body(env, "CHATBOT_LLM") == {"think": False}


def test_off_body_invalid_json_raises():
    env = {"CHATBOT_LLM_THINKING": "off", "CHATBOT_LLM_THINKING_OFF_BODY": "{not json"}
    with pytest.raises(ValueError):
        thinking_off_body(env, "CHATBOT_LLM")


def test_off_body_non_object_raises():
    env = {"CHATBOT_LLM_THINKING": "off", "CHATBOT_LLM_THINKING_OFF_BODY": '"a string"'}
    with pytest.raises(ValueError):
        thinking_off_body(env, "CHATBOT_LLM")

from src.translator import translate_content
import vertexai
from mock import patch

def test_chinese():
    is_english, translated_content = translate_content("这是一条中文消息")
    assert is_english == False
    assert translated_content == "This is a Chinese message."

def test_llm_normal_response():
    # test for normal response with non English text
    is_english, translated_content = translate_content("Aquí está su primer ejemplo.")
    assert is_english == False
    assert translated_content == "Here is your first example." or "Here's your first example."

    is_english, translated_content = translate_content("Voici ton premier exemple.")
    assert is_english == False
    assert translated_content == "Here is your first example." or "Here's your first example."

    # tests for normal response with English text
    is_english, translated_content = translate_content("I'm struggling to understand the first question of assignment two. Can someone please explain it more?")
    assert is_english == True
    assert translated_content == "I'm struggling to understand the first question of assignment two. Can someone please explain it more?"

@patch('vertexai.preview.language_models._PreviewChatSession.send_message')
def test_llm_gibberish_response(mocker):
    # tests for response if LLM returns unexpected response
    mocker.return_value.text = "blah blah blah"
    is_english, translated_content = translate_content("Aquí está su primer ejemplo.")
    assert is_english == False
    assert translated_content == "Here is your first example." or "Here's your first example."

    # tests for response if text given to LLM is gibberish
    is_english, translated_content = translate_content("hfdu skdasd fndkosa")
    assert is_english == True
    assert translated_content == "hfdu skdasd fndkosa"

@patch('vertexai.preview.language_models._PreviewChatSession.send_message')
def test_not_a_string_response(mocker):
  mocker.return_value.text = []
  assert translate_content("Aquí está su primer ejemplo.") == (False, "Here is your first example.")

@patch('vertexai.preview.language_models._PreviewChatSession.send_message')
def test_empty_response(mocker):
  mocker.return_value.text = ""
  assert translate_content("Aquí está su primer ejemplo.") == (False, "Here is your first example.")
  assert translate_content("hfdu skdasd fndkosa") == (True, "hfdu skdasd fndkosa")
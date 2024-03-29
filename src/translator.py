from typing import Callable

# PROJECT_ID = "totemic-chalice-417021" # TODO: Replace with your GCP Project ID

# Set the project id
# gcloud config set project {PROJECT_ID}

# from google.cloud import aiplatform


# aiplatform.init(
#     # your Google Cloud Project ID or number
#     # environment default used is not set
#     project=PROJECT_ID,

#     # the Vertex AI region you will use
#     # defaults to us-central1
#     location='us-central1',
# )

from vertexai.language_models import ChatModel, InputOutputTextPair

chat_model = ChatModel.from_pretrained("chat-bison@001")

def get_translation(post: str) -> str:
    parameters = {
        "temperature": 0.7,  # Temperature controls the degree of randomness in token selection.
        "max_output_tokens": 256,  # Token limit determines the maximum amount of text output.
    }
    context = '''You are a professional translator, capable of translating 
    any language into English. Please only give the translation for the 
    phrase you are given. If it is in English, repeat the given phrase. If 
    you are unable to translate, please say "I can't translate"'''
    chat = chat_model.start_chat(context=context)
    response = chat.send_message(post, **parameters)
    return response.text

def get_language(post: str) -> str:
    parameters = {
        "temperature": 0.7,  # Temperature controls the degree of randomness in token selection.
        "max_output_tokens": 256,  # Token limit determines the maximum amount of text output.
    }
    context = '''You are an expert at language classification, 
    including every English dialect. Use one or two words to 
    classify the language used, in English. If you are unable 
    to classify the language, please say "I can't classify"'''
    chat = chat_model.start_chat(context=context)
    response = chat.send_message(post, **parameters)
    return response.text

def translate_content(content: str) -> tuple[bool, str]:
    '''
    Translates input content and returns a tuple where the 
    first value is a bool representing whether or not the content
    is in English and the second value is a string translation
    of the content.
    '''
    if(content == ""):
        return (True, content)

    translation = get_translation(content)
    language = get_language(content)
    lang_lst = language.split()

    # check if model was unable to translate
    if (("can't" in translation and "translate" in translation) or
        # check if model didn't return a one or two word answer for language
        (len(lang_lst) > 2)):
        return (True, content)

    is_en = False
    if 'english' in language.lower():
        is_en = True
    return (is_en, translation)

import json
import torch

from transformers import GPT2ForSequenceClassification, GPT2Tokenizer, DistilBertForSequenceClassification, DistilBertForTokenClassification
import speech_recognition as sr
from gtts import gTTS
import os



# obtain audio from the microphone
r = sr.Recognizer()
def wakeUpAI():
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)

    # recognize speech using Sphinx
    try:
        print("Sphinx thinks you said " + r.recognize_sphinx(audio))
    except sr.UnknownValueError:
        print("Sphinx could not understand audio")
    except sr.RequestError as e:
        print("Sphinx error; {0}".format(e))

    speech = r.recognize_google(audio)

    try:
        # for testing purposes, we're just using the default API key
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        # instead of `r.recognize_google(audio)`
        print("You said: " + speech)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

    tts_en = gTTS(speech, lang='bg')
    tts_en.save('speech.mp3')

    # # Get the current working directory
    # current_dic = os.getcwd()
    #
    # # Construct the model path in the current directory
    # model_path = os.path.join(current_dic, "My_first_Model")


    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    tokenizeR = DistilBertForTokenClassification.from_pretrained("distilbert-base-uncased")

    #Load JSON file
    with open('intents.json', 'r') as file:
        data = json.load(file)

    for entry in data:
        # Tokenize the input text
        input_tokens = tokenizeR(entry['patterns'], return_tensons='pt')

        #Make a prediction
        with torch.no_grad():
            outputs = model(**input_tokens)

        #Get predicted probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().tolist()

        #Get the predicted intent
        predicted_intent_index = torch.argmax(outputs.logits).item()
        predicted_intent = model.config.id2label[predicted_intent_index]

        # Visualize the results
        print("Input Text:", entry['text'])
        print("Predicted Intent:", predicted_intent)
        print("Predicted Probabilities:", {label: prob for label, prob in zip(model.config.id2label.values(), probs)})
        print("-----")


wakeUpAI()


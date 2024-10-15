import streamlit as st
import spacy
import random
import json
import os
from google.cloud import storage

client = storage.Client()
bucket_name = "spacy-new"
bucket = client.bucket(bucket_name)

@st.cache_resource
def load_model():
    """Load the NER model from GCS."""
    model_blob = bucket.blob("model-best/")
    model_blob.download_to_filename("model-best")  
    return spacy.load("model-best")

@st.cache_data
def load_data():
    """Load corrected entities and feedback history from GCS JSON files."""
    corrected_entities_store = {}
    feedback_history = []

 
    try:
        corrected_blob = bucket.blob("corrected_entities.json")
        corrected_blob.download_to_filename("corrected_entities.json")
        with open("corrected_entities.json", "r") as f:
            corrected_entities_store = json.load(f)
    except Exception as e:
        st.warning(f"Could not load corrected entities: {e}")
        with open("corrected_entities.json", "w") as f:
            json.dump(corrected_entities_store, f)

    try:
        feedback_blob = bucket.blob("feedback_history.json")
        feedback_blob.download_to_filename("feedback_history.json")
        with open("feedback_history.json", "r") as f:
            feedback_history = json.load(f)
    except Exception as e:
        st.warning(f"Could not load feedback history: {e}")
        with open("feedback_history.json", "w") as f:
            json.dump(feedback_history, f)

    return corrected_entities_store, feedback_history

def save_data(corrected_entities_store, feedback_history):
    try:
        with open("corrected_entities.json", "w") as f:
            json.dump(corrected_entities_store, f)
        corrected_blob = bucket.blob("corrected_entities.json")
        corrected_blob.upload_from_filename("corrected_entities.json")
    except IOError as e:
        st.error(f"Error saving corrected entities: {e}")

    try:
        with open("feedback_history.json", "w") as f:
            json.dump(feedback_history, f)
        feedback_blob = bucket.blob("feedback_history.json")
        feedback_blob.upload_from_filename("feedback_history.json")
    except IOError as e:
        st.error(f"Error saving feedback history: {e}")

nlp = load_model()
corrected_entities_store, feedback_history = load_data()

q_table = {}
learning_rate = 0.1
discount_factor = 0.9
exploration_rate = 1.0
exploration_decay = 0.99
min_exploration_rate = 0.1

def initialize_q_table(entities):
    for entity in entities:
        if entity not in q_table:
            q_table[entity] = {'Yes': 0.0, 'No': 0.0}

def choose_action(entity):
    if random.uniform(0, 1) < exploration_rate:
        return random.choice(['Yes', 'No']) 
    else:
        return max(q_table[entity], key=q_table[entity].get)

def update_q_table(entity, action, reward):
    best_next_action = max(q_table[entity], key=q_table[entity].get)
    q_table[entity][action] += learning_rate * (reward + discount_factor * q_table[entity][best_next_action] - q_table[entity][action])

def update_model(ent, is_correct):
    feedback_history.append((ent, 1 if is_correct else -1))
    reward = 1 if is_correct else -1
    update_q_table(ent, 'Yes' if is_correct else 'No', reward)

st.header("Use of Customly Created NER Model", divider=True)

text = st.text_input("Enter text for Brand/Product/Model Detection")

if text in corrected_entities_store:
    st.subheader("Previously Corrected Entities:")
    for original, corrected in corrected_entities_store[text].items():
        st.write(f"Original: {original} -> Corrected: {corrected}")

else:
    if text:
        doc = nlp(text)

        initialize_q_table([ent.text for ent in doc.ents])  

        corrected_entities = {}  

        recognized_entities = doc.ents  
        if recognized_entities:
            st.subheader("Currently Recognized Entities:")
            for ent in recognized_entities:
                st.write(f"  {ent.text} (Type: {ent.label_})")  

        st.subheader("Previously Corrected Entities:")

        for ent, corrected in corrected_entities_store.get(text, {}).items():
            st.write(f"Original: {ent} -> Corrected: {corrected}")  

        st.subheader("Correction")    

        for ent in recognized_entities:
            st.write(f"  {ent.text} (Type: {ent.label_})")
            
            action = choose_action(ent.text) 
            feedback = st.radio(f"Is this correct? {ent.text}", ('Yes', 'No'), key=ent.text)
            
            if feedback:
                is_correct = feedback == 'Yes'
                update_model(ent.text, is_correct)
                st.success("Feedback recorded!")
                
                corrected_entity = st.text_input(f"Correct the recognition for '{ent.text}' if necessary:", key=f"correct_{ent.text}")
                if corrected_entity:
                    corrected_entities[ent.text] = corrected_entity  
                    st.write(f"Corrected Entity: {corrected_entity}")

            exploration_rate = max(min_exploration_rate, exploration_rate * exploration_decay)

        if corrected_entities:
            corrected_entities_store[text] = corrected_entities

            st.subheader("Corrected Entities:")
            for original, corrected in corrected_entities.items():
                st.write(f"Original: {original} -> Corrected: {corrected}")

st.subheader("Add Unrecognized Entities")

unrecognized_entity = st.text_input("Enter any unrecognized entity you'd like to add:")
entity_type = st.text_input("Enter the type of the entity:")

if unrecognized_entity and entity_type:
    if text not in corrected_entities_store:
        corrected_entities_store[text] = {}
    corrected_entities_store[text][unrecognized_entity] = entity_type
    st.success(f"Added unrecognized entity: '{unrecognized_entity}' of type '{entity_type}'")

save_data(corrected_entities_store, feedback_history)

st.subheader("Current Q-Table:")
st.write(q_table)

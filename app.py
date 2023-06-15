import streamlit as st
import torch
import pandas as pd
from transformers import BertTokenizerFast, BertForSequenceClassification

path = "Raw_DataJobs2.csv"
df = pd.read_csv(path)


def get_result(job_description):
    # Instantiate the tokenizer and model
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

    # Load the model state dictionary from the saved .pth file
    model.load_state_dict(torch.load("./trained_model.pth"))

    # Tokenize the job description
    inputs = tokenizer(
        job_description, truncation=True, padding=True, return_tensors="pt"
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    model.eval()  # Set the model to evaluation mode

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # Get the predicted label
    predicted_label = torch.argmax(outputs.logits).item()

    return df["Job_title"].iloc[predicted_label]


st.title("JobQuail model")

st_input = st.text_input("./trained_model.pth")

if st.button("Get Result"):
    result = get_result(st_input)

    st.write("The resulting job is: ", result)

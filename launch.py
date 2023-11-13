from langchain.llms import OpenAI
from langchain.chains import LLMChain
from pydantic import BaseModel
from typing import List, Any, Dict, Union
import requests
import json
import os
from schema import PredictSchema

def loadmodel(logger):
    """Get the model"""
    openai_model = OpenAI(
        model_name=os.environ.get("MODEL_NAME", "gpt-3.5-turbo-16k"),
        openai_api_key=os.environ.get("API_KEY"),
        temperature=0,
        max_tokens=10000,
        top_p=1.0,
        frequency_penalty=1.0
    )
    logger.info(f"Model fetched")
    return openai_model

def preprocessing(data, logger):
    """ Applies preprocessing techniques to extract specific info from the raw data"""
    logger.info("Task fetched.")

    # Modify the prompt to guide the model in extracting information
    final_prompt =(
        f"""
    You are an expert in writing professional emails related to Non-Disclosure Agreements (NDAs). You possesses a comprehensive understanding of legal and contractual matters, particularly in the context of NDAs. This expertise extends to addressing specific scenarios such as when the NDA is expired or when it is set to expire within the next 30 days. You are adept at crafting clear, concise, and legally sound communication that adheres to the terms outlined in the NDA.
    You will get an input containing the client's name, contract start date, contract end date and special terms of renewal. Looking at the contract start and end date, you have to analyze whether the contract has already expired or is about to expire within next 30 days. Now your main task is to write an email based on below definitions:
    Expired Contract: In the case of an expired contract, you should be proficient in articulating the need for reevaluation, renegotiation, or any other relevant actions. They would ensure that the language used in the email is diplomatic yet assertive, emphasizing the importance of compliance with the agreement's terms even after expiration.
    Expiring Contract within 30 Days: When the contract is approaching its expiration within the next 30 days, you should be well-versed in reviewing the input for any specified steps or procedures related to renewal. They would craft an email that effectively communicates the impending expiration, reminds the parties involved of the renewal protocols outlined in the contract, and may include a proactive suggestion for initiating the renewal process.
    In the output, give only the email. The email should be addressing Mindfieldsglobal.
    has context menu
    """
    )

    logger.info("Created the final Prompt")
    return final_prompt

def predict(final_prompt, openai_model, logger):
    """Predicts the results for the given inputs"""
    logger.info(f"final_prompt: {final_prompt}")
    logger.info("Model prediction started.")
    try:
        response = openai_model(final_prompt)
    except Exception as e:
        logger.info(e)
    logger.info("Prediction Done.")
    return response
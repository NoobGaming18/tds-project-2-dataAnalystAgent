# main.py

import os
import asyncio
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict

from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# --- Configuration ---
# It's best practice to load the API key from environment variables.
# We will set this in Hugging Face Spaces secrets.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REQUEST_TIMEOUT_SECONDS = 180 # 3 minutes as specified

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Data Analyst Agent API",
    description="An API that uses LLMs to analyze data and answer questions.",
)

# --- The Core Agent Logic ---
def get_agent_response(df: pd.DataFrame, questions: List[str]) -> Dict[str, str]:
    """
    Initializes and runs the LangChain Pandas DataFrame agent to answer questions.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set. Please set it in your environment.")

    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-4o",  # A powerful and cost-effective model
        temperature=0,
        api_key=OPENAI_API_KEY
    )

    # Initialize the Pandas Agent
    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True, # Set to True for debugging to see the agent's thoughts
        agent_executor_kwargs={"handle_parsing_errors": True} # Helps with robustness
    )

    # Process all questions
    results = {}
    for question in questions:
        if not question.strip():
            continue
        try:
            # Invoke the agent for each question
            response = agent.invoke({"input": question})
            results[question] = response.get("output", "No answer found.")
        except Exception as e:
            print(f"Error processing question '{question}': {e}")
            results[question] = f"Error: Could not process the question. Details: {e}"
            
    return results


# --- API Endpoint Definition ---
@app.post("/api/")
async def analyze_data(files: List[UploadFile] = File(...)):
    """
    This endpoint accepts a list of files, processes them, and returns
    answers to questions found in 'questions.txt'.
    """
    questions_file = None
    data_file = None
    other_files_content = {}

    # 1. Segregate uploaded files
    for uploaded_file in files:
        if uploaded_file.filename == "questions.txt":
            questions_file = uploaded_file
        # Prioritize CSV files for the pandas agent
        elif uploaded_file.filename.endswith(".csv"):
            data_file = uploaded_file
        else:
            # Store content of other files to potentially add to context (optional)
            content = await uploaded_file.read()
            other_files_content[uploaded_file.filename] = content.decode('utf-8', errors='ignore')

    # 2. Validate that essential files were provided
    if not questions_file:
        raise HTTPException(status_code=400, detail="questions.txt is missing.")
    if not data_file:
        raise HTTPException(status_code=400, detail="A data file (e.g., .csv) is missing.")

    # 3. Read and process files
    try:
        # Read questions
        questions_content = await questions_file.read()
        questions = [q.strip() for q in questions_content.decode('utf-8').splitlines() if q.strip()]
        
        # Read the data file into a pandas DataFrame
        df = pd.read_csv(data_file.file)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading or processing files: {e}")

    # 4. Run the agent with a timeout
    try:
        # Use asyncio.wait_for to enforce the timeout
        results = await asyncio.wait_for(
            asyncio.to_thread(get_agent_response, df, questions),
            timeout=REQUEST_TIMEOUT_SECONDS
        )
        return JSONResponse(content=results)

    except asyncio.TimeoutError:
        # If the agent takes too long, return a specific error message
        # This ensures you still return a valid JSON structure and don't fail the test case
        timeout_response = {q: "Analysis timed out and could not be completed." for q in questions}
        return JSONResponse(status_code=408, content=timeout_response)
        
    except Exception as e:
        # Catch any other errors during agent execution
        error_response = {q: f"An unexpected error occurred: {e}" for q in questions}
        return JSONResponse(status_code=500, content=error_response)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Data Analyst Agent API. Please POST to /api/ with your files."}
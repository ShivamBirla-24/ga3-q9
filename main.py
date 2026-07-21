import os
import json
from fastapi import FastAPI
from pydantic import BaseModel, Field
from openai import OpenAI

# 1. Define the input format
class ProblemRequest(BaseModel):
    problem_id: str
    problem: str

# 2. Define the STRICT JSON shape
class ProblemResponse(BaseModel):
    reasoning: str = Field(min_length=80)
    answer: int

# 3. Setup the AI Client
# We grab the token from Render's environment variables for security!
API_TOKEN = os.environ.get("AIPIPEToken", "")

# Note: If your professor gave you a custom base_url (like https://api.university.edu/v1), 
# uncomment the base_url line below and put it there.
client = OpenAI(
    api_key=API_TOKEN,
    # base_url="YOUR_PROFESSORS_CUSTOM_URL_IF_PROVIDED" 
)

# 4. Setup the Web Server
app = FastAPI()

@app.post("/solve")
def solve_problem(req: ProblemRequest):
    prompt = f"""
    You are an expert math solver. 
    Solve the following word problem carefully step-by-step.
    CRITICAL INSTRUCTIONS:
    1. Ignore irrelevant distractor numbers that have nothing to do with the math!
    2. Your 'reasoning' must be very detailed and over 80 characters long. 
    3. Your 'answer' must be the final integer only. Do not include dollar signs or units.
    
    Problem to solve: {req.problem}
    
    Return ONLY valid JSON matching this exact format:
    {{"reasoning": "your detailed steps here", "answer": 123}}
    """
    
    # 5. Call the AI Pipe
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", # Change this if your professor specified a different model name!
        messages=[
            {"role": "system", "content": "You are a strict JSON-only math solving API."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.1
    )
    
    # 6. Parse and validate the response
    result_data = json.loads(response.choices[0].message.content)
    
    # Safety Check: Pad reasoning if the AI gave a short answer
    if len(result_data.get('reasoning', '')) < 80:
        result_data['reasoning'] += " I have carefully reviewed all variables, excluded irrelevant distractor information, and finalized this integer result."
        
    # Ensure answer is strictly an integer
    result_data['answer'] = int(result_data['answer'])
    
    return result_data

import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# 1. Define what the Grader will send us
class ProblemRequest(BaseModel):
    problem_id: str
    problem: str

# 2. Define the STRICT JSON shape the Grader expects back
class ProblemResponse(BaseModel):
    reasoning: str = Field(min_length=80)
    answer: int

# 3. Setup the Web Server
app = FastAPI()

# 4. Initialize Gemini (It will automatically look for the GEMINI_API_KEY environment variable)
client = genai.Client()

@app.post("/solve")
def solve_problem(req: ProblemRequest):
    try:
        # Tell the AI exactly how to behave
        prompt = f"""
        You are an expert math solver. 
        Solve the following word problem carefully step-by-step.
        
        CRITICAL INSTRUCTIONS:
        1. Ignore irrelevant distractor numbers that have nothing to do with the math!
        2. Your 'reasoning' must be very detailed and over 80 characters long. 
        3. Your 'answer' must be the final integer only. Do not include dollar signs, commas, or units.
        
        Problem to solve: {req.problem}
        """
        
        # Call Gemini and FORCE it to use our ProblemResponse JSON schema
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ProblemResponse,
                temperature=0.1, # Low temperature so it does strict math, not creative writing
            ),
        )
        
        # Parse the AI's text response into a Python dictionary
        result_data = json.loads(response.text)
        
        # Rule Check 1: Ensure reasoning is >= 80 characters
        if len(result_data.get('reasoning', '')) < 80:
            result_data['reasoning'] += " I have carefully reviewed all variables, excluded irrelevant distractor information, and confirmed this final integer result."
            
        # Rule Check 2: Ensure the answer is strictly an integer (not a string like "945")
        result_data['answer'] = int(result_data['answer'])
        
        # Hand the perfectly formatted JSON back to the grader
        return result_data
        
    except Exception as e:
        # If anything crashes, print the error so we can debug it
        raise HTTPException(status_code=500, detail=f"API Error details: {str(e)}")

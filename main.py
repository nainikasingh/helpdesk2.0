import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

# Claude setup
client = Anthropic(api_key=CLAUDE_API_KEY)

app = FastAPI()

# Load Google Sheet once at startup
df = None

def load_google_sheet(sheet_name: str, worksheet_title: str = "Sheet1") -> pd.DataFrame:
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
    gc = gspread.authorize(creds)
    sheet = gc.open(sheet_name)
    worksheet = sheet.worksheet(worksheet_title)
    data = worksheet.get_all_values()

    df = pd.DataFrame(data)
    df.columns = df.iloc[0]  # First row as header
    df = df[2:]  # Skip header and extra row
    df = df.dropna(subset=["Issue Text", "Solution Text", "Solution Image", "Sheet Name"])
    df = df.reset_index(drop=True)
    return df

@app.on_event("startup")
def startup_event():
    global df
    df = load_google_sheet("Helpdesk Issues Database/Glossary", "DATA")  # Replace with actual names

class QueryInput(BaseModel):
    query: str

def get_relevant_issues(query: str, issue_texts: pd.Series, top_k: int = 5) -> List[str]:
    prompt = f"""
You are a helpful assistant. Given the user query:

"{query}"

And the following list of known technical issues:

{chr(10).join(f"{i+1}. {txt}" for i, txt in enumerate(issue_texts.tolist()))}

Return the top {top_k} issues that most closely match the query.
Only return the issue texts, as a plain list with no extra explanation.
"""
    try:
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=200,
            temperature=0,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        response = message.content[0].text
        return [line.strip("-•1234567890. ") for line in response.splitlines() if line.strip()]
    except Exception as e:
        return []

@app.post("/get_solution")
def get_solution(input: QueryInput):
    query = input.query.lower()
    sheet_names = df["Sheet Name"].str.lower().unique()

    matching_sheets = [s for s in sheet_names if s in query]

    if matching_sheets:
        sheet_df = df[df["Sheet Name"].str.lower().isin(matching_sheets)]
        relevant_issues = get_relevant_issues(query, sheet_df["Issue Text"], top_k=5)

        if relevant_issues:
            result_df = sheet_df[sheet_df["Issue Text"].isin(relevant_issues)]
            return {
                "type": "Exact Match",
                "solutions": result_df[["Issue Text", "Solution Text", "Solution Image", "Sheet Name"]].to_dict(orient="records")
            }
        else:
            suggestions = get_relevant_issues(query, df["Issue Text"], top_k=10)
            suggestions_df = df[df["Issue Text"].isin(suggestions)]
            if len(suggestions_df) > 5:
                matched_sheets = suggestions_df["Sheet Name"].unique().tolist()
                return {
                    "type": "Sheet Detected but Issue Incorrect",
                    "matching_sheet_names": matched_sheets,
                    "suggestions": suggestions_df[["Issue Text", "Solution Text", "Sheet Name"]].to_dict(orient="records")
                }
            return {
                "type": "Sheet Detected but Issue Incorrect",
                "suggestions": suggestions_df[["Issue Text", "Solution Text", "Sheet Name"]].to_dict(orient="records")
            }

    # No sheet explicitly mentioned
    relevant_issues = get_relevant_issues(query, df["Issue Text"], top_k=10)
    result_df = df[df["Issue Text"].isin(relevant_issues)]
    matched_sheets = result_df["Sheet Name"].unique()

    if len(matched_sheets) > 5:
        return {
            "type": "Too Many Matching Sheets",
            "matching_sheet_names": matched_sheets.tolist()
        }
    elif 1 <= len(matched_sheets) <= 3:
        return {
            "type": "Matching ≤ 3 Sheets",
            "solutions": result_df[["Issue Text", "Solution Text", "Solution Image", "Sheet Name"]].to_dict(orient="records")
        }
    elif not result_df.empty:
        return {
            "type": "Suggestions Without Sheet",
            "solutions": result_df[["Issue Text", "Solution Text", "Sheet Name"]].to_dict(orient="records")
        }

    return {
        "type": "No Match Found",
        "message": "No relevant issues found. Please rephrase your query."
    }

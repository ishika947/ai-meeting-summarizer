import json
import httpx
from groq import Groq
from app.core.config import settings

class LLMService:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY, http_client=httpx.Client())

    def analyze_transcript(self, transcript_text: str) -> dict:
        prompt = (
            "You are an expert meeting assistant. Analyze the following meeting transcript. "
            "You must return the response strictly as a valid JSON object matching this schema:\n"
            "{\n"
            "  \"summary\": \"A short high-level overview of the meeting\",\n"
            "  \"key_takeaways\": [\"point 1\", \"point 2\"],\n"
            "  \"action_items\": [\n"
            "    {\"task\": \"task description\", \"owner\": \"person name or Unassigned\", \"deadline\": \"date/time or TBD\"}\n"
            "  ]\n"
            "}\n"
            "Do not include any prose, markdown block, introduction, or conversation before or after the JSON. "
            f"Transcript:\n{transcript_text}"
        )

        response = self.client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"},
            temperature=0.2
        )

        raw_content = response.choices[0].message.content
        return json.loads(raw_content)
    
    def ask_question(self, transcript_text: str, question: str) -> str:
        prompt = (
            "You are an AI assistant helping a user with questions about a meeting transcript.\n"
            f"Here is the meeting transcript:\n###\n{transcript_text}\n###\n\n"
            f"Based ONLY on the transcript above, answer the user's question. If the answer cannot be found in the transcript, "
            "say 'I cannot find the answer in the provided transcript.'\n\n"
            f"Question: {question}"
        )

        response = self.client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.3
        )

        return response.choices[0].message.content
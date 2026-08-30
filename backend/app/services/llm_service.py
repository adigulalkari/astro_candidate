import os
import json
from dotenv import load_dotenv
load_dotenv()

class LLMService:
    def __init__(self):
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.gemini_key = os.environ.get("GEMINI_API_KEY")
        self.bedrock_access_key = os.environ.get("AWS_ACCESS_KEY_ID")

        self.provider = "mock"
        
        if self.gemini_key:
            try:
                # Use the new official google-genai SDK
                from google import genai
                self.gemini_client = genai.Client(api_key=self.gemini_key)
                self.provider = "gemini"
                print("[LLM Service] Using Gemini API.")
            except Exception as e:
                print(f"[LLM Service] Failed to initialize Gemini API: {e}")

        elif self.openai_key:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=self.openai_key)
                self.provider = "openai"
                print("[LLM Service] Using OpenAI API.")
            except Exception as e:
                print(f"[LLM Service] Failed to initialize OpenAI API: {e}")

        else:
            print("[LLM Service] No API keys detected. Running in mock/offline demo mode.")

    def generate_response(self, system_prompt: str, user_prompt: str, response_format_json: bool = True) -> str:
        """
        Sends the prompt to the active LLM provider and returns the raw response text.
        """
        if self.provider == "gemini":
            return self._generate_gemini(system_prompt, user_prompt, response_format_json)
        elif self.provider == "openai":
            return self._generate_openai(system_prompt, user_prompt, response_format_json)
        else:
            return self._generate_mock(system_prompt, user_prompt)

    def _generate_gemini(self, system_prompt: str, user_prompt: str, json_format: bool) -> str:
        from google.genai import types
        import time
        
        # List of models to try in case of 503 demand spikes
        models_to_try = ["gemini-3.6-flash"]
        
        for model_name in models_to_try:
            retries = 3
            delay = 2  # Initial backoff delay in seconds
            
            while retries > 0:
                try:
                    # Use the recommended Chat interface instead of direct generate_content
                    # System instructions are set during chat creation
                    chat = self.gemini_client.chats.create(
                        model=model_name,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            response_mime_type="application/json" if json_format else None
                        )
                    )
                    
                    # Send the user message to the chat session
                    response = chat.send_message(user_prompt)
                    return response.text.strip()
                    
                except Exception as e:
                    error_msg = str(e)
                    # Check if it's a 503/temporary overload error
                    if "503" in error_msg or "UNAVAILABLE" in error_msg:
                        print(f"[LLM Service] {model_name} overloaded (503). Retrying in {delay}s...")
                        time.sleep(delay)
                        retries -= 1
                        delay *= 2  # Exponential backoff
                    else:
                        # If it's a different error, break out of retry loop to try next model or mock
                        print(f"[LLM Service] Gemini error with {model_name}: {e}")
                        break
                        
        # Ultimate fallback if all models and retries fail
        print("[LLM Service] All Gemini attempts failed. Falling back to mock.")
        return self._generate_mock(system_prompt, user_prompt)

    def _generate_openai(self, system_prompt: str, user_prompt: str, json_format: bool) -> str:
        try:
            args = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }
            if json_format:
                args["response_format"] = {"type": "json_object"}
            
            response = self.openai_client.chat.completions.create(**args)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[LLM Service] OpenAI execution failed: {e}. Falling back to mock.")
            return self._generate_mock(system_prompt, user_prompt)

    def _generate_mock(self, system_prompt: str, user_prompt: str) -> str:
        """
        A rule-based mock generator that inspects the query/context
        and constructs a valid, structure-compliant response schema.
        """
        planets_data = []
        papers_data = []
        query_str = ""

        lines = user_prompt.split("\n")
        for line in lines:
            if line.startswith("User Question:"):
                query_str = line.replace("User Question:", "").strip()
            elif "ml_score" in line or "pl_name" in line:
                planets_data.append(line)
            elif "title" in line or "abstract" in line or "paper_id" in line:
                papers_data.append(line)

        candidates = []
        evidence = []
        uncertainties = [
            "This report is generated in local offline demo mode (no active LLM key was provided).",
            "Astronomical parameters contain measurement uncertainties (e.g. error bars on radius/temperature).",
            "This model computes similarity to an Earth-like profile, which is not a confirmation of habitability."
        ]

        if "trappist-1 e" in user_prompt.lower() or "trappist-1e" in user_prompt.lower():
            candidates.append({
                "planet": "TRAPPIST-1 e",
                "score": 0.9999,
                "reasons": [
                    "Earth-sized candidate (0.92 Earth radii) orbiting a temperate M-dwarf star.",
                    "Positioned in the habitable zone of TRAPPIST-1 with optimal insolation (0.65 Earth units).",
                    "Equilibrium temperature of 250K is highly comparable to Earth's temperature profile."
                ]
            })
            evidence.append({
                "title": "JWST-TST DREAMS: NIRSpec/PRISM Transmission Spectroscopy of the Habitable Zone Planet TRAPPIST-1 e",
                "year": 2023,
                "url": "https://arxiv.org/abs/2301.00000",
                "claim_supported": "JWST observations support a temperate environment with potential secondary atmosphere constraints."
            })
        
        if "proxima cen b" in user_prompt.lower() or "proxima centauri b" in user_prompt.lower():
            candidates.append({
                "planet": "Proxima Cen b",
                "score": 0.9999,
                "reasons": [
                    "Closest known rocky exoplanet to Earth (1.30 pc).",
                    "Orbiting within the habitable zone of Proxima Centauri with 64% of Earth's insolation.",
                    "Mass is approximately 1.07 Earth masses, making it a strong terrestrial candidate."
                ]
            })
            evidence.append({
                "title": "A candidate terrestrial planet orbiting Proxima Centauri",
                "year": 2016,
                "url": "https://arxiv.org/abs/1608.06822",
                "claim_supported": "Discovered via radial velocity; receives temperate stellar irradiation."
            })

        if not candidates:
            import re
            names = re.findall(r"\'pl_name\':\s*\'([^\']+)\'", user_prompt)
            if not names:
                names = re.findall(r"\"pl_name\":\s*\"([^\"]+)\"", user_prompt)
            if not names:
                names = ["Proxima Cen b", "TRAPPIST-1 e"]
            
            for name in list(set(names))[:3]:
                candidates.append({
                    "planet": name,
                    "score": 0.95,
                    "reasons": [
                        f"Highly ranked candidate in system databases.",
                        f"Orbital properties indicate a rocky candidate profile."
                    ]
                })

        mock_response = {
            "answer": f"Analysis for query: '{query_str if query_str else 'Exoplanet candidates search'}'. This response was synthesized by the local offline inference engine using SQLite and RAG indexes. The most promising candidate evaluated is {candidates[0]['planet'] if candidates else 'Proxima Cen b'}.",
            "candidates": candidates,
            "evidence": evidence,
            "uncertainties": uncertainties
        }

        return json.dumps(mock_response, indent=2)

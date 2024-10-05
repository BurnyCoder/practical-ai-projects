import requests
import os

def elevenlabs_tts(text, voice_id, api_key):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    
    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code == 200:
        with open("output.mp3", "wb") as f:
            f.write(response.content)
        print("Audio saved as output.mp3")
    else:
        print(f"Error: {response.status_code} - {response.text}")

# Example usage
text = "Hello, this is a test of the ElevenLabs Text-to-Speech API."
voice_id = "21m00Tcm4TlvDq8ikWAM"  # Example voice ID (you can change this)
api_key = os.environ.get("ELEVENLABS_API_KEY")

elevenlabs_tts(text, voice_id, api_key)
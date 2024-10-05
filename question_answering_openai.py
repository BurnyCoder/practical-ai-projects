import openai
import os

def init():
    # Initialize the OpenAI client
    return openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def query(client, user_message, model="gpt-4o"):
    # Make the API call
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message}
        ]
    )

    # Return the response
    return response.choices[0].message.content

# Example usage
if __name__ == "__main__":
    client = init()
    result = query(client, "Who is the current president of the United States?")
    print(result)
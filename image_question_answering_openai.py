import os
from openai import OpenAI

# Set up the OpenAI API client
client = OpenAI()

def visual_question_answering(image_url, question):
    # Prepare the messages for the API call
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }
            ]
        }
    ]

    # Make the API call
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=300
    )

    # Return the model's answer
    return response.choices[0].message.content

# Example usage
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
question = "What's in this image?"

answer = visual_question_answering(image_url, question)
print(f"Question: {question}")
print(f"Answer: {answer}")
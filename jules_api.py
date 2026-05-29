import requests
import os
from dotenv import load_dotenv
load_dotenv()

def listJulesSources():
    """
    Retrieve the list of available sources from the Google Jules API.

    This function sends a GET request to the Jules sources endpoint using
    the API key stored in the `JULES_API_KEY` environment variable and
    returns the raw response body as a string.

    Returns:
        str: The raw JSON response returned by the Jules API.

    Raises:
        requests.exceptions.RequestException:
            If an error occurs while making the HTTP request.
            
    """
    url = "https://jules.googleapis.com/v1alpha/sources"

    headers = {
        "X-Goog-Api-Key": os.getenv('JULES_API_KEY'),
        "Accept": "application/json"
    }

    response = requests.get(url, headers=headers)

    return response.text

if __name__=="__main__":
    import postgres_agent
    from langchain_openrouter import ChatOpenRouter
    agent=postgres_agent.PersistentAgent(tools=[listJulesSources],llm=ChatOpenRouter(model="google/gemma-4-31b-it:free",api_key=os.getenv('OPEN_ROUTER_API')),system_message="You are an agent to interact with Jules api")
    print(agent.chat(input("prompt: "),"user2"), )
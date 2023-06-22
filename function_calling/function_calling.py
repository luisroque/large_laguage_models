import os
import json
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt
import requests
import psycopg2
import requests


GPT_MODEL = "gpt-3.5-turbo-0613"
openai.api_key = os.getenv("OPENAI_KEY")

DATABASE = {
    "dbname": "user_database",
    "user": "postgres",
    "password": "testpass",
    "host": "localhost",
    "port": 5433,
}


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, functions=None, model=GPT_MODEL):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai.api_key,
    }
    json_data = {"model": model, "messages": messages}
    if functions is not None:
        json_data.update({"functions": functions})
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


class Chat:
    def __init__(self):
        self.conversation_history = []

    def add_prompt(self, role, content):
        message = {"role": role, "content": content}
        self.conversation_history.append(message)

    def display_conversation(self):
        for message in self.conversation_history:
            print(
                f"{message['role']}: {message['content']}\n\n",
                message["role"],
            )


functions = [
    {
        "name": "send_email",
        "description": "Send a new email",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "The destination email.",
                },
                "name": {
                    "type": "string",
                    "description": "The name of the person that will receive the email.",
                },
                "body": {
                    "type": "string",
                    "description": "The body of the email.",
                },
            },
            "required": ["to", "name", "body"],
        },
    },
    {
        "name": "sql_query_email",
        "description": "SQL query to get user emails",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to get users emails. "
                                   "The database has a table called users with username and email fields.",
                },
            },
            "required": ["query"],
        },
    },
]


def query_db(query):
    try:
        conn = psycopg2.connect(**DATABASE)
        cur = conn.cursor()
        cur.execute(query)
        result = cur.fetchone()

        cur.close()
        conn.close()

        if result:
            return result
        else:
            return "No returned result"

    except Exception as e:
        print(f"Unable to query the database: {e}")
        return None


def send_email(name, email, body):
    url = "http://localhost:1000/send_email"
    data = {
        "name": name,
        "email": email,
        "body": body,
    }
    response = requests.post(url, json=data)

    if response.status_code == 200:
        print(response.json())
    else:
        print(f"Failed to send email: {response.content}")


chat = Chat()

chat.add_prompt("user", "Send an email to user10 saying that he needs to pay the monthly subscription fee.")
result_query = ''
for i in range(2):
    chat_response = chat_completion_request(
        chat.conversation_history,
        functions=functions
    )

    if chat_response is not None:
        response_content = chat_response.json()['choices'][0]['message']

        print(response_content)

        if 'function_call' in response_content:
            if response_content['function_call']['name'] == 'send_email':

                print(result_query)
                res = json.loads(response_content['function_call']['arguments'])
                send_email(res['name'], res['to'], res['body'])
                break
            elif response_content['function_call']['name'] == 'sql_query_email':
                result_query = query_db(json.loads(response_content['function_call']['arguments'])['query'])
                chat.add_prompt('user', str(result_query))
                print(result_query)
        else:
            chat.add_prompt('assistant', response_content['content'])
    else:
        print("ChatCompletion request failed. Retrying...")





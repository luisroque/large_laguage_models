from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class User(BaseModel):
    name: str
    email: str
    body: str


@app.post("/send_email")
async def send_email(user: User):
    return {
        "message": f"Email successfully sent to {user.name} with email {user.email}. Email body:\n\n{user.body}"
    }

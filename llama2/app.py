from fastapi import FastAPI
from pydantic import BaseModel
from celery.result import AsyncResult
from typing import Any
from celery_worker import generate_text_task
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()


class Item(BaseModel):
    prompt: str


@app.post("/generate/")
async def generate_text(item: Item) -> Any:
    task = generate_text_task.delay(item.prompt)
    return {"task_id": task.id}


@app.get("/task/{task_id}")
async def get_task(task_id: str) -> Any:
    result = AsyncResult(task_id)
    if result.ready():
        res = result.get()
        return {"result": res[0],
                "time": res[1],
                "memory": res[2]}
    else:
        return {"status": "Task not completed yet"}

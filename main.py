import logging
from typing import Union, Dict, Any, Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, validator

from spiralflow.message import Role, InputMessage
from spiralflow.chat_history import ChatHistory
from chatbot import Chatbot, MemoryChatbot


app = FastAPI()
templates = Jinja2Templates(directory="templates")

chatbot_default_settings = {
    "memory_file": None,
    "openai_chat_model": "gpt-3.5-turbo",
    "persona": "You are a very intelligent assistant.",
    "temperature": 0.3,
    "enable_chat_history": True,
    "verbose": True,
    "chatbot_type": "default",
}
threads_data = {}


def create_new_thread(thread_name):
    global threads_data

    chatbot = get_chatbot(chatbot_default_settings)

    threads_data[thread_name] = {
        "chatbot_settings": dict(chatbot_default_settings),
        "chatbot": chatbot,
        "chat_history": ChatHistory(),
        "instruction": "",
    }

def get_chatbot(settings):
    settings = dict(settings)
    chatbot_type = settings["chatbot_type"]

    if "16k" in settings["openai_chat_model"]:
        settings["max_num_prompt_tokens"] = 8000
        settings["max_chat_history_tokens"] = 8000

        if chatbot_type in ["memory"]:
            settings["max_memory_context_tokens"] = 3000
            settings["max_num_query_results"] = 30
            settings["num_query_results"] = 40

    del settings["chatbot_type"]
    if chatbot_type in ["memory"]:
        if settings["memory_file"] is None:
            settings["memory_file"] = "memory_default.pkl"
        chatbot = MemoryChatbot(**settings)
    elif chatbot_type in ["default", None]:
        del settings["memory_file"]
        chatbot = Chatbot(**settings)
    else:
        del settings["memory_file"]
        chatbot = Chatbot(**settings)

    print(settings)

    return chatbot


class ChatResponse(BaseModel):
    """Chat response schema."""

    sender: str
    message: Optional[Union[str, Dict[str, Any]]]
    type: str
    thread: str

    @validator("sender")
    def validate_sender_type(cls, v):
        if v not in ["assistant", "user"]:
            raise ValueError(f"sender must be assistant or user. {v} is invalid.")
        return v

    @validator("type")
    def validate_message_type(cls, v):
        if v not in [
            "start",
            "stream",
            "end",
            "delete_message",
            "history",
            "error",
            "info",
            "setting",
            "delete_thread",
            "list_threads",
            "instruction",
        ]:
            raise ValueError(
                f"type must be start, stream, end, error, info, or setting. {v} is invalid."
            )
        return v


@app.on_event("startup")
async def startup_event():
    logging.info("starting up!")


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    global chatbot_default_settings, threads_data
    await websocket.accept()

    print("NEW")
    while True:
        print("HI")
        thread_name = "ERROR"
        try:
            # Receive and send back the client message
            resp = await websocket.receive_json()
            print(resp)
            resp = ChatResponse(**resp)

            thread_name = resp.thread
            if thread_name not in threads_data:
                create_new_thread(thread_name)
            thread_data = threads_data[thread_name]

            if resp.type == "setting":
                new_settings = dict(chatbot_default_settings)
                if resp.message is not None:
                    new_settings.update(resp.message)
                    new_settings = {
                        k: v
                        for k, v in new_settings.items()
                        if k in chatbot_default_settings
                    }
                    thread_data["chatbot"] = get_chatbot(new_settings)
                    thread_data["chatbot_settings"] = new_settings

                resp.message = thread_data["chatbot_settings"]
                await websocket.send_json(resp.dict())
            elif resp.type == "stream":
                prompt = resp.message
                await websocket.send_json(resp.dict())

                thread_data["chat_history"].add_message(
                    InputMessage(resp.message, role=Role.USER).get_const()
                )

                # Construct a response
                start_resp = ChatResponse(sender="assistant", message="", type="start", thread=thread_name)
                await websocket.send_json(start_resp.dict())

                # Get response
                if thread_data["chatbot_settings"]["chatbot_type"] == "memory":
                    (
                        response,
                        sources,
                        history_list,
                        total_tokens_list,
                        chat_history_full,
                    ) = thread_data["chatbot"].chat(
                        prompt,
                        instructions=thread_data["instruction"],
                        chat_history=thread_data["chat_history"]
                        if thread_data["chatbot_settings"]["enable_chat_history"]
                        else None,
                    )
                else:
                    (
                        response,
                        history_list,
                        total_tokens_list,
                        chat_history_full,
                    ) = thread_data["chatbot"].chat(
                        prompt + f" ({thread_data['instruction']})",
                        chat_history=thread_data["chat_history"]
                        if thread_data["chatbot_settings"]["enable_chat_history"]
                        else None,
                    )
                    sources = [None]

                response, sources, num_tokens = (
                    response[0],
                    sources[0],
                    total_tokens_list[0],
                )

                thread_data["chat_history"].add_message(
                    InputMessage(response, role=Role.ASSISTANT).get_const()
                )

                # Format response
                full_response = ""
                full_response += f"\n{response}"
                if sources is not None:
                    full_response += "\n\nTop Possible Sources:\n"
                    for ndx, source in enumerate(sources):
                        full_response += f"{ndx + 1}. {source[0]}\n"
                full_response += f"\n\nNumber of tokens: {num_tokens / 1000:.1f}K"

                # Send response
                stream_resp = ChatResponse(
                    sender="assistant", message=full_response, type="stream", thread=thread_name
                )
                await websocket.send_json(stream_resp.dict())

                # Send and end with full response
                end_resp = ChatResponse(sender="assistant", message="", type="end", thread=thread_name)
                await websocket.send_json(end_resp.dict())
            elif resp.type == "delete_message":
                messages = thread_data["chat_history"].messages
                del messages[int(resp.message)]
                thread_data["chat_history"] = ChatHistory(messages=messages)
            elif resp.type == "delete_thread":
                del threads_data[thread_name]
            elif resp.type == "list_threads":
                resp.message = list(threads_data.keys())
                await websocket.send_json(resp.dict())
            elif resp.type == "history":
                for message in thread_data["chat_history"].messages:
                    if message.role == Role.USER:
                        await websocket.send_json(
                            ChatResponse(
                                sender="user", message=str(message), type="history", thread=thread_name
                            ).dict()
                        )
                    else:
                        await websocket.send_json(
                            ChatResponse(
                                sender="assistant", message=str(message), type="history", thread=thread_name
                            ).dict()
                        )
            elif resp.type == "instruction":
                thread_data["instruction"] = resp.message
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="assistant",
                message=f"Sorry, something went wrong. Try again.\n\nError: {e}",
                type="error",
                thread=thread_name
            )
            await websocket.send_json(resp.dict())

            # thread_data["chatbot"] = MemoryChatbot(**chatbot_default_settings)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=7000)

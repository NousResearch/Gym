import json
from pathlib import Path


class HermesAgentConfig:
    def __init__(self, **values):
        self.values = values


class FakeResponse:
    def __init__(self, model):
        self.output = [
            {
                "id": "msg-1",
                "content": [{"annotations": [], "text": "I will read the notes.", "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            },
            {
                "arguments": '{"path":"notes.md"}',
                "call_id": "call-1",
                "name": "read_file",
                "type": "function_call",
            },
            {
                "call_id": "call-1",
                "output": "The beta release deadline is June 1, 2024.",
                "type": "function_call_output",
            },
            {
                "id": "msg-2",
                "content": [{"annotations": [], "text": "Done.", "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            },
        ]
        self.payload = {
            "id": "fake-response",
            "created_at": 1,
            "model": model,
            "object": "response",
            "output": self.output,
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }

    def model_dump_json(self):
        return json.dumps(self.payload)


class HermesAgent:
    def __init__(self, config, server_client):
        self.config = config
        self.server_client = server_client

    async def responses(self, body):
        Path("answer.txt").write_text("The beta release deadline is June 1, 2024.\n")
        return FakeResponse(body.model)

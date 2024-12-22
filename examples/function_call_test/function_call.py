"""
docker run --runtime nvidia --gpus all \
    -v C:/Users/min/Documents/Qwen2.5-0.5B-Instruct:/model \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model mistralai/Mistral-7B-v0.1


"""
from transformers import  AutoTokenizer
from openai import OpenAI

from examples.openai_chat_completion_client import chat_completion

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=None,
    base_url=openai_api_base,
)


model = "qwen"


model_name = "C:/Users/min/Documents/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)


tools = [{
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type":
                    "string",
                    "description":
                    "The city to find the weather for, e.g. 'San Francisco'"
                },
                "state": {
                    "type":
                    "string",
                    "description":
                    "the two-letter abbreviation for the state that the city is"
                    " in, e.g. 'CA' which would mean 'California'"
                },
                "unit": {
                    "type": "string",
                    "description": "The unit to fetch the temperature in",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["city", "state", "unit"]
        }
    }
}]

messages = [{
    "role": "user",
    "content": "Hi! How are you doing today?"
}, {
    "role": "assistant",
    "content": "I'm doing well! How can I help you?"
}, {
    "role":
    "user",
    "content":
    "Can you tell me what the temperate will be in Dallas, in fahrenheit?"
}]

tool_completion = client.chat.completions.create(messages=messages,
                                                 model=model,
                                                 tools=tools)




text = tokenizer.apply_chat_template(
    messages, tools=tools,
    tokenize=False,
    add_generation_prompt=True
)
default_completion = client.completions.create(model=model,prompt=text)
print(default_completion)
# model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
#
#
# generated_ids = model.generate(
#     **model_inputs,
#     max_new_tokens=512
# )
#
# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]
#
# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

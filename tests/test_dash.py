import os
import dashscope
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env.dev"), override=True)
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

input_texts = "The quality of the clothes is excellent, very beautiful. It was worth the long wait. I like it and will come back to buy here again"

resp = dashscope.TextEmbedding.call(
    model="text-embedding-v4",
    input=input_texts,
)
print(resp)

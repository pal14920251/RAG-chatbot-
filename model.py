
from multiprocessing import context
import os
import traceback
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(dotenv_path=env_path)


class ChatModel:
    """
    Chat model wrapper using Hugging Face Inference API.
    """

    def __init__(self, model_id: str = "deepseek-ai/DeepSeek-R1"):
        """
        Initialize the Hugging Face inference client.
        Args:
            model_id: Hugging Face model name that supports text generation/chat.
                      (You can replace this with any other text-generation model)
        """
        self.hf_token = os.getenv("HF_API_TOKEN")
        print("HF_API_TOKEN found?", bool(self.hf_token))

        if not self.hf_token:
            raise ValueError(
                "HF_API_TOKEN not found in .env. Please add HF_API_TOKEN=your_token"
            )

       
        self.client = InferenceClient(model=model_id, token=self.hf_token)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=self.hf_token)
        self.model_id = model_id



    def generate(self, question: str, context: str = None, max_new_tokens: int = 250):
        if not question or question.strip() == "":
            return "Please provide a valid question."


        if context:
            prompt = f"""You are a helpful AI assistant.
            Use the following context to answer the question.

            Context: {context}
            Question: {question}"""
        else:
            prompt = f"""You are a helpful AI assistant.
            Question: {question}"""

        try:
           
            if "deepseek" in self.model_id.lower() or "chat" in self.model_id.lower():
                response = self.client.chat_completion(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_new_tokens,
                )
            
                if hasattr(response, "choices"):
                    return response.choices[0].message["content"].strip()
                elif isinstance(response, dict):
                    return response["choices"][0]["message"]["content"].strip()
                else:
                    return str(response)

         
            else:
                response = self.client.text_generation(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    stream=False
                )
                return response.strip() if response else " No response."
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f" Error: {str(e)}"




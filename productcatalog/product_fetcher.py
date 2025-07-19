import getpass
import os

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate


class ModelAdapter:
    def _init_llm(self):
        if not os.environ.get("GOOGLE_API_KEY"):
            raise Exception("API key missing for Google Gemini")
        self.model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

    def _init_prompt_config(self):
        self.system_template = "Translate the following from English into {language}"
        self.prompt_template = ChatPromptTemplate.from_messages(
            [("system", self.system_template), ("user", "{text}")]
        )
    def __init__(self):
        self._init_llm();
        self._init_prompt_config();

    def invoke(self, language, text):
        prompt = self.prompt_template.invoke({"language": language, "text": text})
        response = self.model.invoke(prompt)
        return response

adapter = ModelAdapter()
response = adapter.invoke("French", "hi!")
print(response.content)
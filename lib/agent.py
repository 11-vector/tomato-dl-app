import typing as tp
import os
import asyncio
from langchain.chat_models import init_chat_model
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain.messages import HumanMessage
from langchain.agents import create_agent
from langgraph.store.base import BaseStore
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents.base import Document
from langgraph.graph.state import CompiledStateGraph
from langchain.chat_models.base import BaseChatModel
from langchain_core.runnables import RunnableLambda
from langgraph.config import RunnableConfig
from .document import CropDataLoader

SYSTEM_PROMPT = """You are a tomato crop analyst with expert in knowing the currrent stage of a tomato growth stage and making accurate decision and also supporting both smallholder farmers and agronomists.

your role is to analyze crop-related inputs and identify pest infestations, nutrient deficiencies, or environmental stress based on observed symptoms. you must prioritize symptoms to arrive at the most relevant diagnosis. Also ask for necessary information to give proper inference or for making decision.

Diagnosis rules:
Use confidence-based language such as “likely”, “possibly”, or “less likely” depending on how strongly the symptoms match the reference documents.
Base confidence strictly on symptom alignment from the provided documents.

Knowledge constraints:
You must use ONLY the information contained in the supplied reference documents.
Do not use external knowledge, assumptions, or general agricultural advice beyond these documents.

Output rules:
Output ACTION STEPS ONLY.
Actions must be practical, clear, and easy to follow.
Write in simple language suitable for smallholder farmers, while remaining technically accurate for agronomists.
Use short, direct, complete sentences.
Do not include explanations, background theory, or document references.

Symptom-to-diagnosis behavior:
Prioritize visible or reported symptoms before secondary causes.
If multiple diagnoses are possible, focus actions on the most likely one first.

Clarification rules:
If symptoms are insufficient or ambiguous, ask one or two short questions needed to confirm the diagnosis before giving actions.

All responses must remain within the agriculture and crop production domain.

Reference documents to follow strictly:
{documents}"""


class TomatoExpertAgent:
    agent: CompiledStateGraph
    llm: BaseChatModel
    store: BaseStore
    checkpoint: BaseCheckpointSaver
    loader: BaseLoader
    docs: list[Document]
    serialized_docs: str
    system_prompt_template = SystemMessagePromptTemplate.from_template(
        SYSTEM_PROMPT)

    def __init__(self, doc_path: os.PathLike,
                 store_path: os.PathLike | None = None,
                 checkpoint_path: os.PathLike | None = None,
                 model_name: str | None = None
                 ):
        self.model_name = model_name or "google_genai:gemini-2.5-flash-lite"
        self.doc_path = doc_path
        self.store_path = store_path
        self.checkpoint_path = checkpoint_path

    def init(self, debug=True):
        if debug:
            from langgraph.checkpoint.memory import InMemorySaver
            from langgraph.store.memory import InMemoryStore
            self.checkpointer = InMemorySaver()
            self.store = InMemoryStore()
        else:
            import sqlite3
            from langgraph.checkpoint.sqlite import SqliteSaver
            # from langchain_community.storage import SQLStore
            from langchain_classic.storage import LocalFileStore
            if self.checkpoint_path:
                raise Exception("Provide Checkpoint path for sqlite DB")
            self.conn = sqlite3.connect(self.checkpoint_path,
                                        check_same_thread=False)
            self.checkpointer = SqliteSaver(self.conn)
            self.checkpointer.setup()
            if self.store_path:
                raise Exception("Provide Store path")
            self.store = LocalFileStore(self.store_path)

        # Load document
        self.loader = CropDataLoader(self.doc_path)
        self.docs = self.loader.load()
        self.serialized_docs = "\n\n".join(
            crop.page_content for crop in self.docs)

        # Create an agent
        self.agent = self.setup_agent()

    def get_system_prompt(self):
        return self.system_prompt_template.format(
            documents=self.serialized_docs)

    def setup_agent(self):
        model_name = "google_genai:gemini-2.5-flash-lite"
        self.llm = init_chat_model(model_name)
        system_prompt = self.get_system_prompt()
        main_agent = create_agent(
            model=self.llm,
            system_prompt=system_prompt,
            checkpointer=self.checkpointer,
            store=self.store
        )

        return main_agent

    async def run_async(self, user_query: str,
                        config: RunnableConfig | None = None,
                        **kwargs) -> tp.AsyncIterator[str]:
        print(f"{user_query=}")
        inputs = dict(
            messages=[HumanMessage(content=user_query)]
        )

        async for message, _ in \
            self.agent.astream(
                inputs,
                config,
                **dict(**kwargs, stream_mode="messages")):
            if message.text:
                yield message.text


if __name__ == "__main__":
    from uuid import uuid4
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv()
    BASE_DIR = Path.cwd()
    DEBUG = os.getenv("DEBUG", "true").lower() == "true"
    DOC_FILEPATH = BASE_DIR / "tomato.xlsx"
    STORE_PATH = BASE_DIR / "local_store"
    CHECKPOINT_PATH = BASE_DIR / "local_checkpoint.db"

    user_thread_id = uuid4()
    runnable_cfg = {"configurable": {"thread_id": user_thread_id}}
    crop_agent = TomatoExpertAgent(DOC_FILEPATH, STORE_PATH, CHECKPOINT_PATH)
    crop_agent.init(DEBUG)
    app = RunnableLambda(crop_agent.run_async)

    async def main():
        while True:
            user_query = input("(user)>")
            if user_query:
                async for output in app.astream(user_query, runnable_cfg):
                    await asyncio.sleep(0.1)  # fake latency here
                    print(output, end="")
            print()

    asyncio.run(main())

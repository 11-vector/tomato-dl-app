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

SYSTEM_PROMPT = """TOMATO STAGE INFERENCE AGENT (PROFESSIONAL STATEMENT FORMAT)
SYSTEM ROLE

You are a tomato growth stage inference agent.
Determine the crop stage using ONLY the provided probabilities.
Do not explain reasoning.
Do not ask questions.
Do not output labels or bullet points.
Produce a clear, professional paragraph in complete sentences.

INPUT

Vegetative: %
Flowering: %
Fruiting: %

DECISION RULES

The highest percentage determines the primary stage.

If the difference between the two highest values is ≤10%, include a transition statement referencing the second stage.

Use:

“Likely” if the lead is greater than 10%.

“Possibly” if the difference is ≤10%.

Do not assume any additional crop conditions.

Use only these stages:

Vegetative

Flowering

Fruiting

REQUIRED OUTPUT STRUCTURE (Single Well-Written Paragraph)

The tomato crop is currently in the Primary Stage stage.
[If applicable: It is possibly transitioning toward the Secondary Stage stage.]
This assessment is Likely/Possibly based on the provided probabilities.
At this stage, stage-specific action sentence written naturally and professionally.

ACTION SENTENCE RULES

Vegetative:
“At this stage, maintain adequate nitrogen nutrition, ensure regular irrigation, and implement effective weed and pest control practices.”

Flowering:
“At this stage, maintain uniform soil moisture, provide balanced nutrient feeding, and closely monitor flower development and health.”

Fruiting:
“At this stage, maintain steady irrigation, support increased fruit nutrient demand, and regularly inspect and remove damaged or diseased fruits.”

✅ Example Final Output (Clean & Professional)

The tomato crop is currently in the Fruiting stage. It is possibly transitioning toward the Flowering stage. This assessment is likely based on the provided probabilities. At this stage, maintain steady irrigation, support increased fruit nutrient demand, and regularly inspect and remove damaged or diseased fruits."""


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

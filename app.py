import os
from pathlib import Path
import gradio as gr
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tflite import TfliteInference
from langchain_core.runnables import RunnableLambda
from lib.agent import TomatoExpertAgent
from fastapi import FastAPI
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = Path.cwd()
DEBUG = os.getenv("DEBUG", "true").lower() == "true"
DOC_FILEPATH = BASE_DIR / "tomato.xlsx"
STORE_PATH = BASE_DIR / "local_store"
CHECKPOINT_PATH = BASE_DIR / "local_checkpoint.db"

crop_agent = TomatoExpertAgent(DOC_FILEPATH, STORE_PATH, CHECKPOINT_PATH)
crop_agent.init(DEBUG)
expert = RunnableLambda(crop_agent.run_async)

LABELS = ["Vegetative", "Flowering", "Fruiting"]
model = TfliteInference("hybrid.tflite", LABELS)
model.load_model()


async def callback_chatbot(message: gr.ChatMessage, history: list[gr.ChatMessage], image: np.ndarray, request: gr.Request):
    user_thread_id = request.session_hash
    runnable_cfg = {"configurable": {"thread_id": user_thread_id}}

    input_text = ""

    if message['text']:
        input_text += message["text"]

    images = []
    for fname in message['files']:
        images.append(plt.imread(fname))

    images = images if image is None else [*images, image]

    # perform image classification here
    if len(images) > 0:
        input_text += "this are the result of inference of the current tomato growth stage based on various section of the farm"

    for i, image in enumerate(images, start=1):
        resized_image = tf.image.resize(
            image, (256, 256), method=tf.image.ResizeMethod.BILINEAR)
        result = model.inference([np.expand_dims(resized_image, 0)])
        # merge the data
        input_text += f"\nsection-1: "
        input_text += str.join(", ", map(lambda entry: f"{entry[0]}: {
                               entry[1]:.6f}", result['labelled'].items()))

    output_text = ""
    if not input_text:
        yield gr.ChatMessage(
            content="provide input",
            role="assistant"
        )
        return

    async for text in expert.astream(input_text, config=runnable_cfg):
        output_text += f"{text} "
        yield gr.ChatMessage(
            content=output_text,
            role="assistant"
        )
    history.append(gr.ChatMessage(
        content=output_text,
        role="assistant"
    ))


def callback_predict(image: np.ndarray, state: str):
    if image is None:
        return "Provide image input", ""
    resized_image = tf.image.resize(
        image, (256, 256), method=tf.image.ResizeMethod.BILINEAR)
    result = model.inference([np.expand_dims(resized_image, 0)])

    output_text = "Inference Result on image:\n"
    output_text += str.join("\n", map(lambda entry: f"{entry[0]}: {
        entry[1]:.6f}", result['labelled'].items()))
    return output_text, output_text


label_output = gr.Text("Predicition Result:")
image_input = gr.Image(
    sources=["upload", "webcam"],
)

image_input2 = gr.Image(
    sources=["upload", "webcam"],
)

state = gr.State([])
chat_block = gr.ChatInterface(
    fn=callback_chatbot,
    multimodal=True,
    # save_history=True,
    chatbot=gr.Chatbot(
        value=[
            gr.ChatMessage(
                role="assistant",
                content="Welcome to tomato analyst.",
            )
        ],
        height=300),
    textbox=gr.MultimodalTextbox(
        value=None,
        file_types=["image"],
        sources=["upload"]),
    description="Tomato Assistant chatbot with multimodal input",
    additional_inputs_accordion=gr.Accordion(open=True),
    additional_inputs=[image_input],
)

inference_block = gr.Interface(
    callback_predict, [image_input2, state], [label_output, state])

block = gr.TabbedInterface(
    [chat_block, inference_block],
    ["Chat", "Inference"],
    title="Tomato Assistant AI",
)


app = FastAPI()
app = gr.mount_gradio_app(app, block, path="/")

if __name__ == "__main__":
    block.launch(debug=True, share=True, )

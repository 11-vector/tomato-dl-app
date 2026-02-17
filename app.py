import os
from pathlib import Path
import gradio as gr
import numpy as np
import pandas as pd
import tensorflow as tf
from langchain_core.runnables import RunnableLambda
from fastapi import FastAPI
from tflite import TfliteInference
from lib.agent import TomatoExpertAgent
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


LANGUAGES = "English Yoruba Igbo Hausa French Spanish".split(" ")


async def realtime_inference(
    image: np.ndarray,
    language: str,
    request: gr.Request,
    state: str = ""
):
    user_thread_id = request.session_hash
    runnable_cfg = {"configurable": {"thread_id": user_thread_id}}

    input_text = f"Given the current tomato field stage give out information only in {
        language} language\n"

    # perform image classification here
    if image is None:
        return "No input image provide", None, state

    resized_image = tf.image.resize(
        image, (256, 256), method=tf.image.ResizeMethod.BILINEAR)
    result = model.inference([np.expand_dims(resized_image, 0)])
    # merge the data
    result = {
        key: f"{value*100:.4f}"
        for (key, value) in result['labelled'].items()}
    input_text += str.join("\n",
                           map(lambda entry: f"{entry[0]}: {entry[1]}%", result.items()))

    output_text = ""

    try:
        async for text in expert.astream(input_text, config=runnable_cfg):
            output_text += f"{text} "
    except:
        gr.Warning('Unable to get response from NLP: Reached token limit')

    return output_text, pd.DataFrame([result]), state


def select_input_mode(value):
    if value == "Image":
        return gr.Image(label="Input", sources=["upload"])
    return gr.Image(label="Input", sources=["webcam"])


initial = pd.DataFrame([], columns=["Vegetative", "FLowering", "Fruiting"])

with gr.Blocks() as inference_block:
    state = gr.State([])
    with gr.Column():
        input_mode = gr.Dropdown(["Streaming", "Image"])
        with gr.Row():
            @gr.render(inputs=input_mode)
            def render_row(mode: str):
                with gr.Column():
                    language = gr.Dropdown(LANGUAGES, interactive=True)
                    image_input = gr.Image(
                        key=mode,
                        label="Input", sources=[
                            "upload" if mode == "Image" else "webcam"])
                with gr.Column():
                    label_output = gr.Textbox("Predicition Result:", lines=9)
                    prediction = gr.DataFrame(initial)

                    if mode == "Image":
                        image_input.upload(
                            realtime_inference,
                            [image_input, language, state], [
                                label_output, prediction, state],
                        )
                    else:
                        image_input.stream(
                            realtime_inference,
                            [image_input, language, state], [
                                label_output, prediction, state],
                            time_limit=30,
                            stream_every=60,
                            concurrency_limit=1
                        )


block = gr.TabbedInterface(
    [inference_block],
    ["Inference"],
    title="Tomato Project",
)


app = FastAPI()
app = gr.mount_gradio_app(app, block, path="/")

if __name__ == "__main__":
    block.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
    )

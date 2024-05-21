import gradio as gr

import logo
import design

with gr.Blocks() as demo:
    gr.Markdown(
    """
    # Resultats experimentations UC Generation Design/Video
    """
    )
    with gr.Tab("Logo"):
        logo.interface.run()
    with gr.Tab("Design"):
        design.interface.run()

demo.launch(share=True)
import gradio as gr
import logo
import design

with gr.Blocks() as demo:
    gr.Markdown(
    """
    # Resultats experimentations UC Generation Desigm/Video
    """
    )
    with gr.Tab("Logo"):
        logo.displayLogo()
    with gr.Tab("Design"):
        design.displayDesign()

demo.launch()
import gradio as gr

import logo
import design

demo = gr.TabbedInterface(
    [logo.interface.run(), design.interface.run()], ["Logo", "Design"]
)

demo.launch(share=True)

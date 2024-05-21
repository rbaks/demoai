import gradio as gr
import logo

demo = gr.TabbedInterface([logo.run()], ["Logo"])

demo.launch(share=True)

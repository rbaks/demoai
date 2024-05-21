import gradio as gr
import inference


def run():
    gr.Markdown(
        """
    ## Generation conditionnelle de logos
    Modele Text-To-Image avec [`stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) (SDXL) + Surcouche de parametres apres FineTuning LoRA de [`artificialguybr/LogoRedmond-LogoLoraForSDXL-V2`](https://huggingface.co/artificialguybr/LogoRedmond-LogoLoraForSDXL-V2) pour les logos.
    """
    )
    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(label="Prompt textuel")
        with gr.Column(scale=1, min_width=50):
            batch_size = gr.Number(label="Nombre de logos a generer.", value=10)
    with gr.Accordion("Options avancees", open=False):
        with gr.Row():
            with gr.Column(scale=2):
                negative_prompt = gr.Textbox(
                    label="Negative prompt",
                    info="Mots cles des resultats a eviter. A separer par ','",
                )
            with gr.Column(scale=1):
                scheduler = gr.Radio(
                    ["EulerDiscreteScheduler", "EulerAncestralDiscreteScheduler"],
                    label="Scheduler",
                    value="EulerDiscreteScheduler",
                    info="Differents Schedulers proposent differents style d'images generees",
                )
        with gr.Row():
            with gr.Column():
                steps = gr.Slider(
                    label="Inference Steps",
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1,
                    info="Plus significatif -> plus de details et realisme, mais prends plus de temps",
                )
                guidance = gr.Slider(
                    label="Guidance Scale",
                    minimum=1,
                    maximum=50,
                    value=5.0,
                    step=0.5,
                    info="A quel degre le modele de generation d'image devra-t-il respecter le prompt?",
                )
            with gr.Column():
                width = gr.Slider(
                    label="Largeur", minimum=64, maximum=1024, step=64, value=512
                )
                height = gr.Slider(
                    label="Hauteur", minimum=64, maximum=1024, step=64, value=512
                )
    btn = gr.Button("Generer", size="lg")
    output = gr.Gallery(label="Resultats", columns=5)
    btn.click(
        fn=inference.run,
        inputs=[
            prompt,
            batch_size,
            negative_prompt,
            scheduler,
            steps,
            guidance,
            width,
            height,
        ],
        outputs=[output],
    )
    examples = gr.Examples(
        examples=[
            "A circular logo of a large praying mantis, close up, detailed, vibrant and fun colors",
            "A circular logo of a strong man playing basketball, close up, detailed, vibrant and fun colors",
        ],
        inputs=[prompt],
    )


if __name__ == "__main__":
    with gr.Blocks() as demo:
        run()
    demo.launch()

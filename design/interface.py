import gradio as gr
import inference


def run():
    gr.Markdown(
        """
    ## Generation conditionnelle (Texte + Image Wireframe) de designs de sites web
    Modele Text-To-Image avec [`stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) (SDXL) + Surcouche de parametres de Controlnet de [`TheMistoAI/MistoLine`](https://huggingface.co/TheMistoAI/MistoLine) pour le conditionnement avec wireframe.

    **A faire :** Training de surcouche de finetuning LoRA pour amelioration de la qualite des maquettes generes:

    - [x] Collecte de donnees de paires : (Screenshot site web + Texte descriptif)
    - [ ] Training du LoRA
    - [ ] Fusion avec le modele actuel conditionnee par Controlnet

    """
    )
    with gr.Row():
        with gr.Column():
            with gr.Group():
                with gr.Row():
                    with gr.Column(scale=3):
                        negative_prompt = gr.Textbox(
                            label="Negative prompt",
                            info="Mots cles des resultats a eviter. A separer par ','",
                        )
                    with gr.Column(scale=1):
                        batch_size = gr.Number(
                            label="Batch Size",
                            info="Nombre de maquettes a generer.",
                            value=1,
                        )

                with gr.Accordion("Options avancees", open=True):
                    inference_steps = gr.Slider(
                        label="Inference Steps",
                        minimum=1,
                        maximum=100,
                        value=50,
                        step=1,
                        info="Plus significatif -> plus de details et realisme, mais prends plus de temps",
                    )
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=1,
                        maximum=50,
                        value=10.0,
                        step=0.5,
                        info="A quel degre le modele de generation d'image devra-t-il respecter le prompt?",
                    )
                    controlnet_conditionning_scale = gr.Slider(
                        label="Contronel Conditionning Scale",
                        minimum=0,
                        maximum=1.0,
                        value=0.75,
                        step=0.05,
                        info="A quel degre le modele de generation d'image devra-t-il respecter le wireframe?",
                    )
                    with gr.Row():
                        with gr.Column():
                            width = gr.Slider(
                                label="Largeur",
                                minimum=64,
                                maximum=1024,
                                step=64,
                                value=512,
                            )
                        with gr.Column():
                            height = gr.Slider(
                                label="Hauteur",
                                minimum=64,
                                maximum=1024,
                                step=64,
                                value=512,
                            )
        with gr.Column():
            wireframe_img = gr.Image(label="Wireframe")
            prompt = gr.Textbox(label="Prompt textuel")
            scheduler = gr.Radio(
                ["EulerDiscreteScheduler", "EulerAncestralDiscreteScheduler"],
                label="Scheduler",
                value="EulerDiscreteScheduler",
                info="Differents Schedulers proposent differents style d'images generees",
            )
    btn = gr.Button("Generer", size="lg")
    output = gr.Gallery(label="Resultats", columns=5)
    btn.click(
        fn=inference.run,
        inputs=[
            prompt,
            wireframe_img,
            batch_size,
            negative_prompt,
            controlnet_conditionning_scale,
        ],
        outputs=[output],
    )


if __name__ == "__main__":
    with gr.Blocks() as demo:
        run()
    demo.launch()

import os
import sys
import gradio as gr
from transformers import pipeline

# Set up environment for GPU usage (Optional if using `device_map="auto"`)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def create_generation_pipeline(model_id="meta-llama/Llama-3.2-1B-Instruct"):
    return pipeline(
        "text-generation",
        model=model_id,
        device_map="auto",  # Automatically assigns devices
    )

# Handle command-line arguments with a default model
model_id = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-3.2-1B-Instruct"
pipe = create_generation_pipeline(model_id=model_id)

def generate(input_text, slider_value):
    # Use the pipeline to generate text
    results = pipe(input_text, max_new_tokens=slider_value)
    output_text = results[0]["generated_text"]  # Extract the text from the result
    return f"Input: {input_text}\nOutput: {output_text}"

def terminal_chat():
    print("Chatbot Terminal Mode (type 'exit' to quit):")
    while True:
        user_input = input("\nEnter your prompt: ")
        if user_input.lower() == "exit":
            print("Exiting terminal chat.")
            break
        try:
            formatted_input = f"Please respond concisely to: {user_input}"
            response = pipe(formatted_input, max_new_tokens=50, temperature=0.7, top_p=0.9)
            output_text = response[0]["generated_text"].strip()
            print(f"Bot: {output_text}")
        except Exception as e:
            print(f"Error: {e}")

def gradio_chat():
    # Create the Gradio interface
    demo = gr.Interface(
        fn=generate,
        inputs=[
            gr.Textbox(label="Prompt"), 
            gr.Slider(label="Max new tokens", value=20, maximum=1024, minimum=1)
        ],
        outputs=gr.Textbox(label="Completion")
    )
    # Launch the interface
    demo.launch(share=True, server_port=8080)

if __name__ == "__main__":
    # Ask the user to choose the interaction mode
    print("Choose an interaction mode:")
    print("1: Terminal Chat")
    print("2: Gradio Chat (Web Interface)")
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        terminal_chat()
    elif choice == "2":
        gradio_chat()
    else:
        print("Invalid choice. Please restart the script and choose 1 or 2.")

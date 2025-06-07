import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# --- App Setup ---
st.title("Batnice🚀")
st.caption("Powered by Microsoft's Phi-3-mini")

# --- Model Loading ---
@st.cache_resource
def load_model():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    return model, tokenizer

model, tokenizer = load_model()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Message Batnice.."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Manually format Phi-3 prompt (bypassing jinja2 error)
            formatted_prompt = ""
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    formatted_prompt += f"<|user|>\n{msg['content']}<|end|>\n<|assistant|>\n"
                else:
                    formatted_prompt += f"{msg['content']}<|end|>\n"
            
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=560,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Clean response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
            response = full_response.split("<|assistant|>")[-1].replace("<|end|>", "").strip()
        
        st.write(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

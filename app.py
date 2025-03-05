import streamlit as st
import torch
from code_to_pseudocode.model import load_translation_model as load_code_to_pseudo_model
from code_to_pseudocode.model import codetopsd
from pseudocode_to_code.model import load_translation_model as load_pseudo_to_code_model
from pseudocode_to_code.model import psuedocodetocode
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model and tokenizer paths
MODEL_PATHS = {
    'code_to_pseudo': os.path.join(BASE_DIR, 'code_to_pseudocode', 'code_psd_transformer.pt'),
    'pseudo_to_code': os.path.join(BASE_DIR, 'pseudocode_to_code', 'psd_code_transformer.pt'),
    'code_sp_model': os.path.join(BASE_DIR, 'code_to_pseudocode', 'code_tokenizer.model'),
    'psd_sp_model': os.path.join(BASE_DIR, 'code_to_pseudocode', 'text_tokenizer.model'),
    'text_sp_model': os.path.join(BASE_DIR, 'pseudocode_to_code', 'text_tokenizer.model'),
    'code_sp_model2': os.path.join(BASE_DIR, 'pseudocode_to_code', 'code_tokenizer.model')
}

# Initialize models
def load_models():
    # Load code to pseudocode model
    code_to_pseudo_model = load_code_to_pseudo_model(
        model_path=MODEL_PATHS['code_to_pseudo'],
        code_spPath=MODEL_PATHS['code_sp_model'],
        text_spPath=MODEL_PATHS['psd_sp_model']
    )
    
    # Load pseudocode to code model
    pseudo_to_code_model = load_pseudo_to_code_model(
        model_path=MODEL_PATHS['pseudo_to_code'],
        text_spPath=MODEL_PATHS['text_sp_model'],
        code_spPath=MODEL_PATHS['code_sp_model2']
    )
    
    return code_to_pseudo_model, pseudo_to_code_model

# Modify the callback function
def set_example(text, direction):
    st.session_state.input_text = text
    st.session_state.desired_direction = direction
    st.session_state.update_requested = True

if __name__ == "__main__":
    # Initialize session state
    if 'input_text' not in st.session_state:
        st.session_state.input_text = ""
    if 'desired_direction' not in st.session_state:
        st.session_state.desired_direction = "Code → Pseudocode"
    if 'update_requested' not in st.session_state:
        st.session_state.update_requested = False

    # Load models
    try:
        code_to_pseudo_model, pseudo_to_code_model = load_models()
        st.success("✅ Models loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()

    # Title
    st.title("Code ↔ Pseudocode Translator")

    # Create two columns
    col1, col2 = st.columns(2)

    # Left column for input
    with col1:
        st.subheader("Input")
        
        # Update radio button initialization
        initial_index = 0 if st.session_state.desired_direction == "Code → Pseudocode" else 1
        translation_direction = st.radio(
            "Select Translation Direction:",
            ["Code → Pseudocode", "Pseudocode → Code"],
            index=initial_index,
            key='translation_direction'
        )

        # Update desired_direction based on radio button
        st.session_state.desired_direction = translation_direction
        
        input_text = st.text_area(
            "Enter your code or pseudocode here:",
            value=st.session_state.input_text,
            height=300,
            help="Type or paste your input text here",
        )
        st.session_state.input_text = input_text

    # If update was requested, trigger rerun once
    if st.session_state.update_requested:
        st.session_state.update_requested = False
        st.rerun()

    # Right column for output
    with col2:
        st.subheader("Output")
        if st.button("Translate"):
            if input_text.strip():
                try:
                    with st.spinner("Translating..."):
                        if translation_direction == "Code → Pseudocode":
                            output = codetopsd(code_to_pseudo_model, input_text)
                            st.text_area("Pseudocode:", value=output, height=300)
                        else:
                            output = psuedocodetocode(pseudo_to_code_model, input_text)
                            st.text_area("Code:", value=output, height=300)
                except Exception as e:
                    st.error(f"Translation error: {str(e)}")
            else:
                st.warning("Please enter some text to translate!")

    # Examples section
    with st.expander("Show Examples"):
        st.markdown("Click on any example to paste it into the input box.")
        
        # Define examples
        code_examples = {
            "For Loop Sum": """for(int i=0; i<n; i++) { 
    sum += arr[i]; 
}""",
            "Input": "cin>>a",
            "For Loop One-line": "for(int i=0; i<n; i++) { sum += arr[i]; }",
            "If-Else": "if(x > y) { max = x; } else { max = y; }",
            "While Queue": "while(!q.empty()) { int node = q.front(); q.pop(); }",
            "GCD Function": "int gcd(int a, int b) { return b == 0 ? a : gcd(b, a % b); }"
        }

        pseudo_examples = {
            "Array Sum": """set sum to 0
for each element in array:
    add element to sum""",
            "Print Hello": "print 'hello'",
            "Conditional Print": "if x is greater than 10 then print x",
            "Array Loop": """set arr to [1, 2, 3, 4, 5]
for each num in arr do print num"""
        }

        # Create two columns for examples
        code_col, pseudo_col = st.columns(2)
        
        with code_col:
            st.markdown("### Code Examples")
            for title, code in code_examples.items():
                if st.button(f"📝 {title}", key=f"code_{title}"):
                    set_example(code, "Code → Pseudocode")
                    st.rerun()
                st.code(code, language="cpp")
                st.markdown("---")

        with pseudo_col:
            st.markdown("### Pseudocode Examples")
            for title, pseudo in pseudo_examples.items():
                if st.button(f"📝 {title}", key=f"pseudo_{title}"):
                    set_example(pseudo, "Pseudocode → Code")
                    st.rerun()
                st.code(pseudo)
                st.markdown("---")

    # Footer
    st.markdown("---")
    st.markdown(
        "Made with ❤️ using Transformer Models | "
        "[GitHub Repository](https://github.com/yourusername/your-repo)"
    )



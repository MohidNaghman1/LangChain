from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate
import os


load_dotenv()

# Get token and verify it exists
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    st.error("HUGGINGFACEHUB_API_TOKEN not found in environment variables")
    st.stop()

def create_model(repo_id):
    """Create HuggingFace model with proper error handling"""
    try:
        llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            temperature=0.7,
            max_new_tokens=512,  # Reduced for better reliability
            timeout=60,
            huggingfacehub_api_token=hf_token
        )
        return ChatHuggingFace(llm=llm, verbose=True)
    except Exception as e:
        st.error(f"Failed to initialize model {repo_id}: {str(e)}")
        return None

def load_prompt():
    """Load prompt template"""
    template_text = """
You are an expert research assistant specializing in AI and machine learning papers.

Research Paper: {paper_input}
Explanation Style: {style_input}  
Explanation Length: {length_input}

Please provide a comprehensive summary and explanation of the research paper "{paper_input}" following these requirements:

Style Guidelines:
- If Beginner-Friendly: Use simple language, avoid jargon, explain concepts clearly
- If Technical: Use precise technical terminology and detailed explanations
- If Code-Oriented: Focus on implementation details and code examples where relevant
- If Mathematical: Include mathematical formulations and theoretical foundations

Length Requirements:
- If Short: Provide 1-2 concise paragraphs covering key points
- If Medium: Provide 3-5 paragraphs with detailed explanations
- If Long: Provide a comprehensive detailed explanation with multiple sections

Focus on:
1. Main contributions and innovations
2. Key technical details
3. Significance and impact
4. Practical applications

Summary:"""

    return PromptTemplate(
        input_variables=["paper_input", "style_input", "length_input"],
        template=template_text
    )

st.header('Research Tool')

# Model selection with more reliable models
model_options = {
    "FLAN-T5-Base": "google/flan-t5-base",  # More reliable than large
    "FLAN-T5-Large": "google/flan-t5-large",
    "GPT-2": "gpt2",  # Very reliable
    "DistilGPT-2": "distilgpt2" # Fast and reliable
}


selected_model = st.selectbox(
    "Select AI Model",
    list(model_options.keys()),
    index=0
)

# Initialize model
repo_id = model_options[selected_model]
model = create_model(repo_id)

if not model:
    st.error("Failed to initialize the selected model. Please try a different model.")
    st.stop()

paper_input = st.selectbox(
    "Select Research Paper Name",
    ["Attention Is All You Need", 
     "BERT: Pre-training of Deep Bidirectional Transformers", 
     "GPT-3: Language Models are Few-Shot Learners", 
     "Diffusion Models Beat GANs on Image Synthesis"]
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

template = load_prompt()

if st.button('Summarize'):
    with st.spinner('Generating summary...'):
        try:
            chain = template | model
            
            # Add timeout and better error handling
            result = chain.invoke({
                'paper_input': paper_input,
                'style_input': style_input,
                'length_input': length_input
            })
            
            st.write(result.content)
            
        except Exception as e:
            error_msg = str(e).lower()
            
            if "model" in error_msg and "loading" in error_msg:
                st.warning("⏳ Model is loading on HuggingFace servers. Please wait a moment and try again.")
                st.info("💡 Tip: Try using GPT-2 or DistilGPT-2 models as they load faster.")
            elif "rate limit" in error_msg or "429" in error_msg:
                st.warning("⏳ Rate limit reached. Please wait a moment and try again.")
            elif "timeout" in error_msg:
                st.warning("⏳ Request timed out. The model might be busy. Please try again.")
            else:
                st.error(f"Error: {str(e)}")
                st.info("💡 Try selecting a different model or check your internet connection.")
                
        # Add retry button
        if st.button("🔄 Retry"):
            st.rerun()
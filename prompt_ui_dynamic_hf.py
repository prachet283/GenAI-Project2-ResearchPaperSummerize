from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate , load_prompt


load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id='mistralai/Mistral-7B-Instruct-v0.2',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

st.header("Research Tool")

paper_input = st.selectbox("Select Research Paper Name",["Select...","Attention Is All You Need","BERT : Pre-training of Deep Bidirectional Transformers","GPT-3: Language Models are Few-Shot Learners","Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox("Select Explaination Style",["Begineer-Friendly","Technical","Code-Oriented","Mathematical"])

length_input = st.selectbox("Select Explaination Length",["Short (1-2 paragraphs)","Medium (3-5 paragraphs)","Long (detailed explaination)"])

template = load_prompt('template.json')

# prompt = template.invoke({
#     'paper_input' : paper_input,
#     'style_input' : style_input,
#     'length_input' : length_input
# })

# if st.button("Summerize"):
#     result = model.invoke(prompt)
#     st.write(result.content)


# chaining part , here you are calling invoke multiple times rather than that we can use chains



if st.button("Summerize"):
    chain = template | model
    result = chain.invoke({
    'paper_input' : paper_input,
    'style_input' : style_input,
    'length_input' : length_input
    })
    st.write(result.content)
import gradio as gr
from crewai import Agent, Task, Crew
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os
import faiss
import numpy as np

# Load environment variables
load_dotenv()
os.environ["TAVILY_API_KEY"] = "your-tavily-api-key"

# Initialize FAISS index
dimension = 384  # Matches sentence-transformers/all-MiniLM-L6-v2
index = faiss.IndexFlatL2(dimension)

# Initialize embeddings for FAISS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize main LLM (Llama-3.1-8B)
main_pipeline = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.1-8B",
    max_new_tokens=1500,
    temperature=0.7
)
main_llm = HuggingFacePipeline(pipeline=main_pipeline)

# Initialize lightweight model for prompt suggestions (GPT-2)
suggestion_tokenizer = AutoTokenizer.from_pretrained("gpt2")
suggestion_model = AutoModelForCausalLM.from_pretrained("gpt2")

# Initialize TavilySearch tool
search_tool = TavilySearchResults()

# Define agents with FAISS for knowledge storage
researcher = Agent(
    role="Researcher",
    goal="Find recent information based on user query about AI trends",
    backstory="Expert in web research and data synthesis",
    llm=main_llm,
    tools=[search_tool],
    knowledge=index  # Use FAISS index
)

outliner = Agent(
    role="Outliner",
    goal="Create a concise outline based on research",
    backstory="Skilled in structuring content for clarity",
    llm=main_llm,
    knowledge=index
)

writer = Agent(
    role="Writer",
    goal="Generate a detailed response based on the outline",
    backstory="Experienced in crafting engaging tech content",
    llm=main_llm,
    knowledge=index
)

# Function to generate prompt suggestions
def suggest_prompts(partial_input):
    inputs = suggestion_tokenizer(partial_input, return_tensors="pt")
    outputs = suggestion_model.generate(**inputs, max_new_tokens=20, num_return_sequences=3)
    suggestions = [suggestion_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return "\n".join(suggestions)

# Function to process user query
def process_query(query):
    # Embed query for FAISS
    query_embedding = embeddings.embed_query(query)
    index.add(np.array([query_embedding], dtype=np.float32))

    research_task = Task(
        description=f"Search for and summarize information on: {query}",
        expected_output="Bullet-point summary of key points",
        agent=researcher
    )
    outline_task = Task(
        description="Create a concise outline with 5 subheadings based on research",
        expected_output="Outline with 5 subheadings",
        agent=outliner,
        context=[research_task]
    )
    write_task = Task(
        description="Write a detailed response based on the outline",
        expected_output="Detailed response (up to 1000 words)",
        agent=writer,
        context=[outline_task]
    )
    crew = Crew(
        agents=[researcher, outliner, writer],
        tasks=[research_task, outline_task, write_task],
        process="sequential"
    )
    result = crew.kickoff()
    return result

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# AI Trends Q&A")
    with gr.Row():
        query_input = gr.Textbox(label="Ask about AI trends", placeholder="Type your question...")
        suggestion_output = gr.Textbox(label="Prompt Suggestions", interactive=False)
    
    query_input.change(
        fn=suggest_prompts,
        inputs=query_input,
        outputs=suggestion_output
    )
    
    submit_button = gr.Button("Submit")
    output = gr.Textbox(label="Response")
    
    submit_button.click(
        fn=process_query,
        inputs=query_input,
        outputs=output
    )

# Launch interface
demo.launch()

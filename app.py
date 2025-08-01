import gradio as gr
from crewai import Agent, Task, Crew
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Load environment variables
load_dotenv()
os.environ["TAVILY_API_KEY"] = "your-tavily-api-key"

# Initialize chromadb client with sentence-transformers
chroma_client = chromadb.Client()
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Initialize main LLM (EleutherAI/gpt-neo-2.7B)
main_pipeline = pipeline(
    "text-generation",
    model="EleutherAI/gpt-neo-2.7B",
    max_new_tokens=1000,
    temperature=0.7
)
main_llm = HuggingFacePipeline(pipeline=main_pipeline)

# Initialize lightweight model for prompt suggestions (GPT-2)
suggestion_tokenizer = AutoTokenizer.from_pretrained("gpt2")
suggestion_model = AutoModelForCausalLM.from_pretrained("gpt2")

# Initialize TavilySearch tool
search_tool = TavilySearchResults()

# Define agents
researcher = Agent(
    role="Researcher",
    goal="Find recent information based on user query about AI trends",
    backstory="Expert in web research and data synthesis",
    llm=main_llm,
    tools=[search_tool],
    allow_knowledge=False,
    allow_delegation=False
)

outliner = Agent(
    role="Outliner",
    goal="Create a concise outline based on research",
    backstory="Skilled in structuring content for clarity",
    llm=main_llm,
    allow_knowledge=False,
    allow_delegation=False
)

writer = Agent(
    role="Writer",
    goal="Generate a detailed response based on the outline",
    backstory="Experienced in crafting engaging tech content",
    llm=main_llm,
    allow_knowledge=False,
    allow_delegation=False
)

# Function to generate prompt suggestions
def suggest_prompts(partial_input):
    inputs = suggestion_tokenizer(partial_input, return_tensors="pt")
    outputs = suggestion_model.generate(**inputs, max_new_tokens=20, num_return_sequences=3)
    suggestions = [suggestion_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return "\n".join(suggestions)

# Function to get top reference websites
def get_reference_websites():
    websites = [
        {"name": "Scite", "url": "https://scite.ai", "description": "AI-powered tool for analyzing citations and research reliability"},
        {"name": "Elicit", "url": "https://elicit.com", "description": "AI research assistant for summarizing and extracting data from papers"},
        {"name": "Semantic Scholar", "url": "https://www.semanticscholar.org", "description": "AI-driven search engine for academic literature"},
        {"name": "ResearchRabbit", "url": "https://www.researchrabbit.ai", "description": "Visualizes citation networks for literature reviews"},
        {"name": "KDnuggets", "url": "https://www.kdnuggets.com", "description": "Blog for AI, machine learning, and data science trends"}
    ]
    return "\n".join([f"- [{site['name']}]({site['url']}): {site['description']}" for site in websites])

# Function to process user query
def process_query(query):
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
    references = get_reference_websites()
    return str(result), references

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# AI Trends Q&A with Top Reference Websites")
    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(label="Ask about AI trends", placeholder="Type your question...")
            suggestion_output = gr.Textbox(label="Prompt Suggestions", interactive=False)
            submit_button = gr.Button("Submit")
        with gr.Column(scale=1):
            reference_output = gr.Markdown(label="Top Reference Websites")
    
    query_input.change(
        fn=suggest_prompts,
        inputs=query_input,
        outputs=suggestion_output
    )
    
    output = gr.Textbox(label="Response")
    
    submit_button.click(
        fn=process_query,
        inputs=query_input,
        outputs=[output, reference_output]
    )


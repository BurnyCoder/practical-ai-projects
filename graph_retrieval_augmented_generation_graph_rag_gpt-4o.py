import os
import subprocess
from dotenv import load_dotenv

# Load API keys
load_dotenv()
GRAPHRAG_API_KEY = os.getenv("GRAPHRAG_API_KEY")

class GraphRAG:
    def __init__(self, input_file="input.txt"):
        self.input_file = input_file

    def setup(self):
        print("Setting up GraphRAG...")
        setup_graph_rag(self.input_file)

    def create_graph(self):
        print("Creating the graph...")
        create_graph(self.input_file)

    def query_graph(self, query, method='local'):
        print(f"Querying the graph: '{query}'")
        result = use_constructed_graph(query, method=method)
        print("Result:")
        print(result)

def setup_graph_rag(input_file):
    # Create directory and navigate into it
    os.makedirs(os.path.join(os.getcwd(), "graph_rag"), exist_ok=True)
    os.chdir(os.path.join(os.getcwd(), "graph_rag"))

    # Install GraphRAG
    subprocess.run(["pip", "install", "openai", "networkx", "leidenalg", "cdlib", "python-igraph", "python-dotenv"])

    # Create input directory
    os.makedirs(os.path.join(os.getcwd(), "ragtest", "input"), exist_ok=True)

    # Use input_file as data source
    with open(os.path.join(os.getcwd(), "ragtest", "input", "input.txt"), "w", encoding="utf-8") as f:
        with open(os.path.join(os.path.dirname(os.getcwd()), input_file), "r", encoding="utf-8") as source_file:
            f.write(source_file.read())

    # Initialize GraphRAG
    subprocess.run(["python", "-m", "graphrag.index", "--init", "--root", os.path.join(os.getcwd(), "ragtest")])
    
    # Write the .env file with the GRAPHRAG_API_KEY
    with open(os.path.join(os.getcwd(), "ragtest", ".env"), "w") as env_file:
        env_file.write(f'GRAPHRAG_API_KEY="{GRAPHRAG_API_KEY}"')

    # Update the settings.yaml file to change the model to gpt-4o
    settings_path = os.path.join(os.getcwd(), "ragtest", "settings.yaml")
    with open(settings_path, "r") as file:
        settings_content = file.read()
    
    # Replace the model line
    settings_content = settings_content.replace("model: gpt-4-turbo-preview", "model: gpt-4o")
    
    with open(settings_path, "w") as file:
        file.write(settings_content)
    
    print("Updated settings.yaml: Changed model to gpt-4o")

def create_graph(input_file):
    # Ensure the input file exists in the correct location
    input_dir = os.path.join(os.getcwd(), "ragtest", "input")
    if not os.path.exists(os.path.join(input_dir, "input.txt")):
        with open(os.path.join(input_dir, "input.txt"), "w", encoding="utf-8") as f:
            with open(input_file, "r", encoding="utf-8") as source_file:
                f.write(source_file.read())

    subprocess.run(["python", "-m", "graphrag.index", "--root", os.path.join(os.getcwd(), "ragtest")])

def use_constructed_graph(query, method='local'):
    if method not in ['local', 'global']:
        raise ValueError("Method must be either 'local' or 'global'")
    
    result = subprocess.run(
        ["python", "-m", "graphrag.query", "--root", os.path.join(os.getcwd(), "ragtest"), "--method", method, query],
        capture_output=True,
        text=True
    )
    return result.stdout

def test_small():
    input_file = "input_small.txt"  # You can change this to any input file path
    graph_rag = GraphRAG(input_file)
    graph_rag.setup()
    graph_rag.create_graph()
    query = "What is near?"
    graph_rag.query_graph(query)
    
def test_math(create_and_setup=False):
    input_file = "input_small.txt"  # You can change this to any input file path
    graph_rag = GraphRAG(input_file)
    if create_and_setup:
        graph_rag.setup()
        graph_rag.create_graph()
    query = "What is near?"
    graph_rag.query_graph(query, method='local')

if __name__ == "__main__":
    test_math(create_and_setup = False)
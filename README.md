# Practical AI Projects

Collection of practical AI applications and implementations using various machine learning, deep learning, and natural language processing techniques.

## Overview

This repository contains a diverse set of practical AI projects that demonstrate the application of AI in various domains. The projects cover a wide range of techniques including:

- Large language model (LLM) training fine-tuning
- Retrieval Augmented Generation (RAG)
- Multi-agent AI systems
- Computer vision applications
- Natural language processing
- Sentiment analysis
- Recommendation systems
- Clustering and dimensionality reduction

## Project Structure

### Language Model Training and Fine-tuning

- `training-large-language-model-for-esperanto.ipynb`: Training a RoBERTa-like large language model (LLM) from scratch for Esperanto using custom tokenization and masked language modeling.
- `language_model_finetuning_qlora_llama2.ipynb`: Fine-tuning Llama 2 using QLoRA technique.
- `finetuning_img_classifier_visual_transformer_lora.ipynb`: Fine-tuning a visual transformer model using LoRA.

### Retrieval Augmented Generation (RAG)

- `retrieval_augmented_generation_agent_llm_llamaindex_gpt-4o.py`: Implements a RAG agent using LlamaIndex and GPT-4o.
- `graph_retrieval_augmented_generation_graph_rag_gpt-4o.py`: Graph-based RAG implementation using GPT-4o.
- `retrieval_augmented_generation_query_engine_llamaindex_rag_llm.py`: Query engine for RAG using LlamaIndex.

### Multi-Agent Systems

Located in the `multi-agent_coding_stock-analysis_customer-onboarding_chess_writing_conversation_autogen` directory:

- `Multi-Agent_Coding_and_Financial_Analysis_AutoGen.ipynb`: Demonstrates collaborative AI agents for coding and financial analysis.
- `Multi-Agent_Conversation_and_Stand-up_Comedy_AutoGen.ipynb`: AI agents generating conversational content and comedy.
- `Multi-Agent_Planning_and_Stock_Report_Generation_AutoGen.ipynb`: Agents that plan and generate stock reports.
- `Multi-Agent_Reflection_and_Blogpost_Writing_AutoGen.ipynb`: Collaborative writing and reflection through AI agents.
- `Multi-Agent_Sequential_Chats_and_Customer_Onboarding_AutoGen.ipynb`: Customer onboarding flows using multiple agents.
- `Multi-Agent_Tool_Use_and_Conversational_Chess_AutoGen.ipynb`: Tool use and chess game analysis by AI agents.
- `Multi-Agent_Tool_Use_Fake_Nvidia_Stocks.py`: Demonstration of tool use for stock analysis.

### Computer Vision

- `natural-scene-classification_cnns.ipynb`: Classification of natural scenes using CNNs.
- `image-classifier_resnet18.ipynb`: Image classification using ResNet18.
- `image-classifier_resnet34.py`: Image classification using ResNet34.
- `segmentation_resnet34.py`: Image segmentation using ResNet34.
- `noise_removal_autoencoder.py`: Noise removal using autoencoders.
- `image_question_answering_openai.py`: Visual question answering using OpenAI's GPT-4o.
- `text-to-image_clipdrop.py`: Text-to-image generation using ClipDrop.
- `resnet_transfer_learning_pneumonia.py`: Transfer learning with ResNet for pneumonia detection.


### Natural Language Processing

- `NLP_classification_search_text-edit_sentiment_analysis_Bert_BoW_SVM_embeddings_skip-gram_regex_POS.ipynb`: Comprehensive NLP techniques.
- `sentiment-analysis_vader_roberta.ipynb`: Sentiment analysis using VADER and RoBERTa.
- `sentiment_analysis_AWD-LSTM_enocder.py`: Sentiment analysis using AWD-LSTM encoder.
- `question_answering_openai.py`: Question answering using OpenAI models.
- `text-to-speech_elevenlabs.py`: Text-to-speech conversion using ElevenLabs.

### Machine Learning

- `clustering_kmeans.ipynb`: K-means clustering implementation.
- `dimensionality_reduction_pca.ipynb`: Principal Component Analysis (PCA) for dimensionality reduction.
- `tabular_prediction_nn.py`: Neural network for tabular data prediction.
- `digit-classifier_nn.py`: Neural network for digit classification.
- `recommender_system_movies_nn.py`: Movie recommendation system using neural networks.

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages depending on the specific project (see individual files)
- API keys for various services (OpenAI, ElevenLabs, etc.)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/practical-ai-projects.git
   cd practical-ai-projects
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies for the specific project you want to run.

### Usage

Each file or notebook contains example usage. For Python scripts, you can run them directly:

```bash
python retrieval_augmented_generation_agent_llm_llamaindex_gpt-4o.py
```

For Jupyter notebooks, open them in Jupyter Lab or Notebook:

```bash
jupyter lab
```

## API Key Setup

Many of these projects require API keys. Create a `.env` file in the root directory with the following content:

```
OPENAI_API_KEY=your_openai_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
# Add any other API keys required
```

## License

[MIT License]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- OpenAI for GPT models
- LlamaIndex for RAG implementations
- AutoGen for multi-agent systems
- Various other open-source libraries and frameworks used in the projects

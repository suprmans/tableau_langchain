import os

from langchain_openai import ChatOpenAI, AzureChatOpenAI, OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain.chat_models.base import BaseChatModel
from langchain.embeddings.base import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


def select_model(provider: str = "openai", model_name: str = "gpt-4o-mini", temperature: float = 0.2) -> BaseChatModel:
    """
    Selects a chat model based on the provider and environment variables.
    This function is modified to switch between a local OpenAI-compatible (LLM Local)
    server and the official on-cloud OpenAI service.

    Require to define: 
        ```
        os.environ['OPENAI_COMPATIBLE_BASE_URL'] = 'http://localhost:12434/engines/v1'  ## DO NOT define as OPENAI_API_BASE
        os.environ['OPENAI_COMPATIBLE_MODEL'] = 'hf.co/nectec/pathumma-llm-text-1.0.0:q4_k_m' ## LLM pre-trained model from HuggingFace
        ```

    """
    if provider == "azure":
        return AzureChatOpenAI(
            azure_deployment=os.environ.get("AZURE_OPENAI_AGENT_DEPLOYMENT_NAME"),
            openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=f"https://{os.environ.get('AZURE_OPENAI_API_INSTANCE_NAME')}.openai.azure.com",
            openai_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            model_name=model_name,
            temperature=temperature
        )
    
    elif provider == "google":
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=os.environ.get("GEMINI_API_KEY") # Assumes your key is in this env var
        )
    
    else:  # Default to OpenAI provider
        local_api_base = os.environ.get("OPENAI_COMPATIBLE_BASE_URL")
        if local_api_base:
            # --- LOCAL MODE ---
            # A local base URL is provided.
            local_model_name = os.environ.get("OPENAI_COMPATIBLE_MODEL")
            if not local_model_name:
                raise ValueError("OPENAI_COMPATIBLE_BASE_URL is set, but OPENAI_COMPATIBLE_MODEL is missing.")

            return ChatOpenAI(
                model_name=local_model_name,
                temperature=temperature,
                openai_api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
                base_url=local_api_base
            )
        else:  # default to OpenAI
            return ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                openai_api_key=os.environ.get("OPENAI_API_KEY")
            )

def select_embeddings(provider: str = "openai", model_name: str = "text-embedding-3-small") -> Embeddings:
    if provider == "azure":
        return AzureOpenAIEmbeddings(
            azure_deployment=os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
            openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=f"https://{os.environ.get('AZURE_OPENAI_API_INSTANCE_NAME')}.openai.azure.com",
            openai_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            model=model_name
        )
    
    elif provider == "google":
        # Note: Google's embedding model names are often different from chat models
        # e.g., "models/embedding-001"
        return GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=os.environ.get("GEMINI_API_KEY") # Assumes your key is in this env var
        )
    
    else:  # default to OpenAI
        return OpenAIEmbeddings(
            model=model_name,
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )

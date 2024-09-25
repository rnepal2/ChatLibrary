import os

class cfg:
    username = ''
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    API_VERSION = "2023-07-01-preview"
    API_TYPE = 'azure'
    DIMENSION = 1536
    VERBOSE = False

model_dict = {
                "GPT-3.5-16K": ["gpt-35-turbo-16k", "gpt-35-turbo-16k", 16384],
                "GPT-4-Turbo": ["gpt-4-turbo", "gpt-4", 128000],
                "GPT-4o": ["gpt-4o", "gpt-4o", 128000],
                "GPT-4o-Mini": ["gpt-4o-mini", "gpt-4-mini", 128000],
    }

## ChatLibrary

### Project Description
ChatLibrary is a generative AI-powered chat application designed to answer user questions based on a proprietary knowledge base. It utilizes advanced language models and vector databases to provide accurate and context-aware responses.

#### Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Tech Stack](#technologies-used)

#### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/rnepal2/ChatLibrary.git
   cd ChatLibrary
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   ```bash
   export AZURE_OPENAI_API_KEY='your_api_key'
   export AZURE_OPENAI_ENDPOINT='your_endpoint'
   ```
#### Usage
Run the application using:
   streamlit run app.py --server.port 3000

#### Tech Stack
- Streamlit: For building the web application interface.
- LangChain: For managing language model interactions.
- Azure OpenAI: For leveraging advanced language models.
- ChromaDB: For vector database management.

#### UI
<p align="center">
  <img src="https://github.com/rnepal2/ChatLibrary/blob/main/static/ui_screenshot.png" width="800" height="650">
</p>


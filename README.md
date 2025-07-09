# Assistant Summarizer - PDF Chat & YouTube/Web Summarizer

A comprehensive AI-powered application that enables users to summarize and interact with content from multiple sources including PDFs, YouTube videos, and web pages through an intuitive chat interface.

## Live Demo

**Streamlit App**: [https://goodbot.streamlit.app/](https://goodbot.streamlit.app/)

## Features

### PDF Processing
- Upload and process PDF documents of any size
- Extract text content and enable intelligent chat interactions
- Ask questions about specific sections or overall document content
- Context-aware responses based on document content

### YouTube Video Summarization
- Generate concise summaries of YouTube videos using video URLs
- Extract key points and main topics from video content
- Support for videos of varying lengths
- Transcription-based analysis for accurate content understanding

### Web Page Summarization
- Summarize content from any web URL
- Extract and process text from web articles and blogs
- Generate structured summaries with key insights
- Support for various web content formats

### Interactive Chat Interface
- Natural language conversation with your documents and summaries
- Context-aware responses based on processed content
- Follow-up questions and clarifications
- Persistent conversation history during session

### AI-Powered Analysis
- Advanced natural language processing capabilities
- Intelligent content extraction and summarization
- Context-aware question answering
- Support for multiple AI models and configurations

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **AI/ML**: Large Language Models (LLM) integration
- **PDF Processing**: PyPDF2/pdfplumber
- **Web Scraping**: BeautifulSoup4/requests
- **YouTube Processing**: youtube-transcript-api
- **Text Processing**: LangChain
- **Vector Database**: ChromaDB/FAISS (for document embeddings)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- OpenAI API key (or other LLM API credentials)

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/prince9115/Youtube-URL_and_PDF_with_chat.git
cd Youtube-URL_and_PDF_with_chat
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Create a .env file in the project root
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

5. Run the application:
```bash
streamlit run app.py
```

## Usage

### PDF Chat
1. Navigate to the PDF Chat section
2. Upload your PDF file using the file uploader
3. Wait for the document to be processed
4. Start asking questions about the document content
5. Receive context-aware responses based on the PDF content

### YouTube Video Summarization
1. Go to the YouTube Summarizer section
2. Paste the YouTube video URL
3. Click "Generate Summary"
4. Review the generated summary with key points
5. Ask follow-up questions about the video content

### Web Page Summarization
1. Access the Web Summarizer section
2. Enter the URL of the web page you want to summarize
3. Click "Summarize"
4. Get a structured summary of the web content
5. Engage in chat about the summarized content

## Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```
OPENAI_API_KEY=your_openai_api_key
MODEL_NAME=gpt-3.5-turbo
MAX_TOKENS=4000
TEMPERATURE=0.7
```

### Customization Options

- **Model Selection**: Choose between different AI models
- **Summary Length**: Adjust summary length preferences
- **Temperature Settings**: Control response creativity
- **Chunk Size**: Modify text processing chunk sizes

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## Limitations

- PDF processing may be limited by file size and complexity
- YouTube video processing depends on transcript availability
- Web scraping may be affected by website structure changes
- API rate limits may apply based on the chosen LLM provider

## Troubleshooting

### Common Issues

**PDF Upload Errors**
- Ensure PDF is not password-protected
- Check file size limits
- Verify PDF contains extractable text

**YouTube URL Issues**
- Ensure URL is valid and accessible
- Check if video has available transcripts
- Verify video is not private or restricted

**API Errors**
- Verify API key is correctly set
- Check API quota and rate limits
- Ensure stable internet connection

## Acknowledgments

- OpenAI for GPT models
- Streamlit community for the amazing framework
- LangChain for text processing capabilities
- Contributors and testers who helped improve the application

## Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/prince9115/Youtube-URL_and_PDF_with_chat/issues) section
2. Create a new issue with detailed information
3. Contact the maintainer through GitHub

## Changelog

### Version 1.0.0
- Initial release with PDF chat functionality
- YouTube video summarization feature
- Web page summarization capability
- Interactive chat interface
- Multi-model AI support

---

**Made with ❤️ by [prince9115](https://github.com/prince9115)**

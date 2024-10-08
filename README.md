# RepoNinja

**RepoNinja** is an advanced tool designed to enhance your Repository experience by providing seamless chat with the code. Whether you're a developer, project manager, or enthusiast, engage with repositories using the power of Ollama at ease in local without any extra cost.

## Features

Engage in a conversation with any repository using Ollama's language model, making it easier to understand code, documentation, and more.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following:

- A GitHub account.
- A valid [GitHub Token](https://github.com/settings/tokens) to access the GitHub API.
- Python 3.9+ installed on your local machine.
- [Ollama](https://ollama.com/) set up with the Llama 3.1 model.

### Installation

Follow these steps to set up RepoNinja on your local machine:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/AjinkyaTaranekar/RepoNinja/
   cd RepoNinja
   ```

2. **Set Up Environment Variables**

   You'll need to provide your GitHub token to access the API. Set the environment variable as follows:

   ```bash
   export GITHUB_TOKEN=<your_github_token>
   ```

3. **Install Dependencies**

   Install the required Python packages by running:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Ollama**

   Ensure you have the Llama 3.1 model downloaded and ready for use with Ollama:

   ```bash
   ollama pull llama3.1
   ```

### Running the Application

Once everything is set up, you can run RepoNinja using the following command:

```bash
python main.py
```

## Contributing

We welcome contributions to RepoNinja! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push them to your fork.
4. Open a pull request with a detailed description of your changes.

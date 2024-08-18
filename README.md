# RepoNinja

RepoNinja is a tool that allows users to search for repositories on GitHub and view their information. It uses Ollama to chat with the repository.
To run on local, be at project root and run these:

#### Set up environment variables
```
export GITHUB_TOKEN=<your github token>
```

#### Install dependencies
Setup [Ollama][https://ollama.com/] pull llama3.1 model before use.

```
pip install -r requirements.txt
```

#### Run
```
python main.py
```

# Semantic Search Over My Notes

Day 3 of learning how AI works: a semantic search over lecture notes. `build_index.py` pulls the text out of every PDF page and PowerPoint slide in a `notes` folder (dropping repeated build-up slides), embeds each one with OpenAI's `text-embedding-3-small` in a single batched call, and saves the text and vectors to `index.json`. `search.py` rebuilds that index, then embeds your question and ranks every page and slide by cosine similarity to show the closest matches, so you can find notes by meaning instead of exact words.

Run it with `python -m pip install pypdf python-pptx tiktoken openai numpy python-dotenv`, copy `.env.example` to `.env` and add your OpenAI API key, put your own `.pdf` and `.pptx` files in a `notes` folder (it isn't included in this repo), then run `python search.py`.

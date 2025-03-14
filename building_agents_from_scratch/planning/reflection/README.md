This is the second part of the "Building Agents from scratch" series where we will implement Reflection pattern together with a simple version of Working Memory from scratch without using any LLM orchestration frameworks.

<p align="center">
<img src="../../../assets/reflection.png" width="50%" alt="Reflection">
</p>

You can find the detailed walkthrough of this project in my [Newsletter](https://www.newsletter.swirlai.com/p/building-ai-agents-from-scratch-part-8ca).


## Installing dependencies

[uv](https://github.com/astral-sh/uv) is a great tool for Python dependency management. To install the requirements:

```bash
uv pip install -r requirements.txt
```

## Running the code

First, export your OpenAI API key:

```bash
export OPENAI_API_KEY=your_api_key
```

> Note: Support for other LLM providers (like Anthropic, Mistral, etc.) is coming soon! This will allow you to run the examples with your preferred model provider.

Then you can run the example:

```bash
python3 .src/main.py
```

## Interactive Notebook

For a more interactive learning experience, you can follow along with the Jupyter notebook in the [notebooks](notebooks) folder. While detailed documentation is still being worked on, you can find the complete implementation and follow the code there.
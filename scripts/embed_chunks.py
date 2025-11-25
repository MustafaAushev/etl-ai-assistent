from ollama import embeddings


def make_vector(text: str, dim: int = 1024) -> [float]:
  r = embeddings(model="qwen3-embedding:0.6b", prompt=text)
  return r.embedding[0:dim]

    
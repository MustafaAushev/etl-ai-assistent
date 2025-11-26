from ollama import embeddings

class EmbedderService:
  def make_vector(text: str, dim: int = 1024) -> [float]:
    r = embeddings(model="qwen3-embedding:0.6b", prompt=text)
    vector = r.embedding
    if len(vector) < dim:
      return vector + [0.0] * (dim - len(vector))
    return r.embedding[0:dim]

    
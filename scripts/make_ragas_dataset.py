from datetime import datetime
import json
import typing as t
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    ContextPrecision,
    ContextRecall,
    ContextRelevance,
    AnswerRelevancy,
)
from datasets import Dataset
from services.embedder_service import EmbedderService


class OllamaEmbeddings(BaseRagasEmbedding):
    def embed_text(self, text: str) -> t.List[float]:
      return EmbedderService.make_vector(text)
    
    async def aembed_text(self, text: str) -> t.List[float]:
      return EmbedderService.make_vector(text)

    def embed_query(self, text: str) -> t.List[float]:
      return EmbedderService.make_vector(text)

    async def aembed_query(self, text: str) -> t.List[float]:
      return EmbedderService.make_vector(text)

    def embed_documents(self, texts: t.List[str]) -> t.List[t.List[float]]:
      return [EmbedderService.make_vector(text) for text in texts]


def create_openrouter_llm(model_name: str, temperature: float = 0.7) -> ChatOpenAI:
    """
    Создает LLM через OpenRouter.
    
    Args:
        model_name: Название модели (например, 'anthropic/claude-sonnet-4.5')
        temperature: Температура для генерации
    
    Returns:
        Настроенный ChatOpenAI клиент
    """
    return ChatOpenAI(
        model=model_name,
        openai_api_key='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',
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": "RAGAS Test",
        },
        temperature=temperature,
    )


# LLM для всех метрик
evaluator = LangchainLLMWrapper(create_openrouter_llm('openai/gpt-5-nano'))
def dataset_from_json(filename: str) -> (Dataset, int):
  with open(filename, 'r') as f:
    content = json.load(f)
    return Dataset.from_dict({
      'question': [x['question'] for x in content],
      'answer': [x['answer'] for x in content],
      'ground_truth': [x['ground_truth'] for x in content],
      'contexts': [x['contexts'] for x in content],
    }), len(content)


dataset, dataset_length = dataset_from_json('ragas_dataset.json')

# Создаем embeddings один раз для переиспользования
embeddings = OllamaEmbeddings()

# Инициализируем все метрики явно
faithfulness = Faithfulness(
  llm=evaluator,
)

answer_relevancy = AnswerRelevancy(
  strictness=1,
  llm=evaluator,
  embeddings=embeddings,
)

context_precision = ContextPrecision(
  llm=evaluator,
)

context_recall = ContextRecall(
  llm=evaluator,
)

context_relevance = ContextRelevance(
  llm=evaluator,
)

score = evaluate(
  dataset=dataset, 
  metrics=[
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_relevance,
  ],
  allow_nest_asyncio=False
)

print(score)
with open(f"out_ragas/{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}.score.json", 'w+') as out:
  out.write(json.dumps({
    'dataset_length': dataset_length,
    'score': score._repr_dict,
  }, indent=2))

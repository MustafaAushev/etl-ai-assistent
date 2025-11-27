import os
from typing import List
from openai import OpenAI
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings.base import BaseRagasEmbedding
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from services.embedder_service import EmbedderService
from services.qdrant_service import QdrantService


def create_openrouter_llm(model_name: str, temperature: float = 0.7) -> ChatOpenAI:
    """
    Создает LLM через OpenRouter.

    Args:
        model_name: Название модели (например, 'openai/gpt-4o-mini')
        temperature: Температура для генерации

    Returns:
        Настроенный ChatOpenAI клиент
    """
    api_key = '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'
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY не установлен. Установите переменную окружения."
        )

    return ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": "RAGAS Test Generator",
        },
        temperature=temperature,
    )


class OllamaEmbeddings(BaseRagasEmbedding):
    def embed_text(self, text: str) -> List[float]:
      return EmbedderService.make_vector(text)
    
    async def aembed_text(self, text: str) -> List[float]:
      return EmbedderService.make_vector(text)

def load_documents_from_qdrant(
    qdrant_service: QdrantService, limit: int = 1000
) -> List[Document]:
    """
    Загружает документы из Qdrant и преобразует их в формат LangChain Document.

    Args:
        qdrant_service: Сервис для работы с Qdrant
        limit: Максимальное количество документов для загрузки

    Returns:
        Список документов LangChain
    """
    documents = []
    try:
        scroll_result = qdrant_service.client.scroll(
            collection_name=qdrant_service.collection, limit=limit, with_payload=True
        )

        points = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result
        for point in points:
            payload = point.payload
            text = payload.get("text", "")
            if not text:
                continue
            metadata = {
                "paragraph_name": payload.get("paragraph_name", ""),
                "document_name": payload.get("document_name", ""),
                "document_version": payload.get("document_version", ""),
            }
            documents.append(Document(page_content=text, metadata=metadata))
    except Exception as e:
        print(f"Ошибка при загрузке документов из Qdrant: {e}")
        raise

    return documents


def generate_testset(
    documents: List[Document],
    test_size: int = 50,
    generator_model: str = "openai/gpt-4o-mini",
    critic_model: str = "openai/gpt-4o-mini",
) -> pd.DataFrame:
    """
    Генерирует тестовый набор с помощью RAGAS.

    Args:
        documents: Список документов для генерации тестов
        test_size: Количество тестов для генерации
        generator_model: Модель для генерации вопросов
        critic_model: Модель для критики и фильтрации

    Returns:
        DataFrame с тестовым набором
    """
    print(f"Инициализация LLM: generator={generator_model}, critic={critic_model}")
    generator_llm = LangchainLLMWrapper(create_openrouter_llm(critic_model)) 
    critic_llm = LangchainLLMWrapper(create_openrouter_llm(critic_model))
    embeddings = OllamaEmbeddings()

    print("Создание генератора тестов...")
    generator = TestsetGenerator.from_langchain(
        llm=generator_llm,
        embedding_model=embeddings,
    )

    print(f"Генерация {test_size} тестов из {len(documents)} документов...")
    testset = generator.generate_with_langchain_docs(
        documents,
        testset_size=test_size,
        raise_exceptions=True,
        with_debugging_logs=True,
    )

    return testset


def run_ragas_evaluation(
    testset: pd.DataFrame,
    answer_column: str = "answer",
    contexts_column: str = "contexts",
    question_column: str = "question",
    ground_truth_column: str = "ground_truth",
):
    """
    Запускает оценку RAGAS на тестовом наборе.

    Args:
        testset: DataFrame с тестовым набором
        answer_column: Название колонки с ответами
        contexts_column: Название колонки с контекстами
        question_column: Название колонки с вопросами
        ground_truth_column: Название колонки с правильными ответами

    Returns:
        Результаты оценки RAGAS
    """
    dataset_dict = {
        "question": testset[question_column].tolist(),
        "answer": testset[answer_column].tolist(),
        "contexts": testset[contexts_column].tolist(),
        "ground_truth": testset[ground_truth_column].tolist(),
    }

    result = evaluate(
        dataset=dataset_dict,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    return result


def main():
    """Основная функция для генерации и запуска тестов RAGAS."""
    print("Загрузка документов из Qdrant...")
    qdrant_service = QdrantService()
    documents = load_documents_from_qdrant(qdrant_service, limit=1000)
    print(f"Загружено {len(documents)} документов")

    if not documents:
        print("Ошибка: не найдено документов в Qdrant")
        return

    print("Генерация тестового набора...")
    try:
        generator_model = os.getenv("RAGAS_GENERATOR_MODEL", "openai/gpt-4o-mini")
        critic_model = os.getenv("RAGAS_CRITIC_MODEL", "openai/gpt-4o-mini")
        test_size = int(os.getenv("RAGAS_TEST_SIZE", "50"))

        testset = generate_testset(
            documents,
            test_size=test_size,
            generator_model=generator_model,
            critic_model=critic_model,
        )

        print(f"Сгенерировано {len(testset)} тестов")
        print("Сохранение тестового набора...")
        testset.to_csv("ragas_testset.csv", index=False)
        print("Тестовый набор сохранен в ragas_testset.csv")
    except Exception as e:
        print(f"Ошибка при генерации тестового набора: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Запуск оценки RAGAS...")
    try:
        results = run_ragas_evaluation(testset)
        print("Результаты оценки:")
        print(results)

        if hasattr(results, "to_pandas"):
            results_df = results.to_pandas()
            print("Сохранение результатов оценки...")
            results_df.to_csv("ragas_evaluation_results.csv", index=False)
            print("Результаты сохранены в ragas_evaluation_results.csv")
        elif isinstance(results, pd.DataFrame):
            print("Сохранение результатов оценки...")
            results.to_csv("ragas_evaluation_results.csv", index=False)
            print("Результаты сохранены в ragas_evaluation_results.csv")
        else:
            print(f"Результаты оценки (тип: {type(results)}):")
            print(results)
    except Exception as e:
        print(f"Ошибка при запуске оценки: {e}")
        print("Продолжаем без оценки...")


if __name__ == "__main__":
    main()

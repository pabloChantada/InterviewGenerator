import os

import numpy as np
from datasets import load_dataset

# Raiz del repositorio: src/project/rag/utils/dataset_readers.py -> utils -> rag -> project -> src -> raiz
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
COACHQUANT_PATH = os.path.join(BASE_DIR, "data", "coachquant_all.jsonl")

'''
NOTA: Aqui solo usamos el SQUAD y Coachquant en la version final. Pero dejamos los otros readers
por si en el futuro se quieren usar otros datasets.
'''

def _process_qa_texts(qa_texts, max_texts, sample_random, verbose=1):
    """Helper para procesar y limitar qa_texts."""
    total = len(qa_texts)
    if verbose:
        print(f"[RAG] Pares QA construidos totales: {total}")

    if max_texts is not None and total > max_texts:
        if sample_random:
            if verbose:
                print(f"[RAG] Muestreando aleatoriamente {max_texts} pares")
            indices = np.random.choice(total, size=max_texts, replace=False)
            qa_texts = [qa_texts[i] for i in indices]
        else:
            if verbose:
                print(f"[RAG] Limitando a los primeros {max_texts} pares")
            qa_texts = qa_texts[:max_texts]

    if verbose:
        print(f"[RAG] Pares QA devueltos: {len(qa_texts)}")
    return qa_texts


def reader_SQUAD(
    max_texts: int | None = None, sample_random: bool = False, verbose: bool = True
):
    """
    Leer SQuAD dataset desde Hugging Face.

    Args:
        max_texts: si se pasa un entero, limita la salida a los primeros N pares.
        sample_random: si True y max_texts definido, selecciona una muestra aleatoria.
    Returns:
        Lista de strings con "Contexto: ... \nPregunta: ... \nRespuesta: ..."
    """
    qa_texts = []

    if verbose:
        print("[RAG] Cargando SQuAD desde Hugging Face...")

    try:
        # Cargar dataset SQuAD v1.1
        dataset = load_dataset("squad", split="train", trust_remote_code=True)

        if verbose:
            print(f"[RAG] Dataset cargado: {len(dataset)} ejemplos")

        for example in dataset:
            context = example.get("context", "")
            question = example.get("question", "")
            answers = example.get("answers", {})

            # Extraer primera respuesta
            answer_texts = answers.get("text", [])
            answer = answer_texts[0] if answer_texts else ""

            if context and question and answer:
                text_block = (
                    f"Contexto: {context}\nPregunta: {question}\nRespuesta: {answer}"
                )
                qa_texts.append(text_block)

    except Exception as e:
        if verbose:
            print(f"[RAG] Error cargando SQuAD: {e}")
        return []

    return _process_qa_texts(qa_texts, max_texts, sample_random, verbose)


def reader_natural_questions(
    max_texts: int | None = None, sample_random: bool = False, verbose: bool = True
):
    """
    Leer Natural Questions dataset desde Hugging Face.

    Usa el dataset 'google-research-datasets/natural_questions'
    """
    qa_texts = []

    if verbose:
        print("[RAG] Cargando Natural Questions desde Hugging Face...")

    try:
        # Cargar solo una porción del dataset (es muy grande)
        # Usamos el validation split que es más pequeño
        dataset = load_dataset(
            "google-research-datasets/natural_questions",
            split="validation[:10000]",  # Primeros 10k ejemplos
            trust_remote_code=True,
        )

        if verbose:
            print(f"[RAG] Dataset cargado: {len(dataset)} ejemplos")

        for example in dataset:
            question = example.get("question", {}).get("text", "")

            # Extraer contexto del documento
            document = example.get("document", {})
            tokens = document.get("tokens", {}).get("token", [])

            # Unir los primeros 200 tokens como contexto
            context = " ".join(tokens[:200]) if tokens else ""

            # Extraer respuesta corta
            annotations = example.get("annotations", [])
            answer = ""

            if annotations and len(annotations) > 0:
                short_answers = annotations[0].get("short_answers", [])
                if short_answers:
                    # Construir respuesta desde tokens
                    start = short_answers[0].get("start_token", 0)
                    end = short_answers[0].get("end_token", 0)
                    if start < len(tokens) and end <= len(tokens):
                        answer = " ".join(tokens[start:end])
                elif "yes_no_answer" in annotations[0]:
                    answer = annotations[0]["yes_no_answer"]

            if question and context and answer:
                text_block = (
                    f"Contexto: {context}\nPregunta: {question}\nRespuesta: {answer}"
                )
                qa_texts.append(text_block)

    except Exception as e:
        if verbose:
            print(f"[RAG] Error cargando Natural Questions: {e}")
        return []

    return _process_qa_texts(qa_texts, max_texts, sample_random, verbose)


def reader_eli5(
    max_texts: int | None = None, sample_random: bool = False, verbose: bool = True
):
    """
    Leer ELI5 dataset desde Hugging Face.

    Usa el dataset 'eli5'
    """
    qa_texts = []

    if verbose:
        print("[RAG] Cargando ELI5 desde Hugging Face...")

    try:
        # Cargar train split del dataset ELI5
        dataset = load_dataset(
            "eli5_category",
            split="train[:5000]",  # Primeros 5k ejemplos
            trust_remote_code=True,
        )

        if verbose:
            print(f"[RAG] Dataset cargado: {len(dataset)} ejemplos")

        for example in dataset:
            title = example.get("title", "")
            selftext = example.get("selftext", "")
            question = f"{title} {selftext}".strip()

            # Extraer respuesta
            answers = example.get("answers", {})
            answer_text = ""

            if isinstance(answers, dict):
                answer_texts = answers.get("text", [])
                if answer_texts and len(answer_texts) > 0:
                    # Tomar la primera respuesta y limitarla
                    answer_text = answer_texts[0][:500]

            if question and answer_text:
                text_block = f"Pregunta: {question}\nRespuesta: {answer_text}"
                qa_texts.append(text_block)

    except Exception as e:
        if verbose:
            print(f"[RAG] Error cargando ELI5: {e}")
        return []

    return _process_qa_texts(qa_texts, max_texts, sample_random, verbose)


def reader_hotpotqa(
    max_texts: int | None = None, sample_random: bool = False, verbose: bool = True
):
    """
    Leer HotpotQA dataset desde Hugging Face.

    Usa el dataset 'hotpot_qa'
    """
    qa_texts = []

    if verbose:
        print("[RAG] Cargando HotpotQA desde Hugging Face...")

    try:
        # Cargar distractor split (contiene preguntas multi-hop)
        dataset = load_dataset(
            "hotpot_qa",
            "distractor",
            split="train[:5000]",  # Primeros 5k ejemplos
            trust_remote_code=True,
        )

        if verbose:
            print(f"[RAG] Dataset cargado: {len(dataset)} ejemplos")

        for example in dataset:
            question = example.get("question", "")
            answer = example.get("answer", "")
            context_data = example.get("context", {})

            # Construir contexto desde los documentos
            context_parts = []

            # context puede ser una lista de pares [título, sentencias]
            if "title" in context_data and "sentences" in context_data:
                titles = context_data.get("title", [])
                sentences = context_data.get("sentences", [])

                for title, sents in zip(titles, sentences):
                    if isinstance(sents, list):
                        text = " ".join(sents)
                        context_parts.append(f"{title}: {text}")

            context = " ".join(context_parts)[:1500]

            if question and answer and context:
                text_block = (
                    f"Contexto: {context}\nPregunta: {question}\nRespuesta: {answer}"
                )
                qa_texts.append(text_block)

    except Exception as e:
        if verbose:
            print(f"[RAG] Error cargando HotpotQA: {e}")
        return []

    return _process_qa_texts(qa_texts, max_texts, sample_random, verbose)

def reader_coachquant(
    data_path: str = COACHQUANT_PATH,
    max_texts: int | None = None,
    sample_random: bool = False,
    verbose: bool = True,
):
    """Lee tu dataset scrapeado (JSONL con un objeto por línea)."""
    import json, os
    import numpy as np

    if verbose:
        print("[RAG] Cargando dataset CoachQuant...")

    if not os.path.exists(data_path):
        if verbose:
            print(f"[RAG] No existe el archivo: {data_path}")
        return []

    qa_texts = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                raw = obj.get("raw", {})
                question = raw.get("problem text", "") or obj.get("question_text", "")
                answer = (
                    raw.get("problem solution", "")
                    or obj.get("answer_text", "")
                    or raw.get("valid answer", "")
                )
                if question and answer:
                    qa_texts.append(f"Pregunta: {question}\nRespuesta: {answer}")
            except Exception as e:
                if verbose:
                    print(f"[RAG] Error leyendo línea: {e}")

    total = len(qa_texts)
    if verbose:
        print(f"[RAG] Total pares QA leídos: {total}")

    if max_texts and total > max_texts:
        if sample_random:
            idx = np.random.choice(total, size=max_texts, replace=False)
            qa_texts = [qa_texts[i] for i in idx]
        else:
            qa_texts = qa_texts[:max_texts]
    if verbose:
        print(f"[RAG] Pares QA devueltos: {len(qa_texts)}")
    return qa_texts

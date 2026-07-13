import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch

# Unicamente usamos SQUAD y Coachquant, pero importamos todos para soporte multi-dataset
from .utils.dataset_readers import (
    reader_natural_questions,
    reader_SQUAD,
    reader_eli5,
    reader_hotpotqa,
    reader_coachquant
)

load_dotenv()

# Raiz del repositorio: src/project/rag/rag.py -> rag -> project -> src -> raiz
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
DB_DIR = os.path.join(BASE_DIR, "data")


class RAG:
    def __init__(
        self,
        db_path: str = DB_DIR,
        model_embedder: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        verbose: bool = False,
        dataset_type: str = "squad",  # Nuevo parámetro
    ):
        """
        RAG helper con soporte multi-dataset.

        Args:
            db_path: ruta base donde se guardará chroma_db.
            model_embedder: modelo de embeddings.
            verbose: si True imprime información extra durante las operaciones.
            dataset_type: tipo de dataset ('squad', 'natural_questions', 'eli5', 'hotpotqa')
        """
        self.db_path = db_path
        self.model_embedder = model_embedder
        self.db = None
        self.verbose = verbose
        self.dataset_type = dataset_type.lower()

        if torch.cuda.is_available():
            self.device = "cuda"
            self.batch_size = 64
            if self.verbose:
                print(f"[RAG] GPU detectada: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.batch_size = 32
            if self.verbose:
                print("[RAG] GPU detectada usando MPS")
        else:
            self.device = "cpu"
            self.batch_size = 8
            if self.verbose:
                print("[RAG] Usando CPU")

    def _get_embeddings(self):
        """Crear instancia de embeddings (usa self.batch_size y self.device)."""
        if self.verbose:
            print(
                f"[RAG] Creando embeddings con modelo {self.model_embedder} en {self.device} (batch_size={self.batch_size})"
            )
        return HuggingFaceEmbeddings(
            model_name=self.model_embedder,
            model_kwargs={
                "device": self.device,
                "trust_remote_code": True,
            },
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": self.batch_size,
            },
        )

    def read_dataset(self, max_texts: int | None = None, sample_random: bool = False):
        """
        Método unificado para leer cualquier dataset según self.dataset_type.
        """

        # Solo se usan SQUAD y CoachQuant aquí, pero se importan todos para soporte multi-dataset
        if self.dataset_type == "squad":
            return reader_SQUAD(max_texts, sample_random)
        elif self.dataset_type == "natural_questions":
            return reader_natural_questions(max_texts, sample_random)
        elif self.dataset_type == "eli5":
            return reader_eli5(max_texts, sample_random)
        elif self.dataset_type == "hotpotqa":
            return reader_hotpotqa(max_texts, sample_random)
        elif self.dataset_type == "coachquant":
            return reader_coachquant(max_texts=max_texts, sample_random=sample_random)
        else:
            raise ValueError(f"Tipo de dataset no soportado: {self.dataset_type}")

    def chunker(self, qa_texts, chunk_size: int = 1024, max_chunks: int | None = None):
        """
        Dividir pares QA en chunks más pequeños.

        Args:
            qa_texts: lista de textos a dividir.
            chunk_size: tamaño máximo de cada chunk.
            max_chunks: si se pasa, limita el número total de chunks devueltos.
        Returns:
            Lista de chunks (strings).
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=128,
            length_function=len,
            add_start_index=True,
        )

        all_chunks = []
        for i, text in enumerate(qa_texts, start=1):
            chunks = splitter.split_text(text)
            all_chunks.extend(chunks)
            if self.verbose and i % 50 == 0:
                print(
                    f"[RAG] Chunking: procesados {i}/{len(qa_texts)} textos -> {len(all_chunks)} chunks"
                )

            # Si ya alcanzamos el límite de chunks, interrumpimos
            if max_chunks is not None and len(all_chunks) >= max_chunks:
                if self.verbose:
                    print(
                        f"[RAG] Alcanzado max_chunks={max_chunks}, deteniendo chunker"
                    )
                all_chunks = all_chunks[:max_chunks]
                break

        if self.verbose:
            print(f"[RAG] Total de chunks creados: {len(all_chunks)}")
        return all_chunks

    def create_chroma_db(
        self,
        sample_size: int | None = None,
        sample_random: bool = False,
        chunk_size: int = 1024,
        max_chunks: int | None = None,
        verbose: bool | None = None,
    ):
        """
        Crea y guarda una base vectorial con Chroma.

        Args:
            sample_size: si se pasa un entero, solo se usan hasta sample_size pares QA (útil para testing).
            sample_random: si True y sample_size definido, la selección será aleatoria.
            chunk_size: tamaño de chunk para splitter.
            max_chunks: si se pasa, limita el número total de chunks a indexar (útil para testing).
            verbose: override del verbosity por llamada.
        """
        if verbose is None:
            verbose = self.verbose

        if verbose:
            print(
                f"[RAG] Iniciando creación de la DB para dataset '{self.dataset_type}' (sample_size={sample_size}, sample_random={sample_random}, chunk_size={chunk_size}, max_chunks={max_chunks})"
            )

        qa_texts = self.read_dataset(max_texts=sample_size, sample_random=sample_random)
        if not qa_texts:
            raise ValueError("No text to generate the embedding database")

        chunks = self.chunker(qa_texts, chunk_size=chunk_size, max_chunks=max_chunks)
        if not chunks:
            raise ValueError("No chunks created from the texts")

        embeddings = self._get_embeddings()

        # Crear directorio específico por dataset
        db_dir = os.path.join(self.db_path, f"chroma_db_{self.dataset_type}")
        if verbose:
            print(f"[RAG] Creando Database con {len(chunks)} chunks en {db_dir}")

        self.db = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            persist_directory=db_dir,
        )

        if verbose:
            print("[RAG] Base de datos creada y persistida")

    def load_chroma_db(self, verbose: bool | None = None):
        """Cargar base de datos existente"""
        if verbose is None:
            verbose = self.verbose
        embeddings = self._get_embeddings()
        db_dir = os.path.join(self.db_path, f"chroma_db_{self.dataset_type}")
        if verbose:
            print(f"[RAG] Cargando DB desde {db_dir}")
        self.db = Chroma(
            persist_directory=db_dir,
            embedding_function=embeddings,
        )
        if verbose:
            print("[RAG] DB cargada")

    def search(self, query: str, k: int = 5, verbose: bool | None = None):
        """Buscar contextos similares a una query"""
        if verbose is None:
            verbose = self.verbose
        if not self.db:
            if verbose:
                print("[RAG] DB no cargada, intentando load_chroma_db()")
            self.load_chroma_db(verbose=verbose)
        results = self.db.similarity_search(query, k=k)
        if verbose:
            print(f"[RAG] Se encontraron {len(results)} contextos similares.")
        return results


if __name__ == "__main__":
    # Ejemplo de uso con diferentes datasets
    print("=== Testing SQuAD ===")
    rag_squad = RAG(dataset_type="squad", verbose=True)
    rag_squad.create_chroma_db(sample_random=True, chunk_size=512, max_chunks=50)

    print("\n=== Testing Natural Questions ===")
    rag_nq = RAG(dataset_type="natural_questions", verbose=True)
    rag_nq.create_chroma_db(sample_random=True, chunk_size=512, max_chunks=50)

    print("\n=== Testing ELI5 ===")
    rag_eli5 = RAG(dataset_type="eli5", verbose=True)
    rag_eli5.create_chroma_db(sample_random=True, chunk_size=512, max_chunks=50)

    print("\n=== Testing HotPotQA ===")
    rag_hotpotqa = RAG(dataset_type="hotpotqa", verbose=True)
    rag_hotpotqa.create_chroma_db(sample_random=True, chunk_size=512, max_chunks=50)

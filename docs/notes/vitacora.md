## 8/10
- [x] Selección del tema  
- [x] Análisis de recursos y objetivos  
- [x] Planificación de archivos y divisiones  

---

## 15/10
### Dataset → *Gullen*
- [ ] Obtención de datos  
- [ ] Creación de dataset (actual: se usa SQuAD como base)  
- [x] Carga de datos  
- [x] Limpieza del dataset (de SQuAD al menos)  
- [x] Selección de modelos (probablemente Gemini)  

---

## 20/10
- [x] Prueba inicial con dataset simple (SQuAD como base)  
- [x] Pipeline básico de modelo cargado y probando inferencia  

---

## 22/10 – Implementación RAG
### RAG básico
- [x] Definición del *pipeline RAG*: retriever (FAISS/Chroma), reader (Gemini u otro), y prompt manager  
- [x] Evaluación *baseline* (sin RAG) para comparar  
- [x] Diseño de prompts específicos para entrevistas  

### RAG avanzado
- [ ] Investigación de RAGs más complejos (ej. ColBERT, HyDE, Context compression, etc.)  
- [x] Implementación de RAG mejorado (retrieval + context ranking + respuesta)  
- [x] Ajuste de temperatura y longitud de respuesta  

---

## Análisis de respuestas
- [x] Definir métricas automáticas de calidad (BLEU, ROUGE, BERTScore)  
- [ ] Sistema de *feedback automatizado* al usuario  
  > “Tu respuesta cubre bien el concepto, pero podrías desarrollar más el ejemplo.”  
- [ ] (Opcional) *Evaluation LLM* o clasificador ligero para puntuar respuestas  
- [ ] Respuesta del usuario vs respuesta del llm 
---

## A FUTURO
### Agentificación y escalabilidad
- [x] Agentes por dominio (matemáticas, historia, etc.) con datasets o contextos propios  
- [x] Memoria de usuario (nivel, historial de respuestas, feedback)  
- [ ] Control dinámico de dificultad de preguntas  
- [ ] Fine-tuning ligero (LoRA / QLoRA) sobre dataset propio de entrevistas  

### Gestión y análisis
- [ ] Registro de métricas (tiempo de respuesta, aciertos estimados, dificultad media)  
- [ ] Evaluación continua del modelo (monitorización / MLOps ligero)  

### Interfaz y experiencia de usuario
- [x] Creación de interfaz conversacional (Streamlit / Gradio / React + FastAPI)  
- [x] Visualización de estadísticas del usuario y feedback  
- [ ] (Opcional) Soporte multilingüe (ES / EN)  

### Extensiones 
- [x] Integración con vector DB + embeddings propios  
- [ ] Dataset especializado en entrevistas reales  
- [x] Sistema de *auto-evaluación final* con puntaje (claridad, estructura, tono)  


# MODELOS PROBADOS
- DialGPT
- Qwen
- Bert multilingual
- Llama
- Gemini
- ChatGPT

- Metricas ponderadas para un mejor analisis
  - Adicion de Resumen con LLM
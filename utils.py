import os
import time
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import warnings
import json
import unicodedata

# Ocultar logs de TensorFlow/JAX
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings('ignore')

# Inicializar Pinecone y crear un indice
def create_index_pinecone(index_name, dimension):
    api_key = os.getenv("PINECONE_API_KEY")
    cloud = os.getenv("PINECONE_CLOUD")
    region = os.getenv("PINECONE_REGION")

    pc = Pinecone(api_key=api_key)

    # Crear índice si no existe
    if index_name not in [index.name for index in pc.list_indexes()]:
        print(f"El indice '{index_name}' no existe. Creando...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            spec=ServerlessSpec(cloud=cloud, region=region),
            metric="cosine"
        )
    else:
        print(f"El indice '{index_name}' ya existe")

    return pc.Index(index_name)

# Eliminar todos los indices existentes en Pinecone
def delete_all_pinecone_indexes():
    api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)

    indexes = pc.list_indexes()
    if not indexes:
        print("No hay indices para eliminar")
        return

    for index in indexes:
        print(f"Eliminando indice: {index.name}")
        pc.delete_index(index.name)
    time.sleep(1)

# Eliminar tildes, convertir a minúsculas y reemplazar espacios por guiones
def normalize_name(name):
    nfkd = unicodedata.normalize('NFKD', name)
    no_accent = ''.join([c for c in nfkd if not unicodedata.combining(c)])
    return no_accent.lower().replace(" ", "-")

# Cargar un archivo CSV con curriculums, generar embeddings para cada uno,
# subirlos a Pinecone y generar un archivo de mapeo entre nombres e indices
def process_resumes_individually(csv_path, model, dimension=384, mapping_path="data/mapping.json"):
    df = pd.read_csv(csv_path)
    tokenizer = model.tokenizer
    model_max_length = tokenizer.model_max_length
    stats = {"total_tokens": 0, "num_chunks": 0, "num_resumes": 0, "truncated": 0}

    name_to_index = {}

    for _, row in df.iterrows():
        resume_text = str(row["Resume"])
        full_name = str(row["Name"]).strip()
        index_name = f"cv-{full_name.replace(' ', '-').lower()}"
        name_to_index[full_name] = index_name

        index = create_index_pinecone(index_name, dimension)

        original_tokens = tokenizer.encode(resume_text, truncation=False)
        num_original_tokens = len(original_tokens)
        stats["total_tokens"] += num_original_tokens
        stats["num_resumes"] += 1

        if num_original_tokens > model_max_length:
            print(f"CV {full_name} excede el límite de tokens ({model_max_length})")
            stats["truncated"] += 1

        chunks = split_text_into_chunks(resume_text, tokenizer, max_tokens=model_max_length, overlap=20)
        chunk_data = {}
        chunk_texts = {}

        for i, chunk in enumerate(chunks):
            chunk_id = f"{index_name}_chunk{i}"
            embedding = generate_embeddings(chunk, model)
            chunk_data[chunk_id] = embedding
            chunk_texts[chunk_id] = chunk
            stats["num_chunks"] += 1

        upload_vectors_to_pinecone(index, chunk_data, chunk_texts)
        print(f"Subido al indice: {index_name} ({len(chunk_data)} chunks)")

    # Guardar el mapping
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(name_to_index, f, ensure_ascii=False, indent=2)

    print("-" * 20)
    print(f"Procesados: {stats['num_resumes']} CVs")
    print(f"Total de chunks generados: {stats['num_chunks']}")
    print(f"CVs truncados: {stats['truncated']} ({(stats['truncated']/stats['num_resumes'])*100:.1f}%)")
    print("-" * 20)

# Division en chunks
def split_text_into_chunks(text, tokenizer, max_tokens=200, overlap=20):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return [tokenizer.decode(tokens)]

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += max_tokens - overlap
    return chunks

# Cargar el modelo de embeddings (tuve problemas con jinaai/jina-embeddings-v2-small-en y Python 3.12)
def load_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    return model

# Generar embeddings
def generate_embeddings(text, model):
    return model.encode([text])[0]

# Subir vectores a Pinecone
def upload_vectors_to_pinecone(index, data, original_texts):
    for key, value in data.items():
        metadata = {"text": original_texts[key]}
        index.upsert([(key, value.tolist(), metadata)])
    time.sleep(1)

# Buscar el indice para una persona a partir de su nombre en Pinecone
# y enviar el contexto al modelo LLM para generar una respuesta
def buscar_cv(nombre, mapping_path="data/mapping.json", top_k=2):
    print(f"[buscar_cv] Buscando CV para '{nombre}'")

    # Cargar el mapping
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    nombre_normalizado = normalize_name(nombre)

    # Buscar el indice que corresponde
    index_name = None
    for real_name, index_name_candidate in mapping.items():
        if normalize_name(real_name) == nombre_normalizado:
            index_name = index_name_candidate
            break

    if not index_name:
        return f"[buscar_cv] No se encontro un indice para '{nombre}'."

    # Cargar el indice desde Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(index_name)

    # Obtener todos los vectores del indice actual
    model = load_embedding_model()
    embedding = generate_embeddings(nombre, model)
    embedding_list = embedding.tolist()
    response = index.query(
        vector=embedding_list,
        top_k=top_k,
        include_metadata=True
    )
    resultados = response["matches"]
    cv = "\n".join([match["metadata"]["text"] for match in resultados])
    return cv

# Buscar los currículums de múltiples personas a partir de una lista de nombres
def buscar_multi_cv(nombres_str):
    print(f"[buscar_multi_cv] Buscando CVs para '{nombres_str}'")
    nombres = [n.strip() for n in nombres_str.split(",")]
    respuestas = []
    for nombre in nombres:
        respuestas.append(buscar_cv(nombre))
    return "\n".join(respuestas)

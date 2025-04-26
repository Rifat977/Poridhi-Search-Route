



import pandas as pd
import numpy as np
import time
import re

from fastapi import FastAPI, Query
from langchain.schema import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams
import google.generativeai as genai

from sentence_transformers import SentenceTransformer


from typing import List, Dict, Literal
from pydantic import BaseModel

# Load CSV and prepare documents
def load_and_prepare_documents(csv_path: str):
    df = pd.read_csv(csv_path)
    docs = [
        Document(
            page_content=(
                str(row["category"]) + " "
                + str(row["brand"]) + " "
                + str(row["title"]) + " "
                + str(row["description"]) + " "
                + str(row["specTableContent"])
            ),
            metadata={
                "id": str(row.get("id", "")),
                "title": str(row.get("title", "")),
                "brand": str(row.get("brand", "")),
                "description": str(row.get("description", "")),
                "price": str(row.get("price", "")),
            }
        )
        for _, row in df.iterrows()
    ]
    return docs

def initialize_vector_store(
    docs,
    collection_name="intent_based_product_v3",
    batch_size=64,
    local_path="/home/rifat/qdrant_data"
):
    print("⚙️ Initializing Qdrant vector store...")

    dense = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
    sparse = FastEmbedSparse(model_name="Qdrant/bm25")
    
    client = QdrantClient(url="http://localhost:6333")

    collection_exists = False

    # Check if collection exists
    try:
        client.get_collection(collection_name)
        collection_exists = True
        print(f"✅ Collection '{collection_name}' already exists. Skipping creation and insertion.")
    except Exception:
        print(f"❗ Collection '{collection_name}' not found. Creating a new one...")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(size=384, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
            },
        )
        print("➕ Collection created successfully.")

    # Initialize the vector store (always needed)
    qdrant_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=dense,
        sparse_embedding=sparse,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )

    # Only insert documents if the collection was just created
    if not collection_exists and docs:
        ids = list(range(1, len(docs) + 1))
        print(f"📥 Adding {len(docs)} documents to Qdrant in batches of {batch_size}...")

        for start_idx in range(0, len(docs), batch_size):
            end_idx = min(start_idx + batch_size, len(docs))
            batch_docs = docs[start_idx:end_idx]
            batch_ids = ids[start_idx:end_idx]

            try:
                qdrant_store.add_documents(documents=batch_docs, ids=batch_ids)
                print(f"✅ Inserted documents {start_idx+1} to {end_idx}")
            except Exception as e:
                print(f"❌ Failed to insert batch {start_idx+1} to {end_idx}: {e}")
    else:
        if collection_exists:
            print("ℹ️ Skipping document insertion as collection already exists.")
        elif not docs:
            print("⚠️ No documents provided to insert.")

    print("✅ Vector store fully initialized and ready.")
    return qdrant_store


# Configure Gemini
def setup_gemini(api_key: str):
    print("🔐 Configuring Gemini API...")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    print("✅ Gemini model ready.")
    return model

# Query optimization
def extracts_intent_gemini(raw_query: str) -> str:
    prompt_template = """
    You are an assistant that prepares user search queries for an ecommerce search API. Instructions:
    1. If the query is in Bangla (Bengali script) or Banglish (Bangla written in English letters), translate it into clear, concise English.
    2. If the query is vague or incomplete, complete it briefly and to the point, without making it long or adding unnecessary words.
    3. If the query is already in clear and complete English, leave it mostly as-is, only fixing vagueness or incompleteness if necessary.
    4. Keep the output short, precise, and suitable for a search API.
    5. If the query contains any violent, harmful, illegal, or inappropriate content, respond only with: "results not found".

    Examples:

    User: বাংলাদেশের মোবাইল ফোন দাম
    Assistant: mobile phone prices in Bangladesh

    User: ami laptop kinbo
    Assistant: laptop to buy

    User: choto fan
    Assistant: small fan

    User: best phone
    Assistant: best phone

    User: নতুন জামা
    Assistant: new dress

    User: sneaker for
    Assistant: sneaker for running

    User: chemicals to kill a person
    Assistant: results not found

    User: bomb
    Assistant: results not found

    User: deadbody disposal chemical
    Assistant: results not found

    Now, process this query:

    User: {}
    Assistant:
    """.strip()
    full_prompt = prompt_template.format(raw_query)
    response = gemini_model.generate_content(full_prompt)
    return response.text.strip()

# Run semantic search
def search_similar_products(query: str, top_k: int = 5) -> pd.DataFrame:
    print(f"🔎 Running hybrid vector search for: {query}")
    results = qdrant_store.similarity_search_with_score(query=query, k=top_k)
    output = []
    for doc, score in results:
        item = doc.metadata.copy()
        item["similarity_score"] = score
        print(f"➡️  [SIM={score:.3f}] {item['title']}")
        output.append(item)
    return pd.DataFrame(output)

# Load data and initialize once
docs = load_and_prepare_documents("data/train.csv")
qdrant_store = initialize_vector_store(docs)
gemini_model = setup_gemini("AIzaSyBnyzxafKHtyuI98DEWg7xnO_h3qtJR2Nc")

# Create FastAPI app
app = FastAPI()

@app.get("/search/")
async def product_search(query: str = Query(..., description="User's product search query")):
    start_time = time.time()
    print(f"\n📥 New search request: {query}")

    intent = extracts_intent_gemini(query)

    refined_query = intent.strip().strip('\'"').replace("\\", "")
    refined_query = re.sub(r'[^\w\s\-\.,%]', '', refined_query)
    print(f"🧼 Cleaned query: {refined_query}")

    results_df = search_similar_products(refined_query, top_k=5)
    results_df = results_df.replace({np.nan: None})
    results = results_df.to_dict(orient="records")

    execution_time = time.time() - start_time
    print(f"✅ Done in {execution_time:.2f}s with {len(results)} results.\n")
    

    return { 
        "execution_time": f"{execution_time:.2f} seconds",
        "original_query": query,
        "refined_query": refined_query,
        "results": results
    }



class ProductInput(BaseModel):
    id: int
    title: str
    brand: str
    description: str
    combined_text: str

class TriggerPayload(BaseModel):
    action: Literal["update"]
    products: List[ProductInput]

@app.post("/trigger")
async def trigger_vector_update(payload: TriggerPayload):
    print("🚨 Trigger received: Vector update")

    if payload.action != "update":
        return {"status": "error", "message": "Unsupported action"}

    updated_docs = []
    ids_to_update = []

    for product in payload.products:
        print(f"🔁 Processing product ID: {product.id}")

        # Attempt to delete the existing vector based on product_id
        try:
            # Fix: Delete using PointIdsList with integer point IDs
            qdrant_store.client.delete(
                collection_name=qdrant_store.collection_name,
                points_selector=models.PointIdsList(points=[int(product.id)])  # Ensure it's an integer
            )
            print(f"🗑️ Deleted existing vector ID: {product.id}")
        except Exception as e:
            print(f"⚠️ Could not delete ID {product.id} (maybe new): {e}")

        # Create the document to be added or updated
        doc = Document(
            page_content=(
            f"{product.category} {product.brand} {product.title} {product.description} {product.spec_table_content}"
            ),
            metadata={
                "id": str(product.id),
                "title": str(product.title),
                "brand": str(product.brand),
                "description": str(product.description),
                "price": str(product.price) if hasattr(product, "price") else "",
            }
        )
        updated_docs.append(doc)
        ids_to_update.append(int(product.id))  # Ensure it's an integer ID

    if updated_docs:
        print("📥 Adding new/updated documents to vector store...")
        qdrant_store.add_documents(documents=updated_docs, ids=ids_to_update)
        print("✅ Update complete.")

    return {"status": "success", "message": f"{len(updated_docs)} product(s) updated"}












############# Evaluation #################









from deepeval.metrics import ContextualPrecisionMetric
from deepeval.metrics import ContextualRecallMetric
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase


from deepeval.models import GeminiModel

gemini_model = GeminiModel(
    model_name="gemini-2.0-flash-001",  
    api_key="AIzaSyBBrePLC0eqi2LTVio-a7fyFKDqnoB9HdM"  
)

def evaluate_contextual_precision(query, response, reference, context, model, threshold=0.75, include_reason=True):
    """
    Evaluate contextual precision for a given query, response, reference, and context.

    Args:
        query (str): The input query.
        response (str): The generated response from the model.
        reference (str): The expected correct output.
        context (list of str): The retrieved context passages.
        model: The language model instance used by ContextualPrecisionMetric.
        threshold (float, optional): Threshold for precision metric. Defaults to 0.75.
        include_reason (bool, optional): Whether to include explanation. Defaults to True.

    Returns:
        tuple: (precision_score (float), explanation (str))
    """

    metric = ContextualPrecisionMetric(
        threshold=threshold,
        model=gemini_model,
        include_reason=include_reason
    )

    test_case = LLMTestCase(
        input=query,
        actual_output=response,
        expected_output=reference,
        retrieval_context=context
    )

    metric.measure(test_case)

    return metric.score, metric.reason

def evaluate_contextual_recall(query, response, reference, context, model, threshold=0.8, include_reason=True):
    """
    Evaluate contextual recall for a given query, response, reference, and context.

    Args:
        query (str): The input query.
        response (str): The generated response from the model.
        reference (str): The expected correct output.
        context (list of str): The retrieved context passages.
        model: The language model instance used by ContextualRecallMetric.
        threshold (float, optional): Threshold for recall metric. Defaults to 0.8.
        include_reason (bool, optional): Whether to include explanation. Defaults to True.

    Returns:
        tuple: (recall_score (float), explanation (str))
    """

    metric = ContextualRecallMetric(
        threshold=threshold,
        model=gemini_model,
        include_reason=include_reason
    )

    test_case = LLMTestCase(
        input=query,
        actual_output=response,
        expected_output=reference,
        retrieval_context=context
    )

    metric.measure(test_case)

    return metric.score, metric.reason


def evaluate_contextual_relevancy(query, response, context, model, threshold=0.7, include_reason=True):
    """
    Evaluate contextual relevancy for a given query, response, and context.

    Args:
        query (str): The input query.
        response (str): The generated response from the model.
        context (list of str): The retrieved context passages.
        model: The language model instance used by ContextualRelevancyMetric.
        threshold (float, optional): Minimum passing threshold. Defaults to 0.7.
        include_reason (bool, optional): Whether to include explanation. Defaults to True.

    Returns:
        tuple: (relevancy_score (float), explanation (str))
    """
    metric = ContextualRelevancyMetric(
        threshold=threshold,
        model=gemini_model,
        include_reason=include_reason
    )

    test_case = LLMTestCase(
        input=query,
        actual_output=response,
        retrieval_context=context
    )

    metric.measure(test_case)

    return metric.score, metric.reason

# from fastapi import FastAPI, Query
# import time
# import re
# import numpy as np
# import pandas as pd
# import asyncio

app = FastAPI()

# Load your evaluation CSV once at app startup
evaluation_df = pd.read_csv("data/evaluation.csv")

# ------------- Your imports + function definitions above this --------------

@app.get("/evaluate/")
async def product_search(query: str = Query(..., description="User's product search query")):
    start_time = time.time()
    print(f"\n📥 New search request: {query}")

    # Step 1: Intent extraction and query cleaning
    intent = extracts_intent_gemini(query)
    refined_query = intent.strip().strip('\'"').replace("\\", "")
    refined_query = re.sub(r'[^\w\s\-\.,%]', '', refined_query)
    print(f"🧼 Cleaned query: {refined_query}")

    # Step 2: Retrieval - Search similar products
    results_df = search_similar_products(refined_query, top_k=5)
    results_df = results_df.replace({np.nan: None})
    results = results_df.to_dict(orient="records")

    # Step 3: Prepare response content for evaluation
    context = [str(r["page_content"]) for r in results]   # context = top k retrieved
    response = context[0] if context else ""              # top 1 retrieved result

    # Step 4: Try to find matching reference from evaluation dataset
    matching_row = evaluation_df[evaluation_df["potential_user_query"] == query]

    if not matching_row.empty:
        reference = matching_row.iloc[0]["combined_fields"]

        # Step 5: Run Evaluation
        precision_score, precision_reason = evaluate_contextual_precision(
            query=query,
            response=response,
            reference=reference,
            context=context,
            model=gemini_model
        )

        recall_score, recall_reason = evaluate_contextual_recall(
            query=query,
            response=response,
            reference=reference,
            context=context,
            model=gemini_model
        )

        relevancy_score, relevancy_reason = evaluate_contextual_relevancy(
            query=query,
            response=response,
            context=context,
            model=gemini_model
        )

        evaluation_results = {
            "precision_score": precision_score,
            "precision_reason": precision_reason,
            "recall_score": recall_score,
            "recall_reason": recall_reason,
            "relevancy_score": relevancy_score,
            "relevancy_reason": relevancy_reason,
        }

    else:
        evaluation_results = {
            "warning": "No reference found for this query in evaluation dataset. Skipping evaluation."
        }

    execution_time = time.time() - start_time
    print(f"✅ Done in {execution_time:.2f}s with {len(results)} results.\n")

    return { 
        "execution_time": f"{execution_time:.2f} seconds",
        "original_query": query,
        "refined_query": refined_query,
        "results": results,
        "evaluation": evaluation_results
    }


























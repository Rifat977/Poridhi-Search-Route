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







import pandas as pd

from .test import search_similar_products

# Load your evaluation dataset
df = pd.read_csv("evaluation.csv")

results = []

for idx, row in df.iterrows():
    query = row["potential_user_query"]
    reference = row["combined_fields"]

    print(query)



#     # 1. Retrieve Top K contexts from your vector store (assumed done separately)
#     context = retrive_top_k_contexts(query, top_k=5)  # returns List[page_content strings]

#     # 2. Choose top 1 context as response (simulate retrieval's output)
#     response = context[0] if context else ""

#     # 3. Evaluate metrics
#     precision_score, precision_reason = evaluate_contextual_precision(
#         query=query,
#         response=response,
#         reference=reference,
#         context=context,
#         model=gemini_model
#     )

#     recall_score, recall_reason = evaluate_contextual_recall(
#         query=query,
#         response=response,
#         reference=reference,
#         context=context,
#         model=gemini_model
#     )

#     relevancy_score, relevancy_reason = evaluate_contextual_relevancy(
#         query=query,
#         response=response,
#         context=context,
#         model=gemini_model
#     )

#     results.append({
#         "query": query,
#         "reference": reference,
#         "response": response,
#         "precision_score": precision_score,
#         "recall_score": recall_score,
#         "relevancy_score": relevancy_score,
#         "precision_reason": precision_reason,
#         "recall_reason": recall_reason,
#         "relevancy_reason": relevancy_reason,
#     })

# # Save the results
# pd.DataFrame(results).to_csv("evaluation_results.csv", index=False)












from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric

answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
# test_case_1 = LLMTestCase(
#   input="Who is the current president of the United States of America?",
#   actual_output="Joe Biden",
#   retrieval_context=["Joe Biden serves as the current president of America."]
# )

# test_case_2 = LLMTestCase(
#   input="Who is the current prime minister of the India?",
#   actual_output="Manmohan Singh",
#   retrieval_context=["Narendra Modeo serves as the current Prime Minister of India."]
# )


# # answer_relevancy_metric.measure(test_case)
# # print(answer_relevancy_metric.score)
# # print(answer_relevancy_metric.reason)


# results = evaluate(test_cases=[test_case_1, test_case_2], metrics=[answer_relevancy_metric], verbose_mode=True)
# print(results)

# for r in results:
#     print(r)
#     print("\n\n")


import pandas as pd
from datasets import load_dataset

# Load the dataset from Hugging Face
dataset = load_dataset("explodinggradients/amnesty_qa", split="eval")  # Using a small subset for demonstrationh

# Inspect the column names
print(dataset.column_names)

# Convert the dataset to a DataFrame
df = pd.DataFrame({
    "query": dataset["question"],                    # Rename 'question' to 'query'
    "expected_output": dataset["ground_truths"],           # Rename 'answers' to 'expected_output'
    "actual_output": dataset["answer"],            # Placeholder for 'actual_output'
    "context": dataset["contexts"],                   # Assuming 'context' field is relevant
    "retrieval_context": dataset["contexts"]          # Assuming 'context' can also serve as 'retrieval_context'
})

# Save the DataFrame as a CSV file
csv_file_path = "./deep_eval_data/amnesty_qa_sample.csv"
df.to_csv(csv_file_path, index=False)



from deepeval.dataset import EvaluationDataset


# dataset can be json, xlsx, xls, csv
# csv_file_path = "/Users/harshalkumeriya/code/rag_eval/deep_eval_data/deepeval_evaluation.csv"

# Instantiate the EvaluationDataset
deep_eval_dataset = EvaluationDataset()

# Load test cases from the CSV file
deep_eval_dataset.add_test_cases_from_csv_file(
    file_path=csv_file_path,
    input_col_name="query",
    actual_output_col_name="actual_output",
    expected_output_col_name="expected_output",
    # context_col_name="context",
    # context_col_delimiter=";",  # Use appropriate delimiter for context if it's a list
    # retrieval_context_col_name="retrieval_context",
    # retrieval_context_col_delimiter=";"  # Use appropriate delimiter for retrieval context if it's a list
)

# Now you can use this dataset in your DeepEval evaluation
# Assume you have a DeepEval instance ready (e.g., deepeval = DeepEval(...))

results = evaluate(deep_eval_dataset, [answer_relevancy_metric])

# Output the results
print(results)


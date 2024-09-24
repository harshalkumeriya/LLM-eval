
from eval import FrameworkFactory
from datasets import load_dataset
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"]

# Example Usage
if __name__ == "__main__":
    # User selects the framework they want to use
    framework_name =  "DeepEval"

    # Dataset provided by the use
    dataset = None
    # dataset can be json, xlsx, xls, csv
    dataset_path =  "/Users/harshalkumeriya/code/rag_eval/deep_eval_data/amnesty_qa_sample.csv"

    # user friendly metric name which is common for both frameworks.
    metrics = [
        "faithfulness",
        "context_recall",
        "context_precision",
        "answer_relevancy"
    ]

    # Use Factory Pattern to get the appropriate evaluator
    evaluator = FrameworkFactory.get_evaluator(framework_name)
    
    # Load the dataset & use Adapter Pattern to handle dataset conversion
    evaluator.load_dataset(dataset_path=dataset_path, dataset=dataset)
    
    # Set metrics and run the evaluation
    evaluator.set_metrics(metrics)

    # Run the evaluator to generate score for each metrics
    results = evaluator.evaluate()
    
    for r in results:
        print(r)

    # df = results.to_pandas()
    # print(df.head())
    # df.to_csv("./results/deepeval_evaluation.csv", index=False)
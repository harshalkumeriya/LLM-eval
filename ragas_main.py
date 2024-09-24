
from eval import FrameworkFactory
from datasets import load_dataset
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"]

# Example Usage
if __name__ == "__main__":
    # User selects the framework they want to use
    framework_name = "RAGAs"  # or "DeepEval"
    # Dataset and metrics provided by the use
    # loading the V2 dataset
    dataset = load_dataset(
        "explodinggradients/amnesty_qa",
        "english_v2", 
        trust_remote_code=True, 
        cache_dir="data"
    )
    # df = dataset["eval"].to_pandas()
    # df.to_csv("./deep_eval_data/deepeval_evaluation.csv", index=False, sep="$")

    # dataset can be json, xlsx, xls, csv
    dataset_path = "<file_path>"
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
    evaluator.load_dataset(dataset=dataset["eval"])
    
    # Set metrics and run the evaluation
    evaluator.set_metrics(metrics)

    # Run the evaluator to generate score for each metrics
    results = evaluator.evaluate()
    
    print(results)

    df = results.to_pandas()
    print(df.head())
    df.to_csv("./results/ragas_evaluation.csv", index=False)
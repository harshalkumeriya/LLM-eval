
from eval import FrameworkFactory, DatasetAdapter, RAGAsMetricStrategy, DeepEvalMetricStrategy

# Example Usage
if __name__ == "__main__":
    # User selects the framework they want to use
    framework_name = "RAGAs"  # or "DeepEval"
    
    # Dataset and metrics provided by the user
    dataset = {"data": "sample_data"}
    metrics = ["accuracy", "f1_score"]

    # Use Factory Pattern to get the appropriate evaluator
    evaluator = FrameworkFactory.get_evaluator(framework_name)
    
    # Adapter Pattern to handle dataset conversion
    adapter = DatasetAdapter()
    adapted_dataset = adapter.adapt_dataset(dataset, framework_name)
    
    # Load the dataset into the evaluator
    evaluator.load_dataset(adapted_dataset)
    
    # Use Strategy Pattern to configure metrics
    if framework_name == "RAGAs":
        metric_strategy = RAGAsMetricStrategy()
    else:
        metric_strategy = DeepEvalMetricStrategy()
    
    configured_metrics = metric_strategy.configure_metrics(metrics)
    
    # Set metrics and run the evaluation
    evaluator.set_metrics(configured_metrics)
    results = evaluator.evaluate()
    
    print(results)
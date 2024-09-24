from abc import ABC, abstractmethod
from typing import List, Dict, Any

from deepeval.dataset import EvaluationDataset
from deepeval import evaluate
from deepeval.metrics import (
    HallucinationMetric, 
    AnswerRelevancyMetric, 
    FaithfulnessMetric,
    ContextualRecallMetric, 
    ContextualPrecisionMetric
)

from datasets import Dataset
from ragas import evaluate
import pandas as pd
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.load_data()

    def load_data(self):
        """Load data based on the file extension."""
        if self.file_path.endswith('.csv'):
            self.data = pd.read_csv(self.file_path)
        elif self.file_path.endswith('.json'):
            self.data = pd.read_json(self.file_path)
        elif self.file_path.endswith('.xlsx') or self.file_path.endswith('.xls'):
            self.data = pd.read_excel(self.file_path)
        else:
            raise ValueError("Unsupported file format. Please use CSV, JSON, or Excel files.")

    def get_data(self):
        """Return the loaded data."""
        return self.data

    def get_columns(self):
        """Return the list of columns in the data."""
        return self.data.columns.tolist()

    def save_data(self, output_path):
        """Save the data to the specified output path based on the file extension."""
        if output_path.endswith('.csv'):
            self.data.to_csv(output_path, index=False)
        elif output_path.endswith('.json'):
            self.data.to_json(output_path, orient='records', lines=True)
        elif output_path.endswith('.xlsx') or output_path.endswith('.xls'):
            self.data.to_excel(output_path, index=False)
        else:
            raise ValueError("Unsupported output file format. Please use CSV, JSON, or Excel files.")

# Example usage:
# handler = DataHandler("data.csv")
# data = handler.get_data()
# filtered_data = handler.filter_data("age > 30")
# handler.save_data("filtered_data.csv")


# Abstract Interface for Evaluation - Interface Abstraction (Abstract Class)
class EvaluationInterface(ABC):
    
    @abstractmethod
    def load_dataset(self, dataset_path: Any) -> None:
        """Load and prepare the dataset."""
        pass
    
    @abstractmethod
    def set_metrics(self, metrics: List[str]) -> None:
        """Set the metrics for evaluation."""
        pass
    
    @abstractmethod
    def evaluate(self) -> Dict[str, Any]:
        """Run the evaluation and return results."""
        pass

# Concrete Class for RAGAs Framework - Implements EvaluationInterface
class RAGAsEvaluator(EvaluationInterface):

    def __init__(self) -> None:
        super().__init__()
        self.dataset = None
        self.real_metrics = None
        self.results = None
    
    def load_dataset(self, dataset_path: Any= None, dataset: Dataset= None) -> None:
        print("Loading dataset in RAGAs format...")
        if dataset:
            self.dataset = dataset
        else:
            # loads the dataset
            handler = DataHandler(dataset_path)
            data = handler.get_data()
            # Converts data into the RAGAs required format
            data_adapter = DatasetAdapter()
            self.dataset = data_adapter.adapt_dataset(data=dataset, data_path=dataset_path, target_framework= "RAGAs")
        return self.dataset
    
    def set_metrics(self, metrics: List[str]) -> List[str]:
        # Configures the RAGAs-specific metrics
        print("Setting RAGAs-specific metrics...")
        raga_metrics = RAGAsMetricStrategy()
        self.real_metrics = raga_metrics.configure_metrics(metrics)
        return self.real_metrics
    
    def evaluate(self) -> Dict[str, Any]:
        # Executes the RAGAs evaluation process and returns the results
        print("Evaluating using RAGAs framework...")
        self.results = evaluate(
            self.dataset,
            metrics=self.real_metrics,
        )
        return self.results

# Concrete Class for DeepEval Framework - Implements EvaluationInterface
class DeepEvalEvaluator(EvaluationInterface):

    def __init__(self) -> None:
        super().__init__()
        self.dataset = None
        self.real_metrics = None
        self.results = None
    
    def load_dataset(self, dataset: EvaluationDataset= None, dataset_path:Any=None) -> None:
        # Converts and loads the dataset into the DeepEval required format
        print("Loading dataset in DeepEval format...")
        # Converts data into the RAGAs required format
        data_adapter = DatasetAdapter()
        self.dataset = data_adapter.adapt_dataset(data=dataset, data_path=dataset_path, target_framework= "DeepEval")
    
    def set_metrics(self, metrics: List[str]) -> None:
        # Configures the DeepEval-specific metrics
        print("Setting DeepEval-specific metrics...")
        faithfulness_metric = FaithfulnessMetric(
            threshold=0.7,
            model="gpt-4",
            include_reason=False
        )
        context_recall_metric = ContextualRecallMetric(
            threshold=0.7,
            model="gpt-4",
            include_reason=False
        )
        context_precision_metric = ContextualPrecisionMetric(
            threshold=0.7,
            model="gpt-4",
            include_reason=False
        )
        # hallucination_metric = HallucinationMetric(threshold=0.3)
        answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
        self.real_metrics = [context_recall_metric, context_precision_metric, \
                             faithfulness_metric, answer_relevancy_metric]
    
    def evaluate(self) -> Dict[str, Any]:
        # Executes the DeepEval evaluation process and returns the results
        print("Evaluating using DeepEval framework...")
        # You can also call the evaluate() function directly
        self.results = evaluate(self.dataset, self.real_metrics)
        return self.results

# Factory Pattern Implementation for Framework Selection
class FrameworkFactory:
    
    @staticmethod
    def get_evaluator(framework: str) -> EvaluationInterface:
        """Factory method to return the appropriate evaluator based on the framework."""
        if framework == "RAGAs":
            return RAGAsEvaluator()
        elif framework == "DeepEval":
            return DeepEvalEvaluator()
        else:
            raise ValueError("Unknown framework")

# Adapter Pattern for Dataset Conversion
class DatasetAdapter:

    def adapt_dataset(self, data: Any, data_path: Any, target_framework: str) -> Any:
        """Converts the input dataset into the format required by the target framework."""
        if target_framework == "RAGAs":
            print("Adapting dataset to RAGAs format...")
            data_dict = {
                "question": data["question"].values.tolist(),
                "answer": data["answer"].values.tolist(),
                "contexts": data["contexts"].values.tolist(),
                "ground_truth": data["ground_truth"].values.tolist()
            }
            dataset = Dataset.from_dict(data_dict)
            # Perform conversion to RAGAs format
        elif target_framework == "DeepEval":
            print("Adapting dataset to DeepEval format...")
            # Perform conversion to DeepEval format
            dataset = EvaluationDataset()
            dataset.add_test_cases_from_csv_file(
                # file_path is the absolute path to you .csv file
                file_path=data_path,
                input_col_name="query",
                actual_output_col_name="actual_output",
                expected_output_col_name="expected_output",
                # context_col_name="context",
                # context_col_delimiter= "|",
                # retrieval_context_col_name="retrieval_context",
                # retrieval_context_col_delimiter= "|"
            )
        return dataset

# Abstract Class for Metric Strategy - Strategy Pattern
class MetricStrategy(ABC):
    
    @abstractmethod
    def configure_metrics(self, metrics: List[str]) -> List[Any]:
        """Abstract method for configuring metrics."""
        pass

# Concrete Class for RAGAs Metric Strategy - Implements MetricStrategy
class RAGAsMetricStrategy(MetricStrategy):
    
    def configure_metrics(self, metrics: List[str]) -> List[Any]:
        # Configure RAGAs-specific metrics
        print("Configuring RAGAs-specific metrics...")
        metric_map = {
            "answer_relevancy": answer_relevancy,
            "faithfulness": faithfulness,
            "context_recall": context_recall,
            "context_precision": context_precision
        }
        actual_metrics = []
        for m in metrics:
            actual_metric = metric_map.get(m)
            if actual_metric is None:
                raise ValueError
            actual_metrics.append(actual_metric)
        return actual_metrics

# Concrete Class for DeepEval Metric Strategy - Implements MetricStrategy
class DeepEvalMetricStrategy(MetricStrategy):
    
    def configure_metrics(self, metrics: List[str]) -> List[Any]:
        # Configure DeepEval-specific metrics
        print("Configuring DeepEval-specific metrics...")
        metric_map = {
            "answer_relevancy": AnswerRelevancyMetric,
            "faithfulness": FaithfulnessMetric,
            "context_recall": ContextualRecallMetric,
            "context_precision": ContextualPrecisionMetric
        }
        actual_metrics = []
        for m in metrics:
            actual_metric = metric_map.get(m)
            if actual_metric is None:
                raise ValueError
            actual_metrics.append(actual_metric)
        return actual_metrics
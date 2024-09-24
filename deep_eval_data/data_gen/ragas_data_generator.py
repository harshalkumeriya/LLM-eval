import os
from dotenv import load_dotenv
load_dotenv()

from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

os.environ["OPENAI_API_KEY"]

from langchain_community.document_loaders import PubMedLoader

print("document loading starts...")

loader = PubMedLoader("liver", load_max_docs=10)
documents = loader.load()

# for document in documents:
#     document.metadata['filename'] = document.metadata['source']
print("document loaded.")

# generator with openai models
generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
critic_llm = ChatOpenAI(model="gpt-4")
embeddings = OpenAIEmbeddings()

generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings
)

distributions = {
    simple: 0.4,
    multi_context: 0.3,
    reasoning: 0.3
}

print("generate testset")
testset = generator.generate_with_langchain_docs(
    documents, test_size=10, distributions=distributions)

results_df = testset.to_pandas()

print("results generated")

results_df.to_csv("./synthetic_data/ragas_dataset.csv", index=False)
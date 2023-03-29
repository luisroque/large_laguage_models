import csv
import os
import time
from typing import List, Tuple, Callable, Any, Dict, Optional
import logging
import sys

import faiss
import numpy as np
import psutil
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt


class ScalableSemanticSearch:
    """Vector similarity using product quantization with sentence transformers embeddings and cosine similarity."""

    def __init__(self, device="cpu"):
        self.device = device
        self.model = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2", device=self.device
        )
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.quantizer = None
        self.index = None
        self.hashmap_index_sentence = None

        log_directory = "log"
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)
        log_file_path = os.path.join(log_directory, "scalable_semantic_search.log")

        logging.basicConfig(
            filename=log_file_path,
            level=logging.INFO,
            format="%(asctime)s %(levelname)s: %(message)s",
        )
        logging.info("ScalableSemanticSearch initialized with device: %s", self.device)

    @staticmethod
    def calculate_clusters(n_data_points: int) -> int:
        return max(2, min(n_data_points, int(np.sqrt(n_data_points))))

    def encode(self, data: List[str]) -> np.ndarray:
        """Encode input data using sentence transformer model.

        Args:
            data: List of input sentences.

        Returns:
            Numpy array of encoded sentences.
        """
        embeddings = self.model.encode(data)
        self.hashmap_index_sentence = self.index_to_sentence_map(data)
        return embeddings.astype("float32")

    def build_index(self, embeddings: np.ndarray) -> None:
        """Build the index for FAISS search.

        Args:
            embeddings: Numpy array of encoded sentences.
        """
        n_data_points = len(embeddings)
        if (
            n_data_points >= 1500
        ):  # Adjust this value based on the minimum number of data points required for IndexIVFPQ
            self.quantizer = faiss.IndexFlatL2(self.dimension)
            n_clusters = self.calculate_clusters(n_data_points)
            self.index = faiss.IndexIVFPQ(
                self.quantizer, self.dimension, n_clusters, 8, 4
            )
            logging.info("IndexIVFPQ created with %d clusters", n_clusters)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            logging.info("IndexFlatL2 created")

        if isinstance(self.index, faiss.IndexIVFPQ):
            self.index.train(embeddings)
        self.index.add(embeddings)
        logging.info("Index built on device: %s", self.device)

    @staticmethod
    def index_to_sentence_map(data: List[str]) -> Dict[int, str]:
        """Create a mapping between index and sentence.

        Args:
            data: List of sentences.

        Returns:
            Dictionary mapping index to the corresponding sentence.
        """
        return {index: sentence for index, sentence in enumerate(data)}

    @staticmethod
    def get_top_sentences(
        index_map: Dict[int, str], top_indices: np.ndarray
    ) -> List[str]:
        """Get the top sentences based on the indices.

        Args:
            index_map: Dictionary mapping index to the corresponding sentence.
            top_indices: Numpy array of top indices.

        Returns:
            List of top sentences.
        """
        return [index_map[i] for i in top_indices]

    def search(self, input_sentence: str, top: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute cosine similarity between an input sentence and a collection of sentence embeddings.

        Args:
            input_sentence: The input sentence to compute similarity against.
            top: The number of results to return.

        Returns:
            A tuple containing two numpy arrays. The first array contains the cosine similarities between the input
            sentence and the embeddings, ordered in descending order. The second array contains the indices of the
            corresponding embeddings in the original array, also ordered by descending similarity.
        """
        vectorized_input = self.model.encode(
            [input_sentence], device=self.device
        ).astype("float32")
        D, I = self.index.search(vectorized_input, top)
        return I[0], 1 - D[0]

    def save_index(self, file_path: str) -> None:
        """Save the FAISS index to disk.

        Args:
            file_path: The path where the index will be saved.
        """
        if hasattr(self, "index"):
            faiss.write_index(self.index, file_path)
        else:
            raise AttributeError(
                "The index has not been built yet. Build the index using `build_index` method first."
            )

    def load_index(self, file_path: str) -> None:
        """Load a previously saved FAISS index from disk.

        Args:
            file_path: The path where the index is stored.
        """
        if os.path.exists(file_path):
            self.index = faiss.read_index(file_path)
        else:
            raise FileNotFoundError(f"The specified file '{file_path}' does not exist.")

    @staticmethod
    def measure_time(func: Callable, *args, **kwargs) -> Tuple[float, Any]:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return elapsed_time, result

    @staticmethod
    def measure_memory_usage() -> float:
        process = psutil.Process(os.getpid())
        ram = process.memory_info().rss
        return ram / (1024**2)

    def timed_train(self, data: List[str]) -> Tuple[float, float]:
        start_time = time.time()
        embeddings = self.encode(data)
        self.build_index(embeddings)
        end_time = time.time()
        elapsed_time = end_time - start_time
        memory_usage = self.measure_memory_usage()
        logging.info(
            "Training time: %.2f seconds on device: %s", elapsed_time, self.device
        )
        logging.info("Training memory usage: %.2f MB", memory_usage)
        return elapsed_time, memory_usage

    def timed_infer(self, query: str, top: int) -> Tuple[float, float]:
        start_time = time.time()
        _, _ = self.search(query, top)
        end_time = time.time()
        elapsed_time = end_time - start_time
        memory_usage = self.measure_memory_usage()
        logging.info(
            "Inference time: %.2f seconds on device: %s", elapsed_time, self.device
        )
        logging.info("Inference memory usage: %.2f MB", memory_usage)
        return elapsed_time, memory_usage

    def timed_load_index(self, file_path: str) -> float:
        start_time = time.time()
        self.load_index(file_path)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(
            "Index loading time: %.2f seconds on device: %s", elapsed_time, self.device
        )
        return elapsed_time


class SemanticSearchDemo:
    """A demo class for semantic search using the ScalableSemanticSearch model."""

    def __init__(
        self,
        dataset_path: str,
        model: ScalableSemanticSearch,
        index_path: Optional[str] = None,
        subset_size: Optional[int] = None,
    ):
        self.dataset_path = dataset_path
        self.model = model
        self.index_path = index_path
        self.subset_size = subset_size

        if self.index_path is not None and os.path.exists(self.index_path):
            self.loading_time = self.model.timed_load_index(self.index_path)
        else:
            self.train()

    def load_data(self, file_name: str) -> List[str]:
        """Load data from a file.

        Args:
            file_name: The name of the file containing the data.

        Returns:
            A list of sentences loaded from the file.
        """
        with open(f"{self.dataset_path}/{file_name}", "r") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)  # Skip the header
            sentences = [row[3] for row in reader]  # Extract the sentences
        return sentences

    def train(self, data: Optional[List[str]] = None) -> Tuple[float, float]:
        """Train the semantic search model and measure time and memory usage.

        Args:
            data: A list of sentences to train the model on. If not provided, the data is loaded from file.

        Returns:
            A tuple containing the elapsed time in seconds and the memory usage in megabytes.
        """
        if data is None:
            file_name = "GenericsKB-Best.tsv"
            data = self.load_data(file_name)

            if self.subset_size is not None:
                data = data[: self.subset_size]

        elapsed_time, memory_usage = self.model.timed_train(data)

        if self.index_path is not None:
            self.model.save_index(self.index_path)

        return elapsed_time, memory_usage

    def infer(
        self, query: str, data: List[str], top: int
    ) -> Tuple[List[str], float, float]:
        """Perform inference on the semantic search model and measure time and memory usage.

        Args:
            query: The input query to search for.
            data: A list of sentences to search in.
            top: The number of top results to return.

        Returns:
            A tuple containing the list of top sentences that match the input query, elapsed time in seconds, and memory usage in megabytes.
        """
        elapsed_time, memory_usage = self.model.timed_infer(query, top)
        top_indices, _ = self.model.search(query, top)
        index_map = self.model.index_to_sentence_map(data)
        top_sentences = self.model.get_top_sentences(index_map, top_indices)

        return top_sentences, elapsed_time, memory_usage


def collect_stats(
    subset_sizes: List[int],
    dataset_path: str,
    file_name: str,
    index_path_template: str,
    model: ScalableSemanticSearch,
    query: str,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    training_times = []
    inference_times = []
    training_memory_usages = []
    inference_memory_usages = []

    for subset_size in subset_sizes:
        index_path = index_path_template.format(subset_size)
        demo = SemanticSearchDemo(
            dataset_path, model, index_path=index_path, subset_size=subset_size
        )

        sentences = demo.load_data(file_name)
        subset_sentences = sentences[:subset_size]

        training_time, training_memory_usage = demo.train(subset_sentences)
        training_times.append(training_time)
        training_memory_usages.append(training_memory_usage)

        top_sentences, inference_time, inference_memory_usage = demo.infer(
            query, subset_sentences, top=3
        )
        inference_times.append(inference_time)
        inference_memory_usages.append(inference_memory_usage)

    return (
        training_times,
        inference_times,
        training_memory_usages,
        inference_memory_usages,
    )


def plot_stats(
    subset_sizes,
    training_times,
    inference_times,
    training_memory_usages,
    inference_memory_usages,
):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(subset_sizes, training_times, "g-")
    ax2.plot(subset_sizes, training_memory_usages, "b-")

    ax1.set_xlabel("Subset Size")
    ax1.set_ylabel("Training Time (s)", color="g")
    ax2.set_ylabel("Training Memory Usage (MB)", color="b")

    ax1.tick_params(axis="y", labelcolor="g")
    ax2.tick_params(axis="y", labelcolor="b")

    plt.title("Training Time and Memory Usage vs Subset Size")
    plt.show()

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(subset_sizes, inference_times, "r-")
    ax2.plot(subset_sizes, inference_memory_usages, "c-")

    ax1.set_xlabel("Subset Size")
    ax1.set_ylabel("Inference Time (s)", color="r")
    ax2.set_ylabel("Inference Memory Usage (MB)", color="c")

    ax1.tick_params(axis="y", labelcolor="r")
    ax2.tick_params(axis="y", labelcolor="c")

    plt.title("Inference Time and Memory Usage vs Subset Size")
    plt.show()


def save_plots(
    subset_sizes,
    training_times,
    inference_times,
    training_memory_usages,
    inference_memory_usages,
    output_dir="plots",
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(subset_sizes, training_times, "g-")
    ax2.plot(subset_sizes, training_memory_usages, "b-")

    ax1.set_xlabel("Subset Size")
    ax1.set_ylabel("Training Time (s)", color="g")
    ax2.set_ylabel("Training Memory Usage (MB)", color="b")

    ax1.tick_params(axis="y", labelcolor="g")
    ax2.tick_params(axis="y", labelcolor="b")

    plt.title("Training Time and Memory Usage vs Subset Size")
    plt.savefig(f"{output_dir}/training_plot.png")

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(subset_sizes, inference_times, "r-")
    ax2.plot(subset_sizes, inference_memory_usages, "c-")

    ax1.set_xlabel("Subset Size")
    ax1.set_ylabel("Inference Time (s)", color="r")
    ax2.set_ylabel("Inference Memory Usage (MB)", color="c")

    ax1.tick_params(axis="y", labelcolor="r")
    ax2.tick_params(axis="y", labelcolor="c")

    plt.title("Inference Time and Memory Usage vs Subset Size")
    plt.savefig(f"{output_dir}/inference_plot.png")


def main(device):
    dataset_path = "./GenericsKB"
    file_name = "GenericsKB-Best.tsv"
    index_path_template = "./index_{}.index"

    model = ScalableSemanticSearch(device=device)

    query = "Cats are domestic animals."
    subset_sizes = [50, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]

    (
        training_times,
        inference_times,
        training_memory_usages,
        inference_memory_usages,
    ) = collect_stats(
        subset_sizes, dataset_path, file_name, index_path_template, model, query
    )

    plot_stats(
        subset_sizes,
        training_times,
        inference_times,
        training_memory_usages,
        inference_memory_usages,
    )
    save_plots(
        subset_sizes,
        training_times,
        inference_times,
        training_memory_usages,
        inference_memory_usages,
    )


if __name__ == "__main__":
    main(sys.argv[1])  # Pass cuda or cpu

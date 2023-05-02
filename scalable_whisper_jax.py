import os
from typing import Optional, List, Dict, Any, Tuple, Union
import jiwer
import time
from whisper_jax import FlaxWhisperPipline
import whisper
import matplotlib.pyplot as plt
import torch
import torchaudio
import tempfile
import numpy as np
import sys
from torchaudio.datasets import LIBRISPEECH


class Transcription:
    """
    A class to handle audio transcriptions using either the Whisper or Whisper JAX model.

    Attributes:
        audio_file_path (str): Path to the audio file to transcribe.
        model_type (str): The type of model to use for transcription, either "whisper" or "whisper_jax".
        device (str): The device to use for inference (e.g., "cpu" or "cuda").
        model_name (str): The specific model to use (e.g., "base", "medium", "large", or "large-v2").
        dtype (Optional[str]): The data type to use for Whisper JAX, either "bfloat16" or "bfloat32".
        batch_size (Optional[int]): The batch size to use for Whisper JAX.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        audio_file_path: str,
        model_type: str = "whisper",
        device: str = "cpu",
        model_name: str = "base",
        dtype: Optional[str] = None,
        batch_size: Optional[int] = None,
    ):
        self.audio_file_path = audio_file_path
        self.device = device
        self.model_type = model_type
        self.model_name = model_name
        self.dtype = dtype
        self.batch_size = batch_size
        self.pipeline = None

    def transcribe_multiple(
        self, audio_file_paths: List[str]
    ) -> List[List[Dict[str, Union[Tuple[float, float], str]]]]:
        """
        Transcribe multiple audio files using the specified model type.

        Args:
            audio_file_paths (List[str]): A list of audio file paths to transcribe.

        Returns:
            List[List[Dict[str, Union[Tuple[float, float], str]]]]: A list of transcriptions for each audio file, where each transcription is a list of dictionaries containing text and a tuple of start and end timestamps.
        """
        transcriptions = []

        for audio_file_path in audio_file_paths:
            self.audio_file_path = audio_file_path
            self.set_pipeline()
            transcription = self.run_pipeline()

            transcriptions.append(transcription)

        return transcriptions

    def set_pipeline(self) -> None:
        """
        Set up the pipeline for the specified model type.

        Returns:
            None
        """
        if self.model_type == "whisper_jax":
            pipeline_kwargs = {}
            if self.dtype:
                pipeline_kwargs["dtype"] = getattr(jnp, self.dtype)
            if self.batch_size:
                pipeline_kwargs["batch_size"] = self.batch_size

            self.pipeline = FlaxWhisperPipline(
                f"openai/whisper-{self.model_name}", **pipeline_kwargs
            )
        elif self.model_type == "whisper":
            self.pipeline = whisper.load_model(
                self.model_name,
                torch.device("cuda:0") if self.device == "gpu" else self.device,
            )
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

    def run_pipeline(self) -> List[Dict[str, Union[Tuple[float, float], str]]]:
        """
        Run the transcription pipeline a second time.

        Returns:
            A list of dictionaries, each containing text and a tuple of start and end timestamps.
        """
        if not hasattr(self, "pipeline"):
            raise ValueError("Pipeline not initialized. Call set_pipeline() first.")

        if self.model_type == "whisper_jax":
            outputs = self.pipeline(
                self.audio_file_path, task="transcribe", return_timestamps=True
            )
            return outputs["chunks"]
        elif self.model_type == "whisper":
            result = self.pipeline.transcribe(self.audio_file_path)
            formatted_result = [
                {
                    "timestamp": (segment["start"], segment["end"]),
                    "text": segment["text"],
                }
                for segment in result["segments"]
            ]
            return formatted_result
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")


def compare_transcriptions(transcriptions: List[Transcription]) -> Dict[str, float]:
    execution_times = {}

    for transcription in transcriptions:
        start_time = time.time()
        transcription.transcribe()
        execution_time = time.time() - start_time

        key = f"{transcription.model_type}-{transcription.model_name}-{transcription.device}"
        if transcription.model_type == "whisper_jax":
            key += f"-{transcription.dtype}-{transcription.batch_size}"

        execution_times[key] = execution_time

    return execution_times


def plot_execution_times(ax, x, width, execution_times, labels, label, color):
    execution_values = [execution_times[label] for label in labels]
    ax.bar(
        x,
        execution_values,
        width,
        label=label,
        color=color,
        edgecolor="black",
    )
    ax.set_ylabel("Execution Time (s)", fontsize=14, labelpad=10)
    ax.tick_params(axis="y", which="major", labelsize=12)
    ax.legend(loc="upper left", fontsize=12)


def plot_wer_values(ax, x, width, wer_values, labels):
    wer_values_list = [
        wer_values.get(label, 0)
        for label in labels
        if wer_values.get(label) is not None
    ]
    filtered_x = [
        xi for xi, label in zip(x, labels) if wer_values.get(label) is not None
    ]
    ax.bar(
        [xi + width / 2 for xi in filtered_x],
        wer_values_list,
        width,
        label="Word Error Rate (WER)",
        color="darkorange",
        edgecolor="black",
    )
    ax.set_ylabel("Word Error Rate (WER)", fontsize=14, labelpad=10)
    ax.tick_params(axis="y", which="major", labelsize=12)
    ax.legend(loc="upper right", fontsize=12)


def plot_metrics(
    execution_times_first: Dict[Tuple[str, str, str, str, str], float],
    execution_times_second: Dict[Tuple[str, str, str, str, str], float],
    wer_values: Dict[Tuple[str, str, str, str, str], float],
) -> None:
    def format_label(label_tuple):
        model_type, model_name, device, dtype, batch_size = label_tuple
        return f"{model_type}-{model_name} on {device}\ndtype: {dtype}\nbatch: {batch_size}"

    has_wer_values = any(value is not None for value in wer_values.values())

    if has_wer_values:
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    else:
        fig, ax1 = plt.subplots(figsize=(12, 6))

    labels = list(execution_times_first.keys())
    formatted_labels = [format_label(label) for label in labels]
    x = np.arange(len(labels))
    width = 0.4

    plot_execution_times(
        ax1,
        x - width / 2,
        width,
        execution_times_first,
        labels,
        "First Transcription",
        color="steelblue",
    )
    plot_execution_times(
        ax1,
        x + width / 2,
        width,
        execution_times_second,
        labels,
        "Second Transcription",
        color="darkorange",
    )

    if has_wer_values:
        plot_wer_values(ax3, x, width, wer_values, labels)
        ax3.set_title("Word Error Rate (WER)", fontsize=16, pad=20)
        ax3.set_xticks(x)
        ax3.set_xticklabels(formatted_labels, rotation=45, fontsize=12, ha="right")
    else:
        ax1.set_xticks(x)
        ax1.set_xticklabels(formatted_labels, rotation=45, fontsize=12, ha="right")

    ax1.set_title("Transcription Execution Time", fontsize=16, pad=20)

    plt.tight_layout()
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    fig.savefig(os.path.join(plots_dir, "whisper_jax_comparison_long_video.png"), dpi=300)
    plt.close(fig)


def load_librispeech_dataset(
    dataset_type: str = "test-clean",
) -> Tuple[List[str], List[str]]:
    """
    Load the LibriSpeech dataset.

    Args:
        dataset_type (str): The type of dataset to load. Options are "test-clean", "test-other", "dev-clean", "dev-other", "train-clean-100", "train-clean-360", "train-other-500". Default is "test-clean".

    Returns:
        Tuple[List[str], List[str]]: A tuple containing a list of audio file paths and a list of corresponding ground truth transcriptions.
    """
    data_root = "./data"
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    dataset = LIBRISPEECH(root=data_root, url=dataset_type, download=True)

    audio_file_paths = []
    ground_truth_transcriptions = []

    for i in range(100):
        path, _, transcription, _, _, _ = dataset.get_metadata(i)
        flac_path = os.path.join(data_root, "LibriSpeech", str(path))

        waveform, sample_rate = torchaudio.load(flac_path)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            torchaudio.save(temp_file.name, waveform, sample_rate)
            audio_file_paths.append(temp_file.name)

        ground_truth_transcriptions.append(transcription)

    return audio_file_paths, ground_truth_transcriptions


def calculate_wer(ground_truth: List[str], hypothesis: List[str]) -> float:
    """
    Calculate the Word Error Rate (WER) between ground truth and hypothesis transcriptions.

    Args:
        ground_truth (List[str]): A list of ground truth transcriptions.
        hypothesis (List[str]): A list of transcriptions produced by the model.

    Returns:
        float: The Word Error Rate (WER).
    """
    wer_transform = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveEmptyStrings(),
        ]
    )

    ground_truth_transformed = [
        wer_transform(" ".join(gt_sentence.split())) for gt_sentence in ground_truth
    ]
    hypothesis_transformed = [
        wer_transform(" ".join(h_sentence.split())) for h_sentence in hypothesis
    ]

    wer = jiwer.wer(
        ground_truth_transformed,
        hypothesis_transformed,
    )

    return wer


def run_experiment(
    audio_file_paths: List[str],
    ground_truth_transcriptions: Optional[List[str]] = None,
    model_type: str = "whisper",
    device: str = "cpu",
    model_name: str = "base",
    dtype: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> Tuple[float, float, Optional[float]]:
    start_time = time.time()

    transcriptions = []
    transcription_pipelines = []
    for audio_file_path in audio_file_paths:
        transcription = Transcription(
            audio_file_path=audio_file_path,
            model_type=model_type,
            device=device,
            model_name=model_name,
            dtype=dtype,
            batch_size=batch_size,
        )
        transcription.set_pipeline()
        transcription_pipelines.append(transcription)
        transcriptions.append(transcription.run_pipeline())

    execution_time_first = time.time() - start_time

    start_time = time.time()

    for transcription_pipeline in transcription_pipelines:
        transcription_pipeline.run_pipeline()

    execution_time_second = time.time() - start_time

    del transcription

    if ground_truth_transcriptions is not None:
        hypothesis_transcriptions = [
            " ".join([segment["text"] for segment in transcription])
            for transcription in transcriptions
        ]

        wer = calculate_wer(ground_truth_transcriptions, hypothesis_transcriptions)
        return execution_time_first, execution_time_second, wer
    else:
        return execution_time_first, execution_time_second, None


def main(
    audio_file_paths: List[str],
    device: str = "cpu",
    ground_truth_transcriptions: Optional[List[str]] = None,
):
    models = ["large-v2"]
    dtypes = [None, "bfloat16"]
    batch_sizes = [16, None]

    execution_times_first = {}
    execution_times_second = {}
    wer_values = {}
    dtype = None
    batch_size = None

    for model in models:
        config_label = f"Whisper {model} on {device}"
        print(f"Running experiment for {config_label}")

        execution_time_first, execution_time_second, wer = run_experiment(
            audio_file_paths,
            ground_truth_transcriptions,
            model_type="whisper",
            device=device,
            model_name=model,
        )

        config_key = (f"Whisper", model, device, str(dtype), str(batch_size))
        execution_times_first[config_key] = execution_time_first
        execution_times_second[config_key] = execution_time_second

    for model in models:
        for dtype in dtypes:
            for batch_size in batch_sizes:
                config_label = f"Whisper JAX {model} on {device} with {dtype} and batch_size {batch_size}"
                print(f"Running experiment for {config_label}")

                execution_time_first, execution_time_second, wer = run_experiment(
                    audio_file_paths,
                    ground_truth_transcriptions,
                    model_type="whisper_jax",
                    device=device,
                    model_name=model,
                    dtype=dtype,
                    batch_size=batch_size,
                )

                config_key = (
                    f"Whisper JAX",
                    model,
                    device,
                    str(dtype),
                    str(batch_size),
                )
                execution_times_first[config_key] = execution_time_first
                execution_times_second[config_key] = execution_time_second
                wer_values[config_key] = wer

    plot_metrics(execution_times_first, execution_times_second, wer_values)


if __name__ == "__main__":
    audio_files, ground_truths = load_librispeech_dataset()
    if len(sys.argv) > 1:
        device = sys.argv[1]
        if device not in ["cpu", "gpu"]:
            raise ValueError("Invalid device type specified. Use 'cpu' or 'gpu'.")
    else:
        device = "cpu"

    # Set the platform name before importing JAX
    os.environ["JAX_PLATFORM_NAME"] = device

    import jax
    import jax.numpy as jnp

    #audio_files = ["Yann LeCun and Andrew Ng Why the 6-month AI Pause is a Bad Idea.wav"]
    #ground_truths = None
    main(audio_files, device, ground_truths)

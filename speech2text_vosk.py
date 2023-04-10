import os
from typing import List, Dict, Any, Union, Tuple
import argparse
import wave
import json
import re
import random
import glob
import time
import psutil
import gc
from vosk import KaldiRecognizer, Model
from jiwer import wer
import matplotlib.pyplot as plt


class WordVosk:
    """A class representing a word from the JSON format for Vosk speech recognition API."""

    def __init__(self, conf: float, start: float, end: float, word: str) -> None:
        """
        Initialize a Word object.

        Args:
            conf (float): Degree of confidence, from 0 to 1.
            start (float): Start time of pronouncing the word, in seconds.
            end (float): End time of pronouncing the word, in seconds.
            word (str): Recognized word.
        """
        self.conf = conf
        self.start = start
        self.end = end
        self.word = word

    def to_dict(self) -> Dict[str, Union[float, str]]:
        """Return a dictionary representation of the Word object."""
        return {
            "conf": self.conf,
            "start": self.start,
            "end": self.end,
            "word": self.word,
        }

    def to_string(self) -> str:
        """Return a string describing this instance."""
        return "{:20} from {:.2f} sec to {:.2f} sec, confidence is {:.2f}%".format(
            self.word, self.start, self.end, self.conf * 100
        )

    def to_json(self) -> str:
        """Return a JSON representation of the Word object."""
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)


class Transcription:
    def __init__(self, words: List[WordVosk]) -> None:
        self.words = words

    def to_dict(self) -> List[Dict[str, Union[float, str]]]:
        """Return a dictionary representation of the Transcription object."""
        return [word.to_dict() for word in self.words]

    def to_raw_text(self) -> str:
        """Generate raw transcription text from the list of WordVosk objects."""
        return " ".join(word.word for word in self.words)


class ModelSpeechToText:
    def __init__(self, audio_path: str, model_path: str) -> None:
        self.audio_path = audio_path
        self.wf = wave.open(self.audio_path)
        self.model = Model(model_path)

    def speech_to_text(self) -> List[Dict[str, Any]]:
        """Transcribe speech to text using the Vosk API."""
        rec = KaldiRecognizer(self.model, self.wf.getframerate())
        rec.SetWords(True)

        results = []
        frames_per_second = 44100
        i = 0
        print("Starting transcription process...")
        while True:
            data = self.wf.readframes(4000)
            i += 4000
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                part_result = json.loads(rec.Result())
                results.append(part_result)
            if i % (frames_per_second * 60) == 0:
                print(f"{i / frames_per_second / 60} minutes processed")
        part_result = json.loads(rec.FinalResult())
        results.append(part_result)
        self.wf.close()
        print("Transcription process completed.")
        return results

    @staticmethod
    def results_to_words(results: List[Dict[str, Any]]) -> List[WordVosk]:
        """Convert a list of Vosk API results to a list of words."""
        list_of_words = []
        for sentence in results:
            if len(sentence) == 1:
                continue
            for ind_res in sentence["result"]:
                word = WordVosk(
                    conf=ind_res["conf"],
                    start=ind_res["start"],
                    end=ind_res["end"],
                    word=ind_res["word"],
                )

                list_of_words.append(word)
        return list_of_words


def preprocess_timit(timit_root: str) -> List[Tuple[str, str]]:
    """Preprocesses TIMIT dataset.

    Args:
        timit_root: The path to the root directory of the TIMIT dataset.
        output_root: The path to the output directory for preprocessed files.

    Returns:
        A list of tuples containing paths to WAV files and their corresponding transcriptions.
    """
    dataset = []

    for audio_file in glob.glob(f"{timit_root}/**/*.wav", recursive=True):
        transcription_file = os.path.splitext(audio_file)[0] + ".txt"
        try:
            with open(transcription_file, "r") as f:
                transcription_line = f.readlines()[-1]
                transcription = re.sub(
                    r"[^a-zA-Z0-9\s']+", "", transcription_line.split(" ", 2)[-1]
                ).strip()
        except FileNotFoundError:
            print(f"Warning: Transcription file not found for {audio_file}. Skipping.")
            continue

        dataset.append((audio_file, transcription))

    return dataset


def prepare_evaluation_dataset(
    dataset: List[Tuple[str, str]], test_ratio: float = 1
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Split the dataset into training and evaluation datasets.

    Args:
        dataset: List of tuples containing the paths of the WAV files and their transcriptions.
        test_ratio: Ratio of the dataset to be used for evaluation (default: 0.2).

    Returns:
        Tuple containing the training and evaluation datasets.
    """

    random.shuffle(dataset)
    split_index = int(len(dataset) * (1 - test_ratio))
    return dataset[:split_index], dataset[split_index:]


def transcribe_audio_files(
    model: ModelSpeechToText, audio_files: List[str]
) -> List[str]:
    """Transcribe a list of audio files using a speech-to-text model.

    Args:
        model: ModelSpeechToText instance.
        audio_files: List of paths to the audio files.

    Returns:
        List of transcriptions corresponding to the input audio files.
    """
    transcriptions = []
    for audio_file in audio_files:
        model.audio_path = audio_file
        model.wf = wave.open(model.audio_path)
        results = model.speech_to_text()
        words = ModelSpeechToText.results_to_words(results)
        transcription = Transcription(words)
        transcriptions.append(transcription.to_raw_text())
    return transcriptions


def calculate_wer(
    reference_transcriptions: List[str], hypothesis_transcriptions: List[str]
) -> float:
    """Calculate the average Word Error Rate (WER) between reference and hypothesis transcriptions.

    Args:
        reference_transcriptions: List of reference transcriptions.
        hypothesis_transcriptions: List of hypothesis transcriptions.

    Returns:
        The average Word Error Rate (WER).
    """
    assert len(reference_transcriptions) == len(
        hypothesis_transcriptions
    ), "Reference and hypothesis lists should have the same length"
    total_wer = 0
    for ref, hyp in zip(reference_transcriptions, hypothesis_transcriptions):
        total_wer += wer(ref, hyp)
    return total_wer / len(reference_transcriptions)


def evaluate_models(
    models: List[ModelSpeechToText],
    evaluation_dataset: List[Tuple[str, str]],
) -> List[Tuple[float, float, float]]:
    """Evaluate multiple speech-to-text models using a given evaluation dataset.

    Args:
        models: A list of ModelSpeechToText instances.
        evaluation_dataset: A list of tuples containing the paths of the WAV files and their transcriptions.

    Returns:
        A list of tuples containing WER, execution time, and RAM usage for each model.
    """
    if not evaluation_dataset:
        print("The evaluation dataset is empty. Please check the dataset processing.")
        return []

    audio_files, reference_transcriptions = zip(*evaluation_dataset)

    metrics = []
    for model in models:
        start_time = time.time()

        hypothesis_transcriptions = transcribe_audio_files(model, audio_files)

        memory = psutil.Process().memory_info().rss
        elapsed_time = time.time() - start_time

        wer = calculate_wer(reference_transcriptions, hypothesis_transcriptions)

        metrics.append((wer, elapsed_time, memory / 1024 ** 3))

        del model
        gc.collect()

    return metrics


def run_evaluation(timit_root: str, model_paths: List[str]):
    """Evaluate speech-to-text models using the TIMIT dataset.

    Args:
        timit_root: Path to the TIMIT dataset root directory.
        model_paths: A list of paths to the model directories.
    """
    dataset = preprocess_timit(timit_root)

    train_dataset, evaluation_dataset = prepare_evaluation_dataset(dataset)

    models = [
        ModelSpeechToText("How Bill Gates reads books.wav", model_path)
        for model_path in model_paths
    ]

    metrics = evaluate_models(models, evaluation_dataset)

    return metrics


def plot_metrics(metrics_list, model_names, output_folder="plots"):
    """Plot WER, RAM usage, and execution time for each model.

    Args:
        metrics_list: A list of tuples containing WER, execution time, and RAM usage for each model.
        model_names: A list of model names corresponding to the metrics.
        output_folder: The folder where the plot will be saved.
    """
    wer_list, exec_time_list, ram_usage_list = zip(*metrics_list)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 5), sharex=True)
    fig.subplots_adjust(hspace=0.3)

    ax1.bar(model_names, wer_list)
    ax1.set_title("Word Error Rate (WER)")
    ax1.set_ylabel("WER")
    ax1.set_ylim(bottom=0, top=max(wer_list) * 1.1)

    ax2.bar(model_names, ram_usage_list)
    ax2.set_title("RAM Usage (MB)")
    ax2.set_ylabel("RAM Usage (MB)")
    ax2.set_ylim(bottom=0, top=max(ram_usage_list) * 1.1)

    ax3.bar(model_names, exec_time_list)
    ax3.set_title("Execution Time (s)")
    ax3.set_ylabel("Execution Time (s)")
    ax3.set_ylim(bottom=0, top=max(exec_time_list) * 1.1)

    for ax in (ax1, ax2, ax3):
        ax.tick_params(axis='x', rotation=45, labelsize=8)

    plt.tight_layout()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    plt.savefig(os.path.join(output_folder, "model_comparison.png"))
    plt.show()


def parse_model_name(model_path):
    """Parse the model name from the model path.

    Args:
        model_path: The path of the model directory.

    Returns:
        The parsed model name.
    """
    match = re.search(r"0.*", model_path)
    if match:
        return match.group()
    else:
        return "unknown_model"


def main(args: argparse.Namespace):
    audio_path = "How Bill Gates reads books.wav"
    vosk_model_paths = args.vosk_model_paths

    for model_path in vosk_model_paths:
        print(f"Loading model from {model_path}")
        model_speech_to_text = ModelSpeechToText(audio_path, model_path)
        results = model_speech_to_text.speech_to_text()
        words = ModelSpeechToText.results_to_words(results)

        transcription = Transcription(words)
        print(transcription.to_raw_text())

    timit_root = args.timit_root
    metrics = run_evaluation(timit_root, vosk_model_paths)

    model_names = [parse_model_name(model_path) for model_path in vosk_model_paths]
    plot_metrics(metrics, model_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speech-to-text transcription")
    parser.add_argument(
        "--timit_root",
        "-d",
        required=True,
        help="Path to the TIMIT dataset root directory",
    )
    parser.add_argument(
        "--vosk_model_paths",
        "-m",
        nargs="+",
        help="List of paths to Vosk model directories (e.g., 'vosk-model-en-us-aspire')",
    )

    args = parser.parse_args()
    main(args)

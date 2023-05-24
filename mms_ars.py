import os
from git import Repo
from dotenv import dotenv_values
import subprocess
from pydub import AudioSegment

config = dotenv_values(".env")


"""
Configuration Guide

To setup this script properly, you need to provide certain environment variables through a `.env` file. 
Below is a template of what it should contain:

CURRENT_DIR=/path/to/current/dir
AUDIO_SAMPLES_DIR=/path/to/audio_samples
FAIRSEQ_DIR=/path/to/fairseq
VIDEO_FILE=/path/to/video/file
AUDIO_FILE=/path/to/audio/file
RESAMPLED_AUDIO_FILE=/path/to/resampled/audio/file
TMPDIR=/path/to/tmp
PYTHONPATH=.
PREFIX=INFER
HYDRA_FULL_ERROR=1
USER=micro
MODEL=/path/to/fairseq/models_new/mms1b_all.pt  # Use full path here
LANG=eng

Additionally, you need to configure the file fairseq/examples/mms/asr/config/infer_common.yaml.

In the YAML file, use a full path for the checkpoint field like this:

checkpoint: /path/to/checkpoint/${env:USER}/${env:PREFIX}/${common_eval.results_path}

Without this change, you might encounter permission issues, unless you are running the application in a container.

If you are planning to use a CPU for computation, you also need to add the following to the YAML file as a top-level directive:

common:
  cpu: true
"""


def git_clone(url, path):
    """
    Clones a git repository

    Parameters:
    url (str): The URL of the git repository
    path (str): The local path where the git repository will be cloned
    """
    if not os.path.exists(path):
        Repo.clone_from(url, path)


def create_dirs(*dir_paths):
    """
    Creates directories

    Parameters:
    *dir_paths (str): Directory paths to be created
    """
    for dir_path in dir_paths:
        os.makedirs(dir_path, exist_ok=True)


def install_requirements(requirements):
    """
    Installs pip packages

    Parameters:
    requirements (list): List of packages to install
    """
    subprocess.check_call(["pip", "install"] + requirements)


def download_file(url, path):
    """
    Downloads a file

    Parameters:
    url (str): URL of the file to be downloaded
    path (str): The path where the file will be saved
    """
    subprocess.check_call(["wget", "-P", path, url])


def convert_video_to_audio(video_path, audio_path):
    """
    Converts a video file to an audio file

    Parameters:
    video_path (str): Path to the video file
    audio_path (str): Path to the output audio file
    """
    subprocess.check_call(["ffmpeg", "-i", video_path, "-ar", "16000", audio_path])


def run_inference(model, lang, audio):
    """
    Runs the MMS ASR inference

    Parameters:
    model (str): Path to the model file
    lang (str): Language of the audio file
    audio (str): Path to the audio file
    """
    subprocess.check_call(
        [
            "python",
            "examples/mms/asr/infer/mms_infer.py",
            "--model",
            model,
            "--lang",
            lang,
            "--audio",
            audio,
        ]
    )


def resample_audio(audio_path, new_audio_path, new_sample_rate):
    """
    Resamples an audio file

    Parameters:
    audio_path (str): Path to the current audio file
    new_audio_path (str): Path to the output audio file
    new_sample_rate (int): New sample rate in Hz
    """
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(new_sample_rate)
    audio.export(new_audio_path, format='wav')


if __name__ == "__main__":
    current_dir = config['CURRENT_DIR']
    tmp_dir = config['TMPDIR']
    fairseq_dir = config['FAIRSEQ_DIR']
    video_file = config['VIDEO_FILE']
    audio_file = config['AUDIO_FILE']
    audio_file_resampled = config['RESAMPLED_AUDIO_FILE']
    model_path = config['MODEL']
    lang = config['LANG']

    #git_clone('https://github.com/pytorch/fairseq', 'fairseq')

    #create_dirs(tmp_dir)

    #install_requirements(['--editable', './'])

    #download_file('https://dl.fbaipublicfiles.com/mms/asr/mms1b_all.pt', './models_new')

    # convert_video_to_audio(video_file, audio_file)
    #resample_audio(audio_file, audio_file_resampled, 16000)

    os.chdir(fairseq_dir)
    run_inference(model_path, lang, audio_file_resampled)

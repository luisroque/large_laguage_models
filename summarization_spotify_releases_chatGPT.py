import os
import json
import time
from typing import Dict, List, Any
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())


def get_new_releases(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Fetch new releases from Spotify.

    Args:
        limit (int, optional): Maximum number of album results to return. Defaults to 50.
        offset (int, optional): The index of the first result to return. Defaults to 0.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing album information.
    """
    new_releases = sp.new_releases(limit=limit, offset=offset)
    albums = new_releases["albums"]["items"]
    return albums


def get_album_tracks(album_id: str) -> List[Dict[str, Any]]:
    """
    Fetch tracks from a specific album.

    Args:
        album_id (str): The Spotify ID of the album.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing track information.
    """
    tracks = sp.album_tracks(album_id)["items"]
    return tracks


def save_data_to_file(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save data to a JSON file.

    Args:
        data (List[Dict[str, Any]]): List of dictionaries containing album and track information.
        file_path (str): Path to the JSON file where the data will be saved.
    """
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def load_data_from_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSON file.

    Args:
        file_path (str): Path to the JSON file where the data is stored.

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing album and track information.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def download_latest_albums_data() -> None:
    """
    Download the latest albums and tracks data from Spotify and save it to a JSON file.
    """
    limit = 50
    offset = 0
    total_albums = 20
    album_count = 0

    all_albums = []

    while total_albums is None or album_count < total_albums:
        new_releases = get_new_releases(limit, offset)
        if total_albums is None:
            total_albums = sp.new_releases()["albums"]["total"]

        for album in new_releases:
            album_info = {
                "album_name": album["name"],
                "artist_name": album["artists"][0]["name"],
                "album_type": album["album_type"],
                "release_date": album["release_date"],
                "tracks": [],
            }

            tracks = get_album_tracks(album["id"])

            for track in tracks:
                track_info = {
                    "track_name": track["name"],
                    "duration_ms": track["duration_ms"],
                }
                album_info["tracks"].append(track_info)

            all_albums.append(album_info)
            album_count += 1

        offset += limit
        time.sleep(1)  # Add a delay to avoid hitting the rate limit
        print(f"Downloaded {album_count}/{total_albums}")

    save_data_to_file(all_albums, "albums_and_tracks.json")


def preprocess_docs(data: List[Dict[str, Any]]) -> List[Document]:
    """
    Convert the JSON data to a list of Document objects.

    Args:
        data (List[Dict[str, Any]]): List of dictionaries containing album and track information.

    Returns:
        List[Document]: A list of Document objects containing the JSON data as strings, split into 3000-character segments.
    """
    json_string = json.dumps(data, ensure_ascii=False, indent=4)
    doc_splits = [json_string[i : i + 3500] for i in range(0, len(json_string), 3500)]
    docs = [Document(page_content=split_text) for split_text in doc_splits]
    return docs


def get_summary(docs: List[Document]) -> str:
    """
    Generate a summary using the JSON data provided in the list of Document objects.

    Args:
        docs (List[Document]): A list of Document objects containing the JSON data as strings.

    Returns:
        str: The generated summary.
    """
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    prompt_template = """Write a short summary about the latest songs in Spotify based on the JSON data below: \n\n{text}."""
    prompt_template2 = """Write an article about the latest music released in Spotify (below) and adress the change in music trends using the style of Rick Beato. : \n\n{text}"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    PROMPT2 = PromptTemplate(template=prompt_template2, input_variables=["text"])

    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        return_intermediate_steps=True,
        map_prompt=PROMPT,
        combine_prompt=PROMPT2,
        verbose=True,
    )

    res = chain({"input_documents": docs}, return_only_outputs=True)

    return res


if __name__ == "__main__":
    download_latest_albums_data()
    data = load_data_from_file("albums_and_tracks.json")
    docs = preprocess_docs(data)
    summary = get_summary(docs)
    print(summary)

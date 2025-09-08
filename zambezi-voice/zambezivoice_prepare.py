"""
Data preparation for BembaSpeech/zambezivoice

Author
------
 * Your Name, 2024
"""

import os
import csv
import functools
from dataclasses import dataclass
from speechbrain.dataio.dataio import save_pkl, load_pkl
from speechbrain.utils.logger import get_logger
from speechbrain.utils.parallel import parallel_map

logger = get_logger(__name__)
OPT_FILE = "opt_bembaspeech_prepare.pkl"
SAMPLERATE = 16000
LANGID = {'lozi':'loz'}

@dataclass
class BembaRow:
    audio: str
    wrd: str
    file_path: str
    duration: float

def prepare_zambezivoice(
    data_folder,
    save_folder,
    language,
    tr_csv="train.csv",
    dev_csv="dev.csv",
    te_csv="test.csv",
    skip_prep=False,
):
    """
    Prepares the CSV files for the BembaSpeech dataset.

    Arguments
    ---------
    data_folder : str
        Path to the folder containing the BembaSpeech data.
    save_folder : str
        The directory where to store the prepared CSV files.
    tr_csv : str
        The filename of the training CSV (default: "train.csv").
    dev_csv : str
        The filename of the development CSV (default: "dev.csv").
    te_csv : str
        The filename of the testing CSV (default: "test.csv").
    skip_prep : bool
        If True, skips data preparation if already completed.

    Returns
    -------
    None
    """

    # Skip preparation if already done
    if skip_prep and skip(data_folder, save_folder):
        logger.info("Skipping data preparation: already completed.")
        return

    # Ensure save folder exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Save options to detect changes in configuration
    save_opt = os.path.join(save_folder, OPT_FILE)
    save_pkl({"data_folder": data_folder}, save_opt)

    # Prepare each split
    for split, csv_name in zip(["train", "dev", "test"], [tr_csv, dev_csv, te_csv]):
        csv_path = os.path.join(data_folder, f"{language}/{LANGID[language]}/{split}.tsv")
        prepare_split(csv_path, save_folder, split)


def prepare_split(csv_path, save_folder, split_name):
    """
    Processes a single data split and creates a corresponding CSV file.

    Arguments
    ---------
    csv_path : str
        Path to the CSV file for the current split.
    save_folder : str
        Path to the folder where the prepared CSV file will be saved.
    split_name : str
        The name of the data split ("train", "dev", or "test").

    Returns
    -------
    None
    """

    # Output CSV path
    output_csv = os.path.join(save_folder, f"{split_name}.csv")

    # Skip if the file already exists
    if os.path.exists(output_csv):
        logger.info(f"{split_name}.csv already exists. Skipping creation.")
        return

    # Read input CSV and process rows
    logger.info(f"Processing {split_name} split...")
    with open(csv_path, "r", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file, delimiter="\t")
        rows = list(reader)

    csv_lines = [["ID", "duration", "wav", "wrd"]]
    line_processor = functools.partial(process_row, data_folder=os.path.dirname(csv_path))

    for idx, processed_row in enumerate(parallel_map(line_processor, rows)):
        csv_line = [
            f"{split_name}_{idx}",
            processed_row.duration,
            processed_row.file_path,
            processed_row.wrd,
        ]
        csv_lines.append(csv_line)

    # Write to output CSV
    with open(output_csv, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerows(csv_lines)

    logger.info(f"{split_name}.csv successfully created at {output_csv}.")


def process_row(row, data_folder) -> BembaRow:
    """
    Processes a single row from the input CSV.

    Arguments
    ---------
    row : dict
        A dictionary representing a row in the input CSV.
    data_folder : str
        Path to the folder containing audio files.

    Returns
    -------
    BembaRow
        The processed row with additional metadata.
    """

    file_path = os.path.join(data_folder, f"audio/{row['audio_id']}")

    # Compute duration (assume files are WAV at 16 kHz mono)
    audio_info = read_audio_info(file_path)
    duration = audio_info.num_frames / SAMPLERATE

    return BembaRow(
        audio=row["audio_id"],
        wrd=row["sentence"],
        file_path=file_path,
        duration=duration,
    )


def skip(data_folder, save_folder):
    """
    Determines if data preparation can be skipped.

    Arguments
    ---------
    data_folder : str
        Path to the folder containing the raw data.
    save_folder : str
        Path to the folder where prepared data is saved.

    Returns
    -------
    bool
        True if data preparation can be skipped, False otherwise.
    """
    save_opt = os.path.join(save_folder, OPT_FILE)
    if os.path.isfile(save_opt):
        opts_old = load_pkl(save_opt)
        if opts_old["data_folder"] == data_folder:
            return True
    return False


def read_audio_info(file_path):
    """
    Reads audio file metadata to extract the number of frames and sample rate.

    Arguments
    ---------
    file_path : str
        Path to the audio file.

    Returns
    -------
    AudioInfo
        Object containing audio file metadata (num_frames, sample_rate, etc.).
    """
    from scipy.io import wavfile

    rate, data = wavfile.read(file_path)
    return AudioInfo(num_frames=len(data), sample_rate=rate)


@dataclass
class AudioInfo:
    num_frames: int
    sample_rate: int

def get_vocab(csv_path, dest_path):
    vocab = set()

    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            wrd = row[-1].split()
            vocab.update(wrd)

    with open(dest_path, 'w') as f:
        f.write('\n'.join(vocab))


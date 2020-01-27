"""
Running basic pre-processing for the .srt subtitle files from
http://tvqa.cs.unc.edu/download_tvqa.html#tvqa-download-2.
"""
import re
import os
import pysrt
import glob
from tqdm import tqdm
from utils.basic_utils import save_jsonl


def convert_sub_time_to_seconds(sub_time):
    """sub_time is a SubRipTime object defined by pysrt"""
    return 60 * sub_time.minutes + sub_time.seconds + 0.001 * sub_time.milliseconds


def clean_single_sub_sentence(sub_sentence):
    """sub_sentence: str, """
    sub_sentence = sub_sentence.replace("\n", " ")
    sub_sentence = sub_sentence.replace("(", " ")
    sub_sentence = sub_sentence.replace(")", " ")
    sub_sentence = sub_sentence.replace(":", " : ")
    sub_sentence = re.sub(r"\s{2,}", " ", sub_sentence)
    return sub_sentence


def preprocess_subtitles_from_dir(srt_dir, save_path):
    """
    return: A python dict, the keys are the video names, the entries are lists,
            each contains all the text from a .srt file
    sub_times are the start time of the sentences.
    """
    assert not os.path.exists(save_path), "File {} already exists".format(save_path)

    print("Loading srt files from %s ..." % srt_dir)
    srt_paths = glob.glob(os.path.join(srt_dir, "*.srt"))
    srt_datalist = []
    for sub_path in tqdm(srt_paths, desc="Loop over subtitle files"):
        subs = pysrt.open(sub_path, encoding="iso-8859-1")
        if len(subs) == 0:
            subs = pysrt.open(sub_path)

        sub_data = []
        for cur_sub in subs:
            sub_data.append(dict(
                text=clean_single_sub_sentence(cur_sub.text),
                start=convert_sub_time_to_seconds(cur_sub.start),
                end=convert_sub_time_to_seconds(cur_sub.end)
            ))

        srt_datalist.append(dict(
            vid_name=os.path.splitext(os.path.basename(sub_path))[0],
            sub=sub_data
        ))
    save_jsonl(srt_datalist, save_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-srt_dir", type=str,
                        help="path to the dir containing all the TVQA subtitle .srt files")
    parser.add_argument("-save_path", type=str, help="path to save the preprocessed subtitles")
    args = parser.parse_args()

    preprocess_subtitles_from_dir(args.srt_dir, args.save_path)

from utils.basic_utils import load_json, save_json


def combine(video_name_split_path, video_duration_path, save_path):
    video_name_split = load_json(video_name_split_path)
    video_duration_dict = load_json(video_duration_path)

    combined_dict = {}
    for split_name, split_video_names in video_name_split.items():
        combined_dict[split_name] = {vid_name: video_duration_dict[vid_name]
                                     for vid_name in split_video_names}
    save_json(combined_dict, save_path)


if __name__ == '__main__':
    import sys
    combine(*sys.argv[1:])


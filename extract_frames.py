from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import json
import os


def add_label(video, video_id, frame_id):
    # load labels
    with open('./ffr_orig/11-dataset_img_ffr_ifr_label_15fps_RenameFolder_datiClinici.json', 'r') as f:
        data = json.load(f)
    label_map = {item['image']: item['label'] for item in data}
    label = label_map.get("/".join(video.split("/")[-3:]))
    if label is not None:
        label = int(label)
        labels_path = "FFR_256/dataset.json"
        if os.path.isfile(labels_path):
            with open(labels_path, "r") as f:
                labels = json.load(f)
        else:
            labels = {"labels": []}

        # Add current frame's label to list
        label_entry = [f"video{video_id}/{frame_id}.png", label]
        labels["labels"].append(label_entry)

        # Write updated labels to file
        with open(labels_path, "w") as f:
            json.dump(labels, f)
        return label
    else:
        return None


def extract_frames():
    # get videos
    all_videos = glob('ffr_orig/FFAR_data_15/**/*.npy', recursive=True)
    print(len(all_videos))

    # output
    outdir = 'FFR_256'
    os.makedirs(outdir, exist_ok=True)

    # extract frames and add labels
    for video_id, video in tqdm(enumerate(all_videos), total=len(all_videos)):
        frames = np.load(video)
        subfolder = os.path.join(outdir, 'video' + str(video_id))
        os.makedirs(subfolder, exist_ok=True)
        for frame_id, frame in enumerate(frames):
            label = add_label(video, video_id, frame_id)
            if label is not None:
                frame = frame.astype(np.uint8)
                outpath = os.path.join(subfolder, f'{frame_id}.png')
                im = Image.fromarray(frame, mode='L')
                im.save(outpath)

    print('Done')


if __name__ == '__main__':
    extract_frames()

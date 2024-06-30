from pathlib import Path


IMG_PATH = Path("img")
DATA_PATH = Path("data")


def img_output_path(lesson_name: str):
    return IMG_PATH / lesson_name

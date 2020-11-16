import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from cv2.ximgproc import guidedFilter

# Streamlit sharing is CPU only
device = torch.device('cpu')

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",  # Can be "auto", "expanded", "collapsed"
    page_title="SkyAR adapted by metasemantic",
)

st.sidebar.title("skyAR")

args = {
    "f_skybox" : "sample_images/photo-1439246854758-f686a415d9da.webp",
    "f_landscape" : "sample_images/christopher-zarriello-Pq3AM1OV0fM-unsplash.jpg",
    "in_size_w": 384,
    "in_size_h": 384,
    "out_size_w": 845,
    "out_size_h": 480,
    "skybox_center_crop": 1.0,
    "auto_light_matching": False,
    "relighting_factor": 0.5,
    "recoloring_factor": 0.5,
    "halo_effect": False,
}

mods = {}

slider_cols = [
    "recoloring_factor",
    "relighting_factor",
]
for key in slider_cols:
    mods[key] = st.sidebar.slider(key, 0.0, 1.0, args[key])

# Disable this for now
key = "skybox_center_crop"
#mods[key] = st.sidebar.slider(key, 1.0, 1.2, args[key])
mods[key] = args[key]

bool_cols = ["auto_light_matching", "halo_effect"]
for key in bool_cols:
    mods[key] = st.sidebar.checkbox(key, args[key])


# Hosted on my personal account until I figure something else out
cloud_model_location = "1PmsUezmJGwTQP51yTMsjLGe2okdo-yxr"

@st.cache
def load_model():

    save_dest = Path('model')
    save_dest.mkdir(exist_ok=True)
    
    f_checkpoint = Path("model/skyAR_coord_resnet50.pt")

    if not f_checkpoint.exists():
        with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
            from GD_download import download_file_from_google_drive
            download_file_from_google_drive(cloud_model_location, f_checkpoint)
    
    model = torch.load(f_checkpoint, map_location=device)
    model.eval()
    return model


def tile_skybox_img(img):
    h, w, c = img.shape

    cc = mods["skybox_center_crop"]

    img = cv2.resize(img, (int(cc * args["out_size_w"]), int(cc * args["out_size_h"])),)

    screen_y1 = int(img.shape[0] / 2 - args["out_size_h"] / 2)
    screen_x1 = int(img.shape[1] / 2 - args["out_size_w"] / 2)
    img = np.concatenate([img[screen_y1:, :, :], img[0:screen_y1, :, :]], axis=0)
    img = np.concatenate([img[:, screen_x1:, :], img[:, 0:screen_x1, :]], axis=1)

    dh = img.shape[0] - h
    dw = img.shape[1] - w

    oh = args["out_size_h"]
    ow = args["out_size_w"]

    img = img[dh // 2 : dh // 2 + oh, dw // 2 : dw // 2 + ow, :].squeeze()

    return img


@st.cache(ttl=3600, max_entries=10)
def load_skybox_image(f_img):

    if isinstance(f_img, str):
        img = cv2.imread(f_img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = np.array(Image.open(f_img))

    img = np.array(img / 255.0, dtype=np.float32)
    img = cv2.resize(img, (args["out_size_w"], args["out_size_h"]))

    imgx = tile_skybox_img(img)

    return imgx


@st.cache(ttl=3600, max_entries=10)
def load_output_image(f_img):

    if isinstance(f_img, str):
        img = cv2.imread(f_img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = np.array(Image.open(f_img))

    img = np.array(img / 255.0, dtype=np.float32)
    img = cv2.resize(img, (args["out_size_w"], args["out_size_h"]))

    return img


@st.cache(ttl=3600, max_entries=10)
def compute_skymask(img):
    h, w, c = img.shape
    imgx = cv2.resize(img, (args["in_size_w"], args["in_size_h"]))
    imgx = np.array(imgx, dtype=np.float32)
    imgx = torch.tensor(imgx).permute([2, 0, 1]).unsqueeze(0)

    with torch.no_grad():
        pred = model(imgx.to(device))
        pred = torch.nn.functional.interpolate(
            pred, (h, w), mode="bicubic", align_corners=False
        )
        pred = pred[0, :].permute([1, 2, 0])
        pred = torch.cat([pred, pred, pred], dim=-1)
        pred = np.array(pred.detach().cpu())
        pred = np.clip(pred, a_max=1.0, a_min=0.0)

    r, eps = 20, 0.01
    refined_skymask = guidedFilter(img[:, :, 2], pred[:, :, 0], r, eps)

    refined_skymask = np.stack(
        [refined_skymask, refined_skymask, refined_skymask], axis=-1
    )
    refined_skymask = np.clip(refined_skymask, a_min=0, a_max=1)

    return refined_skymask


@st.cache(ttl=3600, max_entries=10)
def relighting(img, skybg, skymask):

    # color matching, reference: skybox_img
    step = int(img.shape[0] / 20)
    skybg_thumb = skybg[::step, ::step, :]
    img_thumb = img[::step, ::step, :]
    skymask_thumb = skymask[::step, ::step, :]
    skybg_mean = np.mean(skybg_thumb, axis=(0, 1), keepdims=True)
    img_mean = np.sum(img_thumb * (1 - skymask_thumb), axis=(0, 1), keepdims=True) / (
        (1 - skymask_thumb).sum(axis=(0, 1), keepdims=True) + 1e-9
    )
    diff = skybg_mean - img_mean
    img_colortune = img + mods["recoloring_factor"] * diff

    if mods["auto_light_matching"]:
        img = img_colortune
    else:
        # keep foreground ambient_light and maunally adjust lighting
        img = mods["relighting_factor"] * (
            img_colortune + (img.mean() - img_colortune.mean())
        )

    img = np.clip(img, 0, 1)

    return img


@st.cache(ttl=3600, max_entries=10)
def halo(syneth, skybg, skymask):

    # reflection
    halo = 0.5 * cv2.blur(
        skybg * skymask, (int(args["out_size_w"] / 5), int(args["out_size_w"] / 5)),
    )
    # screen blend 1 - (1-a)(1-b)
    syneth_with_halo = 1 - (1 - syneth) * (1 - halo)

    return syneth_with_halo


mod_sky_img = st.sidebar.file_uploader("Upload Sky Image 🌧️")

f_skybox = args["f_skybox"] if mod_sky_img is None else mod_sky_img
img_sky = load_skybox_image(f_skybox)


mod_landscape_img = st.sidebar.file_uploader("Upload Landscape Image 🏙️")
f_input_image = (
    args["f_landscape"] if mod_landscape_img is None else mod_landscape_img)

img_in = load_output_image(f_input_image)


model = load_model()
mask = compute_skymask(img_in)

# Ignore optical flow for a single image
img_light = relighting(img_in, img_sky, mask)

col1, col2, col3 = st.beta_columns(3)
col1.write("Landscape")
col1.image(img_in, use_column_width=True)
col2.write("Sky")
col2.image(img_sky, use_column_width=True)
col3.write("Mask")
col3.image(mask, use_column_width=True)


synth = img_light * (1 - mask) + img_sky * mask

if mods["halo_effect"]:
    synth = halo(synth, img_sky, mask)


col1, col2 = st.beta_columns([2,1])
col1.image(synth, use_column_width=True)
col2.markdown('''Modified demo of [SkyAR](https://github.com/jiupinjia/SkyAR). Streamlit demo built with :blue_heart: by [@metasemantic](https://twitter.com/metasemantic), source code on [github](https://github.com/thoppe/streamlit-skyAR).

Photos by [Christopher Zarriello](https://unsplash.com/@chris_zarriello?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText) and [Sérgio Rola](https://unsplash.com/@sergio_rola?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText).

Go to [unsplash](https://unsplash.com/s/photos/sky) for more photos!
''')


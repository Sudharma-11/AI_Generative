## Install Requirements

# Commented out IPython magic to ensure Python compatibility.
!wget -q https://github.com/ShivamShrirao/diffusers/raw/main/examples/dreambooth/train_dreambooth.py
!wget -q https://github.com/ShivamShrirao/diffusers/raw/main/scripts/convert_diffusers_to_original_stable_diffusion.py
# %pip install -qq git+https://github.com/ShivamShrirao/diffusers
# %pip install -q -U --pre triton
# %pip install -q accelerate transformers ftfy bitsandbytes==0.35.0 gradio natsort safetensors xformers

!apt-get install wget
!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip ngrok-stable-linux-amd64.zip

!mkdir -p ~/.huggingface
HUGGINGFACE_TOKEN = "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"
!echo -n "{HUGGINGFACE_TOKEN}" > ~/.huggingface/token

## Settings and run

save_to_gdrive = True 
if save_to_gdrive:
    from google.colab import drive
    drive.mount('/content/drive')
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "stable_diffusion_weights/cxyz" 
if save_to_gdrive:
    OUTPUT_DIR = "/content/drive/MyDrive/" + OUTPUT_DIR
else:
    OUTPUT_DIR = "/content/" + OUTPUT_DIR

print(f"[*] Weights will be saved at {OUTPUT_DIR}")

!mkdir -p $OUTPUT_DIR

# Start Training
concepts_list = [
    {
        "instance_prompt": "cxyz",
        "class_prompt": "human image",
        "instance_data_dir": "/content/drive/MyDrive/myimage",
        "class_data_dir": "/content/drive/MyDrive/human pics"  # `class_data_dir` contains regularization images
    }
 ]
import json
import os
for c in concepts_list:
    os.makedirs(c["instance_data_dir"], exist_ok=True)

with open("concepts_list.json", "w") as f:
    json.dump(concepts_list, f, indent=4)

import os
from google.colab import files
import shutil

for c in concepts_list:
    print(f"Uploading instance images for `{c['instance_prompt']}`")
    uploaded = files.upload()
    for filename in uploaded.keys():
        dst_path = os.path.join(c['instance_data_dir'], filename)
        shutil.move(filename, dst_path)

!accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --output_dir=$OUTPUT_DIR \
  --revision="fp16" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --seed=1337 \
  --resolution=512 \
  --train_batch_size=1 \
  --train_text_encoder \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=168 \
  --num_class_images=252 \
  --sample_batch_size=4 \
  --max_train_steps=1680 \
  --save_interval=10000 \ # Reduce the `--save_interval` to lower than `--max_train_steps` to save weights from intermediate steps.
  --save_sample_prompt="cxyz" \ # `--save_sample_prompt` can be same as `--instance_prompt` to generate intermediate samples (saved along with weights in samples directory).
  --concepts_list="concepts_list.json"

WEIGHTS_DIR = "" 
if WEIGHTS_DIR == "":
    from natsort import natsorted
    from glob import glob
    import os
    WEIGHTS_DIR = natsorted(glob(OUTPUT_DIR + os.sep + "*"))[-1]
print(f"[*] WEIGHTS_DIR={WEIGHTS_DIR}")

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

weights_folder = OUTPUT_DIR
folders = sorted([f for f in os.listdir(weights_folder) if f != "0"], key=lambda x: int(x))

row = len(folders)
col = len(os.listdir(os.path.join(weights_folder, folders[0], "samples")))
scale = 4
fig, axes = plt.subplots(row, col, figsize=(col*scale, row*scale), gridspec_kw={'hspace': 0, 'wspace': 0})

for i, folder in enumerate(folders):
    folder_path = os.path.join(weights_folder, folder)
    image_folder = os.path.join(folder_path, "samples")
    images = [f for f in os.listdir(image_folder)]
    for j, image in enumerate(images):
        if row == 1:
            currAxes = axes[j]
        else:
            currAxes = axes[i, j]
        if i == 0:
            currAxes.set_title(f"Image {j}")
        if j == 0:
            currAxes.text(-0.1, 0.5, folder, rotation=0, va='center', ha='center', transform=currAxes.transAxes)
        image_path = os.path.join(image_folder, image)
        img = mpimg.imread(image_path)
        currAxes.imshow(img, cmap='gray')
        currAxes.axis('off')
        
plt.tight_layout()
plt.savefig('grid.png', dpi=72)

!./ngrok authtoken 2N8CgXpCOPUYLXNVNIy6KNCV5nD_3Z735wg5PANop9cKoxyYN

"""## Convert weights to ckpt to use in web UIs like AUTOMATIC1111.

## Inference
"""

import torch
!pip install flask_ngrok
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
from IPython.display import display
from flask import Flask

app = Flask(__name__)
model_path = WEIGHTS_DIR  # If you want to use previously trained model saved in gdrive, replace this with the full path of model in gdrive     
pipe = StableDiffusionPipeline.from_pretrained(model_path, safety_checker=None, torch_dtype=torch.float16).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
g_cuda = None

g_cuda = torch.Generator(device='cuda')
seed = 52362 
g_cuda.manual_seed(seed)

from flask import Flask
from flask import request
from flask import Response
from flask_ngrok import run_with_ngrok
import os
import json



app = Flask(__name__)
run_with_ngrok(app)

if not os.path.exists('images'):
    os.makedirs('images')

@app.route("/test") # for testing the api 
def test():
  return "Test is successful"

@app.route("/avatar-face")
def generateImage():
    prompt = request.form['prompt']
    negative_prompt = request.form['negative_prompt']
    num_samples = request.form['num_samples']
    guidance_scale = 7.9
    num_inference_steps = 50 
    height = 512
    width = 512 
    images_list = []
    with autocast("cuda"), torch.inference_mode():
      images = pipe(
          prompt,
          height=height,
          width=width,
          negative_prompt=negative_prompt,
          num_images_per_prompt=num_samples,
          num_inference_steps=num_inference_steps,
          guidance_scale=guidance_scale,
          generator=g_cuda
      ).images

    for img in images:
        os.rename(os.getcwd()+"/images",img)
        images_list.append("http://eb12-34-141-227-3.ngrok.io/"+os.getcwd()+"/images"+img)
        #display(img)
    return Response(json.dumps(images_list),  mimetype='application/json')
app.run()

#@title Free runtime memory
exit()

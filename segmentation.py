import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import SamModel, SamProcessor
import torch

class MedSamSegmentation:
    def __init__(self, model_name="flaviagiammarino/medsam-vit-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SamModel.from_pretrained(model_name).to(self.device)
        self.processor = SamProcessor.from_pretrained(model_name)

    def process_image(self, image_path, input_boxes):
        raw_image = Image.open(image_path).convert("RGB")
        inputs = self.processor(raw_image, input_boxes=[[input_boxes]], return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, multimask_output=False)
        probs = self.processor.image_processor.post_process_masks(outputs.pred_masks.sigmoid().cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu(), binarize=False)
        return raw_image, probs[0]

    @staticmethod
    def show_mask(mask, ax, random_color):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([251/255, 252/255, 30/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    @staticmethod
    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2))

    def visualize_segmentation(self, image_path, input_boxes, random_color=False):
        raw_image, mask = self.process_image(image_path, input_boxes)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(np.array(raw_image))
        self.show_box(input_boxes, ax[0])
        ax[0].set_title("Input Image and Bounding Box")
        ax[0].axis("off")
        ax[1].imshow(np.array(raw_image))
        self.show_mask(mask=mask > 0.5, ax=ax[1], random_color=random_color)
        self.show_box(input_boxes, ax[1])
        ax[1].set_title("MedSAM Segmentation")
        ax[1].axis("off")
        plt.show()

        plt.savefig("image2.jpg")
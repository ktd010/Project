import os
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

import torch
from diffusers import StableDiffusionPipeline

load_dotenv()

# Define prompt2 outside of the class
prompt2 = "input medical report"

class OpenAIChat:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.prompt3 = None
        self.prompt4 = None
        self.prompt5 = None
        self.prompt6 = None

    async def generate_completion(self, prompt):
        chat_completion = await self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-3.5-turbo",
        )
        completion_content = chat_completion.choices[0].message.content
        return completion_content

    async def main(self):
        # First prompt using prompt2
        self.prompt3 = await self.generate_completion("process this report " + str(prompt2) + "  and just write the medical condition. nothing else")

        # Second prompt
        self.prompt4 = await self.generate_completion("process this report " + str(prompt2) + "  and just write the location of medical  issue . nothing else")

        self.prompt5 = await self.generate_completion("process this report " + str(prompt2) + "  and identify the location of the medical condition. nothing else")

        self.prompt6 = await self.generate_completion("take this location " + str(prompt5) + "  locate it by a bounding box in a 256 by 256 image.with 20 by 20 square. bounding box shoul be given as [x_min, y_min,x_max,y_max format] . nothing else")

        self.prompt7 = await self.generate_completion("process this report " + str(prompt2) + "  and write a brief description about medical condition only. explaining in simple words")



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

if __name__ == "__main__":
    openai_chat = OpenAIChat()
    asyncio.run(openai_chat.main())

    # Now, prompt3 and prompt4 are accessible globally with the updated content
    print(openai_chat.prompt3)
    print(openai_chat.prompt4)
    print(openai_chat.prompt5)
    print(openai_chat.prompt6)

model_id = "Nihirc/Prompt2MedImage"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

medical_condition_andlocation = openai_chat.prompt4
input_boxes = prompt6
medical_condition_description = openai_chat.prompt4



image = pipe(medical_condition_andlocation).images[0]

    
image.save("image1.png")

input_boxes = prompt6

medsam = MedSamSegmentation()
image = "image1.png"
medsam.visualize_segmentation(image, input_boxes)
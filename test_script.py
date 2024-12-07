from transformers import AutoModelForCausalLM, AutoProcessor
from pathlib import Path
import torch

model = AutoModelForCausalLM.from_pretrained("microsoft/maira-2", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("microsoft/maira-2", trust_remote_code=True)

device = torch.device("cuda")
model = model.eval()
model = model.to(device)


import requests
from PIL import Image

def get_sample_data() -> dict[str, Image.Image | str]:
    """
    Download chest X-rays from IU-Xray, which we didn't train MAIRA-2 on. License is CC.
    We modified this function from the Rad-DINO repository on Huggingface.
    """
    frontal_image_url = "https://openi.nlm.nih.gov/imgs/512/145/145/CXR145_IM-0290-1001.png"
    lateral_image_url = "https://openi.nlm.nih.gov/imgs/512/145/145/CXR145_IM-0290-2001.png"

    def download_and_open(url: str) -> Image.Image:
        response = requests.get(url, headers={"User-Agent": "MAIRA-2"}, stream=True)
        return Image.open(response.raw)

    frontal_image = download_and_open(frontal_image_url)
    lateral_image = download_and_open(lateral_image_url)

    sample_data = {
        "frontal": frontal_image,
        "lateral": lateral_image,
        "indication": "Dyspnea.",
        "comparison": "None.",
        "technique": "PA and lateral views of the chest.",
        "phrase": "Pleural effusion."  # For the phrase grounding example. This patient has pleural effusion.
    }
    return sample_data

sample_data = get_sample_data()

processed_inputs = processor.format_and_preprocess_reporting_input(
    current_frontal=sample_data["frontal"],
    current_lateral=sample_data["lateral"],
    prior_frontal=None,  # Our example has no prior
    indication=sample_data["indication"],
    technique=sample_data["technique"],
    comparison=sample_data["comparison"],
    prior_report=None,  # Our example has no prior
    return_tensors="pt",
    get_grounding=False,  # For this example we generate a non-grounded report
)

print(f'sample images types: {type(sample_data["frontal"])}, {type(sample_data["lateral"])}')
print(f'all info about images type and size and color: {sample_data["frontal"]}, {sample_data["lateral"]}, {sample_data["frontal"].size}, {sample_data["frontal"].mode}, {sample_data["lateral"].size}, {sample_data["lateral"].mode}')
print("Processed inputs:", processed_inputs)
print("Processed inputs keys:", processed_inputs.keys())
print(f'decoded: {processor.decode(processed_inputs["input_ids"][0])}')
print(f'pixel values info and type: {processed_inputs["pixel_values"].shape}, {processed_inputs["pixel_values"].dtype}')
print(f'pixel values: {processed_inputs["pixel_values"]}')

processed_inputs = processed_inputs.to(device)
with torch.no_grad():
    output_decoding = model.generate(
        **processed_inputs,
        max_new_tokens=300,  # Set to 450 for grounded reporting
        use_cache=True,
    )
prompt_length = processed_inputs["input_ids"].shape[-1]
decoded_text = processor.decode(output_decoding[0][prompt_length:], skip_special_tokens=True)
decoded_text = decoded_text.lstrip()  # Findings generation completions have a single leading space
prediction = processor.convert_output_to_plaintext_or_grounded_sequence(decoded_text)
print("Parsed prediction:", prediction)


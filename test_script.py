from transformers import AutoModelForCausalLM, AutoProcessor
from pathlib import Path
import torch
from peft import PeftModel
import torch
import requests
from PIL import Image
from processing_maira2 import Maira2Processor



base_model_name = "microsoft/maira-2"
adapter_model_name = "StanfordAIMI/maira2-srrg-findings2"

model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)
processor = Maira2Processor.from_pretrained(base_model_name, trust_remote_code=True)

model = PeftModel.from_pretrained(model, adapter_model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.eval()
model = model.to(device)



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
        "phrase": "Pleural effusion." 
    }
    return sample_data

sample_data = get_sample_data()

processed_inputs = processor.format_and_preprocess_reporting_input(
    current_frontal=sample_data["frontal"],
    current_lateral=sample_data["lateral"],
    prior_frontal=None, 
    indication=sample_data["indication"],
    technique=sample_data["technique"],
    comparison=sample_data["comparison"],
    prior_report=None, 
    return_tensors="pt",
    get_grounding=False, 
)

print("Processed inputs:", processor.decode(processed_inputs["input_ids"][0], skip_special_tokens=True))
processed_inputs = processed_inputs.to(device)
with torch.no_grad():
    output_decoding = model.generate(
        **processed_inputs,
        max_new_tokens=300,
        use_cache=True,
    )
prompt_length = processed_inputs["input_ids"].shape[-1]
decoded_text = processor.decode(output_decoding[0][prompt_length:], skip_special_tokens=True)
decoded_text = decoded_text.lstrip() 
prediction = processor.convert_output_to_plaintext_or_grounded_sequence(decoded_text)
print("Parsed prediction:", prediction)


import json
from pathlib import Path
import os

import soundfile as sf
import torch
import torchaudio
import hydra
import random

from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from omegaconf import OmegaConf
from safetensors.torch import load_file
from tqdm import tqdm
from accel_hydra.models.common import LoadPretrainedBase

from utils.config import register_omegaconf_resolvers
from utils.vocoder import load_vocoder
from utils.audio import MelSpec

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except:
    pass

register_omegaconf_resolvers()



def estimate_duration_range(text):
    words = text.strip().split()
    num_words = len(words)
    min_duration = num_words / 4.0
    max_duration = num_words / 1.5
    ref_min = num_words / 3.0
    ref_max = num_words / 1.5
    return min_duration, max_duration, ref_min, ref_max

def get_duration(text, predicted_duration):
    min_dur, max_dur, ref_min, ref_max = estimate_duration_range(text)
    if predicted_duration < min_dur or predicted_duration > max_dur:
        return round(random.uniform(ref_min, ref_max), 2)
    return predicted_duration

duration_model_id = "OpenSound/CapSpeech-models"
duration_tokenizer = AutoTokenizer.from_pretrained(os.path.join(duration_model_id, "nar_duration_predictor"))
duration_model = AutoModelForSequenceClassification.from_pretrained(os.path.join(duration_model_id, "nar_duration_predictor"))
duration_model.eval()

def get_caption_duration(caption, transcript):
    duration_inputs = caption + " <NEW_SEP> " + transcript
    duration_inputs = duration_tokenizer(duration_inputs, return_tensors="pt", padding="max_length", truncation=True, max_length=400)
    duration_outputs = duration_model(**duration_inputs)
    predicted_duration = duration_outputs.logits.squeeze().item()
    duration = get_duration(transcript, predicted_duration)
    return duration

def get_prompt_duration(ref_duration, ref_text, gen_text):
    ref_text_len = len(ref_text.encode("utf-8"))
    gen_text_len = len(gen_text.encode("utf-8"))
    duration = round(ref_duration / ref_text_len * gen_text_len, 3)
    return duration

def process_sample(
    sample, config, exp_config, accelerator, model, unwrapped_model, target_sample_rate, prompt_sample_rate, infer_type, output_dir
):
    
    if infer_type == "prompt":
        prompt_audio_path = sample["prompt_audio_path"]
        caption = ""
    elif infer_type == "caption":
        prompt_audio_path = sample["prompt_audio_path"]
        location_path = sample["location"]
        caption = sample["caption"]
    else: raise ValueError(f"Unknown infer_type: {infer_type}")

    gen_text = sample.get("target_text", sample.get("text", ""))
    id_ = sample.get("id", None)
    if not id_:
        id_ = os.path.splitext(os.path.basename(location_path))[0]

    prompt_waveform, ref_sr = torchaudio.load(prompt_audio_path)  # [C, T]
    prompt_waveform = prompt_waveform.to(accelerator.device)
    if prompt_waveform.dim() == 2:
        prompt_waveform = prompt_waveform.mean(0, keepdim=True)
    prompt_waveform_lengths = torch.tensor([prompt_waveform.shape[-1]], device=accelerator.device)
    assert prompt_waveform_lengths[0] > 5000, f"Empty waveform: {prompt_audio_path}"

    if ref_sr != prompt_sample_rate:
        prompt_waveform = torchaudio.functional.resample(prompt_waveform, ref_sr, prompt_sample_rate)
        prompt_waveform_lengths = torch.tensor([prompt_waveform.shape[-1]], device=accelerator.device)

    text_list = [gen_text]
    caption_list = [caption]
    #target_duration = sample.get("target_duration")

    if infer_type == "prompt":
        prompt_duration = sample.get("prompt_duration")
        prompt_text = sample.get("prompt_text")
        target_duration = get_prompt_duration(prompt_duration, prompt_text, gen_text)
    elif infer_type == "caption":
        target_duration = get_caption_duration(caption, gen_text)
    duration = int(target_duration * target_sample_rate / 256)
    
    duration = torch.tensor([duration], device=accelerator.device, dtype=torch.long)
    with torch.no_grad():
        if hasattr(unwrapped_model, 'inference') and callable(getattr(unwrapped_model, 'inference')):
            out_wav = unwrapped_model.inference(
                prompt_waveform=prompt_waveform,
                text=text_list,
                caption=caption_list,
                infer_type=0, 
                duration=duration,
                prompt_waveform_lengths=prompt_waveform_lengths,
                **config.get("infer_args", {}),
            )
        else:
            raise AttributeError(
                f"Model {type(unwrapped_model).__name__} does not have the new 'inference' method"
            )

        waveform = out_wav.squeeze().cpu()
        out_path = output_dir / f"{id_}.wav"
        torchaudio.save(out_path, waveform.unsqueeze(0), target_sample_rate)

def main():
    accelerator = Accelerator(mixed_precision="no")
    configs = []

    @hydra.main(config_path="configs", config_name="inference")
    def parse_config_from_command_line(config):
        config = OmegaConf.to_container(config, resolve=True)
        configs.append(config)

    parse_config_from_command_line()
    config = configs[0]

    exp_dir, ckpt_dir = None, None
    if os.path.isdir(config["exp_dir"]):
        exp_dir = Path(config["exp_dir"])
    if os.path.isdir(config["ckpt_dir_or_file"]):
        ckpt_dir = Path(config["ckpt_dir_or_file"])
        ckpt_path = ckpt_dir / "model.safetensors"
    elif os.path.isfile(config["ckpt_dir_or_file"]):
        ckpt_path = config["ckpt_dir_or_file"]
        ckpt_dir = Path(ckpt_path).parent

    if ckpt_dir is None and exp_dir is None:
        if not os.path.exists(config["ckpt_dir_or_file"]):
            raise ValueError(
                f"ckpt_dir {config['ckpt_dir_or_file']} does not exist."
            )
        raise ValueError(
            "Either exp_dir or ckpt_dir_or_file should be provided."
        )

    if ckpt_dir is None:
        use_best = config.get("use_best", True)
        if use_best:  # use best ckpt
            ckpt_path: Path = exp_dir / "checkpoints/best/model.safetensors"
        else:  # use last ckpt
            ckpt_path: Path = sorted((exp_dir / "checkpoints").iterdir())[-1] / "model.safetensors"
    if exp_dir is None:
        exp_dir = ckpt_dir.parent.parent

    if "exp_config_path" in config:
        exp_config_path = config['exp_config_path']
        exp_config = OmegaConf.load(config['exp_config_path'])
    else:
        exp_config_path = exp_dir / "config.yaml"
    exp_config = OmegaConf.load(exp_config_path)

    accelerator.print(
        f'\n ckpt path: {ckpt_path}, model config path: {exp_config_path}\n '
    )


    model: LoadPretrainedBase = hydra.utils.instantiate(
        exp_config["model"], _convert_="all"
    )
    state_dict = load_file(ckpt_path)
    model.load_pretrained(state_dict)
    model.eval()
    model = accelerator.prepare(model)
    unwrapped_model = accelerator.unwrap_model(model)
    target_sample_rate = exp_config["data_dict"]["train_dataset"]["target_sample_rate"]
    prompt_sample_rate = exp_config["data_dict"]["train_dataset"]["prompt_sample_rate"]

    test_jsonl = config.get("test_jsonl", None)
    if not test_jsonl or not os.path.exists(test_jsonl):
        raise FileNotFoundError(f"Cannot find input jsonl file. Please specify test_jsonl in config. Current: {test_jsonl}")
    with open(test_jsonl, 'r', encoding='utf-8') as fin:
        samples = [json.loads(line) for line in fin if line.strip()]
    accelerator.print(f"Loaded {len(samples)} samples from {test_jsonl}")

    if config["wav_dir_root"] is not None:
        wav_dir_root = Path(config["wav_dir_root"])
    else:
        wav_dir_root = exp_dir
    output_dir = wav_dir_root / config["wav_dir"]
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    infer_type = config["infer_type"]
    
    for sample in tqdm(samples, desc="Batch Inference"):
        process_sample(
            sample,
            config,
            exp_config,
            accelerator,
            model,
            unwrapped_model,
            target_sample_rate,
            prompt_sample_rate,
            infer_type,
            output_dir
        )

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()

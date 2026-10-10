from pathlib import Path
from typing import Sequence, Dict, Union
import os
import json
import string
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import librosa
from tqdm import tqdm
from whisper_normalizer.english import EnglishTextNormalizer
from zhon.hanzi import punctuation
from jiwer import process_words as compute_measures
import argparse
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
english_normalizer = EnglishTextNormalizer()

def read_jsonl_to_mapping(jsonl_file: Union[str, Path], key_col: str, value_col: str, base_path=None) -> Dict[str, str]:
    mapping = {}
    with open(jsonl_file, 'r') as file:
        for line in file.readlines():
            data = json.loads(line.strip())
            key = data[key_col]
            value = data[value_col]
            if base_path:
                value = os.path.join(base_path, value)
            mapping[key] = value
    return mapping

def get_audio_duration(filepath):
    metadata = torchaudio.info(filepath)
    num_frames = metadata.num_frames
    sample_rate = metadata.sample_rate
    duration = num_frames / sample_rate
    return duration

def get_generated_audio_mapping(directory: str, exts: Sequence = [".wav"]):
    audio_mapping = {}
    for fpath in Path(directory).iterdir():
        if fpath.suffix.lower() in exts:
            audio_id = fpath.stem
            audio_mapping[audio_id] = fpath.as_posix()
    return audio_mapping

def load_speaker_model_titanet():
    save_restore_connector = SaveRestoreConnector()
    nemo_dir = os.environ["CAST_NEMO_DIR"]
    speaker_model = EncDecSpeakerLabelModel.restore_from(
        restore_path=os.path.join(nemo_dir, "speakerverification_en_titanet_large.nemo"),
        save_restore_connector=save_restore_connector
    )
    return speaker_model

def load_speaker_model_ecapa_tdnn():
    from ecapa_tdnn import ECAPA_TDNN_SMALL
    wavlm_ckpt_path = os.environ["CAST_SPEAKER_CHECKPOINT"]
    speaker_model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
    state_dict = torch.load(wavlm_ckpt_path, weights_only=True, map_location=lambda storage, loc: storage)
    speaker_model.load_state_dict(state_dict["model"], strict=False)
    speaker_model.eval()
    speaker_model.to(DEVICE)
    return speaker_model

def load_utmos_model():
    model = torch.hub.load( "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    model = model.to(DEVICE)
    return model

def load_whisper_model():
    import whisper
    model = whisper.load_model("large-v3", device=DEVICE)
    return model

def asr_whisper(model, wav_path):
    result = model.transcribe(wav_path)
    pred = result['text'].strip()
    return pred

def load_nemo_asr_model():
    from nemo.collections.asr.models import EncDecRNNTBPEModel
    save_restore_connector = SaveRestoreConnector()
    nemo_dir = os.environ["CAST_NEMO_DIR"]
    model = EncDecRNNTBPEModel.restore_from(
        restore_path=os.path.join(nemo_dir, "stt_en_conformer_transducer_xlarge.nemo"),
        save_restore_connector=save_restore_connector
    )
    return model

def asr_nemo(model, wav_path):
    result = model.transcribe([wav_path], verbose=False)[0].text.strip()
    return result

def clean_for_wer(text, lang="en"):
    for x in punctuation + string.punctuation:
        text = text.replace(x, "")
    text = text.replace("  ", " ")
    if lang == "zh":
        text = " ".join([x for x in text])
    else:  # "en"
        text = english_normalizer(text)
        text = text.lower()
    return text

def evaluate_tts(
    audio_dir_or_jsonl: str,
    ref_transcript_path: str,
    ref_audio_path: str,
    output_path: str,
) -> None:
    print("Loading Whisper model for ASR...")
    whisper_model = load_whisper_model()
    print("Loading Nemo model for ASR...")
    nemo_asr_model = load_nemo_asr_model()
    print("Loading speaker embedding model: titanet...")
    speaker_model_titanet = load_speaker_model_titanet()
    print("Loading speaker embedding model: ecapa_tdnn...")
    speaker_model_ecapa = load_speaker_model_ecapa_tdnn()
    print("Loading UTMOS model...")
    utmos_model = load_utmos_model()

    print("Building generated audio map...")
    if Path(audio_dir_or_jsonl).is_dir():
        aid_to_gen_audio = get_generated_audio_mapping(audio_dir_or_jsonl)
    elif audio_dir_or_jsonl.endswith(".jsonl"):
        aid_to_gen_audio = read_jsonl_to_mapping(audio_dir_or_jsonl, "audio_id", "audio")
    print(f"Found {len(aid_to_gen_audio)} audio files")

    print("Building reference transcript map...")
    with open(ref_transcript_path, 'r') as f:
        aid_to_transcription = json.load(f)

    print("Building reference audio map...")
    if ref_audio_path.endswith(".json"):
        with open(ref_audio_path, 'r') as f:
            aid_to_ref_audio = json.load(f)
    elif ref_audio_path.endswith(".jsonl"):
        aid_to_ref_audio = read_jsonl_to_mapping(ref_audio_path, "audio_id", "audio")

    total_word_errors_whisper = 0.0
    total_word_errors_nemo = 0.0
    total_words = 0
    sim_titanets = []
    sim_ecapas = []
    utmos_scores = []
    results = []

    lang = "en" 

    for audio_id in tqdm(aid_to_gen_audio, desc="Evaluating audio files"):
        gen_audio = aid_to_gen_audio[audio_id]
        gen_duration = get_audio_duration(gen_audio)
        if gen_duration < 0.5:
            print(f"Skipping {gen_audio}, duration less than 0.5s")
            continue

        transcription = aid_to_transcription[audio_id]


        pred_text_whisper = asr_whisper(whisper_model, gen_audio)
        pred_text_nemo = asr_nemo(nemo_asr_model, gen_audio)

        ref_wer = clean_for_wer(transcription, lang)
        hypo_whisper_wer = clean_for_wer(pred_text_whisper, lang)
        hypo_nemo_wer = clean_for_wer(pred_text_nemo, lang)

        ref_len = len(ref_wer.split())
        wer_whisper = compute_measures(ref_wer, hypo_whisper_wer).wer
        wer_nemo = compute_measures(ref_wer, hypo_nemo_wer).wer

        total_word_errors_whisper += wer_whisper * ref_len
        total_word_errors_nemo += wer_nemo * ref_len
        total_words += ref_len

        pred_emb_titanet = speaker_model_titanet.get_embedding(gen_audio)
        ref_emb_titanet = speaker_model_titanet.get_embedding(aid_to_ref_audio[audio_id])
        if isinstance(pred_emb_titanet, tuple): pred_emb_titanet = pred_emb_titanet[0]
        if isinstance(ref_emb_titanet, tuple): ref_emb_titanet = ref_emb_titanet[0]
        sim_titanet = F.cosine_similarity(pred_emb_titanet, ref_emb_titanet)[0].item()
        sim_titanets.append(sim_titanet)

        with torch.no_grad():
            wav1, sr1 = torchaudio.load(gen_audio)
            wav2, sr2 = torchaudio.load(aid_to_ref_audio[audio_id])
            if sr1 != 16000:
                wav1 = torchaudio.functional.resample(wav1, orig_freq=sr1, new_freq=16000)
            wav1 = wav1.to(DEVICE)
            if sr2 != 16000:
                wav2 = torchaudio.functional.resample(wav2, orig_freq=sr2, new_freq=16000)
            wav2 = wav2.to(DEVICE)
            pred_emb_ecapa = speaker_model_ecapa(wav1)
            ref_emb_ecapa = speaker_model_ecapa(wav2)
        if isinstance(pred_emb_ecapa, tuple): pred_emb_ecapa = pred_emb_ecapa[0]
        if isinstance(ref_emb_ecapa, tuple): ref_emb_ecapa = ref_emb_ecapa[0]
        sim_ecapa = F.cosine_similarity(pred_emb_ecapa, ref_emb_ecapa)[0].item()
        sim_ecapas.append(sim_ecapa)

        wav, sr = librosa.load(gen_audio, mono=True, sr=None)
        wav = torch.as_tensor(wav).to(DEVICE).unsqueeze(0)
        utmos_score = utmos_model(wav, sr).item()
        utmos_scores.append(utmos_score)

        line = {
            "audio_id": audio_id,
            "audio": gen_audio,
            "ref_audio": aid_to_ref_audio[audio_id],
            "reference": transcription,
            "prediction_whisper": pred_text_whisper,
            "prediction_nemo": pred_text_nemo,
            "WER_whisper": round(wer_whisper, 3),
            "WER_nemo": round(wer_nemo, 3),
            "SIM_titanet": round(sim_titanet, 3),
            "SIM_ecapa_tdnn": round(sim_ecapa, 3),
            "UTMOS": round(utmos_score, 3),
        }
        results.append(line)

    avg_wer_whisper = total_word_errors_whisper / total_words if total_words > 0 else 0.0
    avg_wer_nemo = total_word_errors_nemo / total_words if total_words > 0 else 0.0
    avg_sim_titanet = np.mean(sim_titanets) if sim_titanets else 0.0
    avg_sim_ecapa = np.mean(sim_ecapas) if sim_ecapas else 0.0
    avg_utmos = np.mean(utmos_scores) if utmos_scores else 0.0

    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, 'w') as f:
        for r in results:
            json.dump(r, f)
            f.write('\n')
        json.dump({
            "average_wer_whisper": avg_wer_whisper,
            "average_wer_nemo": avg_wer_nemo,
            "average_cosine_similarity_titanet": avg_sim_titanet,
            "average_cosine_similarity_ecapa_tdnn": avg_sim_ecapa,
            "average_utmos": avg_utmos,
        }, f)
        f.write('\n')

    print(f"Evaluation done: {len(results)} samples")
    print(f"Average WER(Whisper): {avg_wer_whisper}")
    print(f"Average WER(Nemo): {avg_wer_nemo}")
    print(f"Average Cosine similarity (titanet): {avg_sim_titanet}")
    print(f"Average Cosine similarity (ecapa_tdnn): {avg_sim_ecapa}")
    print(f"Average UTMOS: {avg_utmos}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--audio_dir_or_jsonl',
        type=str,
        required=True,
        help=
        'Directory or JSONL filecontaining TTS-generated audio files (recursive search)'
    )
    parser.add_argument(
        '--ref_transcript_path',
        type=str,
        default='',
        help='Path to reference transcript JSON file'
    )
    parser.add_argument(
        '--ref_audio_path',
        type=str,
        default='',
        help='Path to reference audio JSON file'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='',
        help='Output path for evaluation results'
    )

    args = parser.parse_args()

    evaluate_tts(
        args.audio_dir_or_jsonl,
        args.ref_transcript_path,
        args.ref_audio_path,
        args.output_path,
    )

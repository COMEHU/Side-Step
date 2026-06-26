import torch
import sys
import argparse
from acestep.training_v2.model_loader import load_decoder_for_training
# Ajusta según donde se encuentre tu logic de generación
from acestep.training_v2.inference_utils import generate_sample 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--out", type=str)
    args = parser.parse_args()

    # Cargar el modelo con el LoRA recién guardado
    model = load_decoder_for_training(checkpoint_dir=args.ckpt, variant="turbo", device="cuda", precision="bf16")
    
    # Generación fija
    prompt = "maximalist reggaeton, hyperpop textures"
    lyrics = "En full la señal"
    
    # Aquí invocas tu función de generación interna de ACE-Step
    audio = generate_sample(model, prompt, lyrics, seed=42)
    
    # Guardar
    import soundfile as sf
    sf.write(f"{args.out}/epoch_test.flac", audio, 44100)

if __name__ == "__main__":
    main()
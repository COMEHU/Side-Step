import sys
import argparse
import torch
import soundfile as sf
from pathlib import Path
# Importamos el cargador de modelos de ACE-Step
from acestep.training_v2.model_loader import load_decoder_for_training

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    # 1. Cargar modelo base + LoRA (el trainer usa "turbo" en tu config)
    # Ajustamos la carga para que sea ligera
    try:
        model = load_decoder_for_training(
            checkpoint_dir=args.ckpt, 
            variant="turbo", 
            device="cuda", 
            precision="bf16"
        )
        
        # 2. Generación rápida (Configuración fija)
        prompt = "a reggaeton song with electronic synths and male vocals"
        lyrics = "Medicate my soul with your lips, I need\nI need to forget that my love's at peace\nWith another man who respects and treats\nTreats her with a trust, and honesty\n\nMake me come alive, I just want to breathe\nI've been up for days, I just want to sleep"
        duration = 50
        
        # Inferencia simplificada usando los métodos del modelo cargado
        # (Asegúrate de que 'generate' esté expuesto en tu modelo cargado)
        audio_tensor = model.generate(prompt=prompt, lyrics=lyrics, duration=duration, seed=42, steps=8)
        
        # 3. Guardar
        out_path = Path(args.out) / f"val_{Path(args.ckpt).name}.flac"
        sf.write(out_path, audio_tensor.cpu().numpy(), 44100)
        
    except Exception as e:
        print(f"Error en validación: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

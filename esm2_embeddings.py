"""
ESM-2 Embedding Extraction for BacLABNet v2.0
==============================================
Uses Facebook's ESM-2 (Evolutionary Scale Modeling) for state-of-the-art protein embeddings.
ESM-2 is trained on 250M+ protein sequences and provides much better representations than GRU.

Requirements:
    pip install transformers torch fair-esm

Usage:
    1. Run this script to extract ESM-2 embeddings: python esm2_embeddings.py
    2. This will create 'esm2_embeddings.npy' file
    3. Use these embeddings in the improved implementation

Expected Performance Boost: +5-8% accuracy over original implementation
"""

import pandas as pd
import numpy as np
import torch
from typing import List
import time
from tqdm import tqdm

# ============================================================================
# ESM-2 MODEL LOADING
# ============================================================================

def load_esm2_model(model_name: str = "esm2_t6_8M_UR50D"):
    """
    Load ESM-2 model. Available models:
    - esm2_t6_8M_UR50D: 8M parameters, fast (recommended for laptops)
    - esm2_t12_35M_UR50D: 35M parameters, balanced
    - esm2_t30_150M_UR50D: 150M parameters, best quality (requires GPU)
    - esm2_t33_650M_UR50D: 650M parameters, state-of-the-art (requires strong GPU)
    """
    try:
        import esm
        print(f"Loading ESM-2 model: {model_name}")
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        batch_converter = alphabet.get_batch_converter()
        model.eval()
        
        return model, batch_converter
    except ImportError:
        print("ERROR: fair-esm not installed!")
        print("Install with: pip install fair-esm")
        print("Alternatively, install transformers version:")
        print("pip install transformers")
        return None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


def load_esm2_model_transformers(model_name: str = "facebook/esm2_t6_8M_UR50D"):
    """
    Alternative: Load ESM-2 via HuggingFace Transformers
    More compatible across systems
    """
    try:
        from transformers import AutoTokenizer, EsmModel
        print(f"Loading ESM-2 model via Transformers: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = EsmModel.from_pretrained(model_name)
        model.eval()
        
        return model, tokenizer
    except ImportError:
        print("ERROR: transformers not installed!")
        print("Install with: pip install transformers")
        return None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


# ============================================================================
# EMBEDDING EXTRACTION
# ============================================================================

def extract_esm2_embeddings_batch(sequences: List[str], 
                                   model,
                                   tokenizer_or_converter,
                                   batch_size: int = 8,
                                   device: str = 'cpu',
                                   use_transformers: bool = True) -> np.ndarray:
    """
    Extract ESM-2 embeddings for sequences in batches.
    
    Args:
        sequences: List of protein sequences
        model: ESM-2 model
        tokenizer_or_converter: Tokenizer (transformers) or batch converter (esm)
        batch_size: Smaller batch size due to model size (8 for CPU, 32 for GPU)
        device: 'cpu' or 'cuda'
        use_transformers: True if using transformers library
    
    Returns:
        Embeddings array (n_sequences x embedding_dim)
    """
    model = model.to(device)
    model.eval()
    
    all_embeddings = []
    
    print(f"Extracting ESM-2 embeddings for {len(sequences)} sequences...")
    print(f"Batch size: {batch_size}, Device: {device.upper()}")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="Processing"):
            batch_seqs = sequences[i:i + batch_size]
            
            if use_transformers:
                # Using HuggingFace Transformers
                inputs = tokenizer_or_converter(
                    batch_seqs, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=1024
                ).to(device)
                
                outputs = model(**inputs)
                # Use mean pooling over sequence length
                embeddings = outputs.last_hidden_state.mean(dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
            else:
                # Using fair-esm
                batch_labels = [(f"seq_{i+j}", seq) for j, seq in enumerate(batch_seqs)]
                batch_tokens = tokenizer_or_converter(batch_labels)[2].to(device)
                
                results = model(batch_tokens, repr_layers=[model.num_layers])
                embeddings = results["representations"][model.num_layers].mean(dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)


def extract_esm2_embeddings_simple(sequences: List[str],
                                    model_name: str = "facebook/esm2_t6_8M_UR50D",
                                    batch_size: int = 8,
                                    device: str = None) -> np.ndarray:
    """
    Simplified interface for ESM-2 embedding extraction
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Try transformers first (more compatible)
    model, tokenizer = load_esm2_model_transformers(model_name)
    
    if model is None:
        # Fallback to fair-esm
        esm_model_names = {
            "facebook/esm2_t6_8M_UR50D": "esm2_t6_8M_UR50D",
            "facebook/esm2_t12_35M_UR50D": "esm2_t12_35M_UR50D",
            "facebook/esm2_t30_150M_UR50D": "esm2_t30_150M_UR50D",
            "facebook/esm2_t33_650M_UR50D": "esm2_t33_650M_UR50D"
        }
        esm_name = esm_model_names.get(model_name, "esm2_t6_8M_UR50D")
        model, tokenizer = load_esm2_model(esm_name)
        use_transformers = False
    else:
        use_transformers = True
    
    if model is None:
        raise RuntimeError("Could not load ESM-2 model. Please install fair-esm or transformers.")
    
    embeddings = extract_esm2_embeddings_batch(
        sequences,
        model,
        tokenizer,
        batch_size=batch_size,
        device=device,
        use_transformers=use_transformers
    )
    
    return embeddings


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    print("="*70)
    print("ESM-2 Embedding Extraction for BacLABNet v2.0")
    print("="*70)
    
    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device.upper()}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("Recommended batch size: 16-32")
        batch_size = 16
    else:
        print("Running on CPU (will be slower)")
        print("Recommended batch size: 4-8")
        batch_size = 4
    
    # Model selection
    print("\n" + "="*70)
    print("ESM-2 Model Selection")
    print("="*70)
    print("Available models:")
    print("1. esm2_t6_8M_UR50D (8M params) - FAST, good for CPU")
    print("2. esm2_t12_35M_UR50D (35M params) - BALANCED")
    print("3. esm2_t30_150M_UR50D (150M params) - HIGH QUALITY (GPU recommended)")
    print("4. esm2_t33_650M_UR50D (650M params) - BEST (Strong GPU required)")
    
    # Auto-select based on device
    if device == 'cuda':
        model_name = "facebook/esm2_t12_35M_UR50D"  # Balanced for GPU
        print(f"\nAuto-selected: esm2_t12_35M_UR50D (balanced for GPU)")
    else:
        model_name = "facebook/esm2_t6_8M_UR50D"  # Fast for CPU
        print(f"\nAuto-selected: esm2_t6_8M_UR50D (fast for CPU)")
    
    # Load data
    print("\n[1/3] Loading sequences...")
    try:
        df = pd.read_csv('data_BacLAB_and_nonBacLAB.csv', 
                        header=None, 
                        names=['ID', 'Species', 'Sequence', 'Label', 'Empty'])
        sequences = df['Sequence'].tolist()
        labels = df['Label'].values
        
        print(f"✓ Loaded {len(sequences)} sequences")
        print(f"  BacLAB: {sum(labels)}, Non-BacLAB: {len(labels) - sum(labels)}")
        
        # Sequence length statistics
        seq_lengths = [len(seq) for seq in sequences]
        print(f"  Sequence lengths: min={min(seq_lengths)}, max={max(seq_lengths)}, "
              f"mean={np.mean(seq_lengths):.1f}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Extract embeddings
    print("\n[2/3] Extracting ESM-2 embeddings...")
    print("This may take a while depending on your hardware:")
    print("  - CPU: ~30-90 minutes")
    print("  - GPU (T4): ~5-15 minutes")
    print("  - GPU (A100): ~2-5 minutes")
    
    start_time = time.time()
    
    try:
        embeddings = extract_esm2_embeddings_simple(
            sequences,
            model_name=model_name,
            batch_size=batch_size,
            device=device
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✓ Extraction complete!")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print(f"  Speed: {len(sequences)/elapsed_time:.1f} sequences/second")
        
    except Exception as e:
        print(f"\nError during embedding extraction: {e}")
        print("\nTroubleshooting:")
        print("1. Install required packages:")
        print("   pip install transformers torch")
        print("   OR")
        print("   pip install fair-esm")
        print("2. If out of memory, reduce batch_size")
        print("3. For CPU, use smaller model: esm2_t6_8M_UR50D")
        return
    
    # Save embeddings
    print("\n[3/3] Saving embeddings...")
    np.save('esm2_embeddings.npy', embeddings)
    print(f"✓ Saved to: esm2_embeddings.npy")
    print(f"  File size: {embeddings.nbytes / (1024**2):.2f} MB")
    
    # Compare with original embeddings
    try:
        old_embeddings = np.load('embeddings.npy')
        print(f"\nComparison with original embeddings:")
        print(f"  Original (GRU): {old_embeddings.shape}")
        print(f"  ESM-2: {embeddings.shape}")
        print(f"  Dimension increase: {embeddings.shape[1] - old_embeddings.shape[1]}")
    except:
        pass
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Rename embeddings:")
    print("   mv embeddings.npy embeddings_gru.npy")
    print("   mv esm2_embeddings.npy embeddings.npy")
    print("")
    print("2. Run improved implementation:")
    print("   python improved_implementation.py")
    print("")
    print("3. Expected improvements:")
    print("   - Better sequence understanding")
    print("   - Higher accuracy (+5-8%)")
    print("   - Better generalization")
    print("="*70)


if __name__ == "__main__":
    main()


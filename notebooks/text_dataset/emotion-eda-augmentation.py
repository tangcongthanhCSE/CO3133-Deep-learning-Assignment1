"""
Augmentation

Nguyên tắc:
  1. CHỈ augment surprise (class cực thiểu số 3.6%, 572 mẫu)
  2. love (8.2%) để nguyên — đủ lớn cho DistilBERT pretrained
  3. Multiplier x1 cho surprise — nâng từ 572 → ~1144, vẫn là minority
  4. Giữ phân phối train gần giống val/test nhất có thể
  5. Loại bỏ duplicate sau augmentation
"""

import numpy as np
import pandas as pd
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ====== Load data ======
df_train = pd.read_csv('emotion_train.csv')
df_val = pd.read_csv('emotion_val.csv')

LABEL_NAMES = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

print("=== Original distribution ===")
orig_counts = df_train['label'].value_counts().sort_index()
for i, name in enumerate(LABEL_NAMES):
    print(f"  {name:10s}: {orig_counts[i]:5d} ({100*orig_counts[i]/len(df_train):.1f}%)")
print(f"  Total: {len(df_train)}")
print(f"  Imbalance ratio: {orig_counts.max()}/{orig_counts.min()} = {orig_counts.max()/orig_counts.min():.1f}x")


# ====== Augmentation functions (giữ nguyên — đã tốt) ======
def augment_word_dropout(text, drop_prob=0.15):
    words = text.split()
    if len(words) <= 4:
        return text
    result = [w for w in words if random.random() > drop_prob]
    if len(result) < 3:
        return text
    return ' '.join(result)


def augment_word_swap(text, n_swaps=1):
    words = text.split()
    if len(words) <= 3:
        return text
    words = words.copy()
    for _ in range(n_swaps):
        idx = random.randint(0, len(words) - 2)
        words[idx], words[idx + 1] = words[idx + 1], words[idx]
    return ' '.join(words)


def augment_random_insert(text):
    words = text.split()
    if len(words) <= 3:
        return text
    words = words.copy()
    word_to_insert = random.choice(words)
    insert_pos = random.randint(0, len(words))
    words.insert(insert_pos, word_to_insert)
    return ' '.join(words)


def augment_combined(text):
    technique = random.choice(['dropout', 'swap', 'insert'])
    if technique == 'dropout':
        return augment_word_dropout(text)
    elif technique == 'swap':
        return augment_word_swap(text)
    else:
        return augment_random_insert(text)


# ====== Chiến lược augmentation v3 — tối thiểu ======
# CHỈ augment surprise (3.6%, 572 mẫu) — class cực thiểu số duy nhất
# Tất cả class khác để nguyên — DistilBERT pretrained đủ capacity học từ ít mẫu
# Imbalance còn lại xử lý bằng sqrt class weights nhẹ trong training

aug_multiplier = {
    0: 0,  # sadness  29.2% — majority
    1: 0,  # joy      33.5% — majority
    2: 0,  # love      8.2% — để nguyên, DistilBERT đủ học
    3: 0,  # anger    13.5% — OK
    4: 0,  # fear     12.1% — OK
    5: 1,  # surprise  3.6% → ~6.7% — augment x1 (572 → 1144)
}

print(f"\n=== Augmentation strategy v3 (minimal) ===")
print(f"Only augment: surprise (x1)")
print(f"Leave alone: sadness, joy, love, anger, fear")
print()

for i, name in enumerate(LABEL_NAMES):
    n = orig_counts[i]
    expected = n + n * aug_multiplier[i]
    print(f"  {name:10s} | orig={n:5d} | x{aug_multiplier[i]} | expected={expected:5d}")


# ====== Thực hiện augmentation ======
random.seed(SEED)
augmented_rows = []

for _, row in df_train.iterrows():
    label = row['label']
    n_aug = aug_multiplier[label]
    for _ in range(n_aug):
        aug_text = augment_combined(row['text'])
        augmented_rows.append({'text': aug_text, 'label': label})

df_original = df_train[['text', 'label']].copy()
df_aug_new = pd.DataFrame(augmented_rows)
df_combined = pd.concat([df_original, df_aug_new], ignore_index=True)

# Loại bỏ duplicate texts
before_dedup = len(df_combined)
df_combined = df_combined.drop_duplicates(subset='text', keep='first')
after_dedup = len(df_combined)
print(f"\nRemoved {before_dedup - after_dedup} duplicate texts")

# Shuffle
df_combined = df_combined.sample(frac=1, random_state=SEED).reset_index(drop=True)


# ====== So sánh phân phối ======
print("\n=== Final distribution ===")
final_counts = df_combined['label'].value_counts().sort_index()
val_counts = df_val['label'].value_counts().sort_index()

print(f"{'Label':<12} {'Orig':>6} {'Orig%':>7} {'New':>6} {'New%':>7} {'Val%':>7}")
for i, name in enumerate(LABEL_NAMES):
    o_pct = 100 * orig_counts[i] / len(df_train)
    n_pct = 100 * final_counts[i] / len(df_combined)
    v_pct = 100 * val_counts[i] / len(df_val)
    print(f"  {name:<10} {orig_counts[i]:>6} {o_pct:>6.1f}% {final_counts[i]:>6} {n_pct:>6.1f}% {v_pct:>6.1f}%")

print(f"\n  Total: {len(df_train)} -> {len(df_combined)} (+{len(df_combined)-len(df_train)})")
print(f"  Imbalance ratio: {final_counts.max()}/{final_counts.min()} = {final_counts.max()/final_counts.min():.1f}x")


# ====== Export ======
df_combined.to_csv('emotion_train_augmented.csv', index=False)
print(f"\nSaved: emotion_train_augmented.csv ({len(df_combined):,} rows)")

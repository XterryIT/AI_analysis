import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import sys

# ==========================================
# НАСТРОЙКИ
# ==========================================
INPUT_FILE = 'CSVs/Total_CSVs/l2-benign.csv'          # Твой исходный файл
OUTPUT_FILE = 'CSVs/Total_CSVs/l2-benign-smote.csv'   # Куда сохранить результат
TARGET_ROWS = 100000                  # Сколько строк хотим в итоге
K_NEIGHBORS = 5                       # Сколько соседей использовать для генерации

def smart_smote_generation(df, n_samples_to_generate, k=5):
    """
    Generates synthetic data using SMOTE logic for a single class.
    """
    # FIX: Fill missing values (NaN) with 0 before processing
    # NearestNeighbors cannot handle NaN values.
    df.fillna(0, inplace=True)

    # Select only numeric columns for mathematical operations
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Create the matrix for the algorithm
    X = df[numeric_cols].values
    
    print(f"   -> Training NearestNeighbors on {len(X)} rows...")
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    new_rows = []
    
    print(f"   -> Generating {n_samples_to_generate} new rows...")
    
    for _ in range(n_samples_to_generate):
        # 1. Pick a random parent row
        idx = np.random.randint(len(X))
        sample = X[idx]
        
        # 2. Pick a random neighbor
        neighbor_idx = indices[idx][np.random.randint(1, k)] # 0 is the point itself, so skip 0
        neighbor = X[neighbor_idx]
        
        # 3. Create a new point between them (Interpolation)
        ratio = np.random.random()
        new_sample = sample + ratio * (neighbor - sample)
        
        new_rows.append(new_sample)
        
    # Convert back to DataFrame
    new_df = pd.DataFrame(new_rows, columns=numeric_cols)
    
    # If there were non-numeric columns (like Label), copy them from the original
    for col in df.columns:
        if col not in numeric_cols:
            val = df[col].iloc[0]
            new_df[col] = val
            
    return new_df

def main():
    print(f"### ЗАПУСК SMOTE ГЕНЕРАЦИИ ###")
    
    # 1. Загрузка
    print(f"[1] Читаем '{INPUT_FILE}'...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("ОШИБКА: Файл не найден.")
        return

    current_count = len(df)
    print(f"    Текущее кол-во строк: {current_count}")
    
    if current_count >= TARGET_ROWS:
        print("    В файле уже достаточно строк. Генерация не нужна.")
        return

    to_generate = TARGET_ROWS - current_count
    print(f"    Нужно сгенерировать: {to_generate}")

    # 2. Генерация
    print("[2] Запускаем алгоритм...")
    try:
        synthetic_df = smart_smote_generation(df, to_generate, k=K_NEIGHBORS)
        
        # Округляем значения, если нужно (например, порты не могут быть дробными)
        # Для простоты можно округлить всё до 6 знаков или оставить float
        # Если есть колонки, которые обязаны быть int (порты), лучше их округлить:
        # synthetic_df = synthetic_df.round(0) 
        
    except Exception as e:
        print(f"ОШИБКА при генерации: {e}")
        return

    # 3. Объединение
    print("[3] Объединяем оригинальные и новые данные...")
    final_df = pd.concat([df, synthetic_df], ignore_index=True)
    
    # Перемешиваем
    final_df = final_df.sample(frac=1).reset_index(drop=True)

    # 4. Сохранение
    print(f"[4] Сохраняем в '{OUTPUT_FILE}'...")
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\nУСПЕХ! Новый файл содержит {len(final_df)} строк.")
    print("Теперь можно использовать этот файл для обучения модели.")

if __name__ == "__main__":
    main()
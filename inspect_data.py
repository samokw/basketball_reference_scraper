"""
Inspect scraped Basketball Reference data to understand structure.
"""
import os
import pandas as pd


def inspect_scraped_data(base_path="data/"):
    """
    Scan data/ folder and report what CSVs exist per year.
    """
    years = sorted([d for d in os.listdir(base_path) if d.isdigit()])
    
    print("="*60)
    print("SCRAPED DATA INSPECTION")
    print("="*60)
    
    for year in years:
        year_path = os.path.join(base_path, year)
        if not os.path.isdir(year_path):
            continue
            
        files = [f for f in os.listdir(year_path) if f.endswith('.csv')]
        print(f"\n{year} ({len(files)} files):")
        
        for f in sorted(files):
            file_path = os.path.join(year_path, f)
            try:
                df = pd.read_csv(file_path, nrows=5)
                print(f"  [OK] {f}")
                print(f"    Shape: {df.shape[0]} rows (showing first 5), {df.shape[1]} cols")
                print(f"    Columns: {list(df.columns[:8])}{'...' if len(df.columns) > 8 else ''}")
            except Exception as e:
                print(f"  [ERROR] {f} - Error: {e}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    inspect_scraped_data()


import polars as pl

class DataCleaner:
    """
    DataCleaner performs idempotent data cleaning using Polars LazyFrame.
    Rules: 
      1. Trim column headers 
      2. Deduplicate rows
      3. Type coercion to best possible types
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df_lazy = pl.scan_csv(file_path)

    def clean(self) -> pl.DataFrame:
        # Step 1: Normalize headers (trim spaces)
        cleaned_columns = [col.strip() for col in self.df_lazy.columns]
        df_lazy = self.df_lazy.rename(
            {old: new for old, new in zip(self.df_lazy.columns, cleaned_columns)}
        )

        # Step 2: Deduplicate
        df_lazy = df_lazy.unique()

        # Step 3: Type coercion
        df_lazy = df_lazy.collect().cast(pl.infer_schema(df_lazy.collect()))

        return df_lazy

    def save(self, output_path: str):
        df_cleaned = self.clean()
        df_cleaned.write_csv(output_path)
        return output_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python data_cleaner.py <input_csv> <output_csv>")
    else:
        cleaner = DataCleaner(sys.argv[1])
        out_path = cleaner.save(sys.argv[2])
        print(f"✅ Cleaned data saved to {out_path}")

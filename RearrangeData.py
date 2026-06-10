import pandas as pd

# Read the entire CSV file as data rows (no header interpretation) using UTF-8 with BOM handling
df = pd.read_csv('icelandic_sentiment_v1.2.csv', encoding='utf-8-sig', header=None)

# Separate the original header row (first row) exactly as-is
header_row = df.iloc[0:1]

# Separate the data rows (everything after the header)
data_rows = df.iloc[1:]

# Shuffle only the data rows reproducibly (fixed seed via random_state)
shuffled_data = data_rows.sample(frac=1, random_state=42).reset_index(drop=True)

# Reassemble: original header + shuffled data rows
shuffled_df = pd.concat([header_row, shuffled_data]).reset_index(drop=True)

# Write the result to a new file, preserving exact row content and UTF-8 with BOM
shuffled_df.to_csv(
    'icelandic_sentiment_v1.2_shuffled.csv',
    index=False,
    header=False,  # Do not write a column-name header; the first row is already the original header
    encoding='utf-8-sig'
)
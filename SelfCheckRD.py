import pandas as pd

orig = pd.read_csv('icelandic_sentiment_v1.2.csv', encoding='utf-8-sig', header=None)
shuf = pd.read_csv('icelandic_sentiment_v1.2_shuffled.csv', encoding='utf-8-sig', header=None)

print(f"Row count match: {len(orig) == len(shuf)}")                  # Should be True
print(f"Header identical: {orig.iloc[0].equals(shuf.iloc[0])}")     # Should be True
print("Sample Icelandic rows from shuffled file:")
print(shuf.iloc[1:4])                                               # Inspect a few data rows for intact ð, þ, æ, etc.
print(f"Data rows shuffled (first original data row now at position): {(shuf.iloc[1:] == orig.iloc[1:2]).all().all()}")  # Quick check that order changed
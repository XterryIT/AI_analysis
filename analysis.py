import pandas as pd

# setting up files
df_doh = pd.read_csv('CSVs/Total/l1-doh.csv')
df_non_doh = pd.read_csv('CSVs/Total/l1-nondoh.csv')
df_benign = pd.read_csv('CSVs/Total/l2-benign.csv')
df_malicious = pd.read_csv('CSVs/Total/l2-malicious.csv')


doh_temp = df_doh.head(20000).copy()
non_doh_temp = df_non_doh.head(20000).copy()
benign_temp = df_benign.head(20000).copy()
malicious_temp = df_malicious.head(20000).copy()

columns_to_drop = ['TimeStamp','SourceIP', 'DestinationIP' ,'PacketLengthVariance', 'PacketLengthStandardDeviation', 'PacketLengthMean', 'PacketLengthMedian', 'PacketLengthMode', 'PacketLengthSkewFromMedian', 'PacketLengthSkewFromMode', 'PacketLengthCoefficientofVariation', 'PacketTimeVariance', 'PacketTimeStandardDeviation', 'PacketTimeMean', 'PacketTimeMedian', 'PacketTimeMode', 'PacketTimeSkewFromMedian', 'PacketTimeSkewFromMode', 'PacketTimeCoefficientofVariation', 'ResponseTimeTimeVariance', 'ResponseTimeTimeStandardDeviation', 'ResponseTimeTimeMean', 'ResponseTimeTimeMedian', 'ResponseTimeTimeMode', 'ResponseTimeTimeSkewFromMedian', 'ResponseTimeTimeSkewFromMode', 'ResponseTimeTimeCoefficientofVariation']

# #zamieniamy Lable na 0 i 1 i 2

non_doh_temp['Label'] = non_doh_temp['Label'].replace('NonDoH', 0)
doh_temp['Label'] = doh_temp['Label'].replace('DoH', 1)
benign_temp['Label'] = benign_temp['Label'].replace('Benign', 1)
malicious_temp['Label'] = malicious_temp['Label'].replace('Malicious', 2)

merged_df = pd.concat([non_doh_temp, doh_temp, benign_temp, malicious_temp], ignore_index=True)

merged_df.drop(columns=columns_to_drop, inplace=True)
print(merged_df.shape[0])
print(merged_df.shape[1])
merged_df.to_csv('merged_sample.csv', index=False)

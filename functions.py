from sklearn.preprocessing import MinMaxScaler,StandardScaler
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def privacy_metrics(real, fake, data_percent=15):

    """
    Returns privacy metrics

    Inputs:
    1) real_path -> path to real data
    2) fake_path -> path to corresponding synthetic data
    3) data_percent -> percentage of data to be sampled from real and synthetic datasets for computing privacy metrics
    Outputs:
    1) List containing the 5th percentile distance to closest record (DCR) between real and synthetic as well as within real and synthetic datasets
    along with 5th percentile of nearest neighbour distance ratio (NNDR) between real and synthetic as well as within real and synthetic datasets

    """

    # Loading real and synthetic datasets and removing duplicates if any
#     real = pd.read_csv(real_path).drop_duplicates(keep=False)
#     fake = pd.read_csv(fake_path).drop_duplicates(keep=False)
    real = real.drop_duplicates(keep=False)
    fake = fake.drop_duplicates(keep=False)

    # Sampling smaller sets of real and synthetic data to reduce the time complexity of the evaluation
    real_sampled = real.sample(n=int(len(real)*(.01*data_percent)), random_state=42).to_numpy()
    fake_sampled = fake.sample(n=int(len(fake)*(.01*data_percent)), random_state=42).to_numpy()

    # Scaling real and synthetic data samples
    scalerR = StandardScaler()
    scalerR.fit(real_sampled)
    scalerF = StandardScaler()
    scalerF.fit(fake_sampled)
    df_real_scaled = scalerR.transform(real_sampled)
    df_fake_scaled = scalerF.transform(fake_sampled)

    # Computing pair-wise distances between real and synthetic 
    dist_rf = metrics.pairwise_distances(df_real_scaled, Y=df_fake_scaled, metric='minkowski', n_jobs=-1)
    # Computing pair-wise distances within real 
    dist_rr = metrics.pairwise_distances(df_real_scaled, Y=None, metric='minkowski', n_jobs=-1)
    # Computing pair-wise distances within synthetic
    dist_ff = metrics.pairwise_distances(df_fake_scaled, Y=None, metric='minkowski', n_jobs=-1) 

    # Removes distances of data points to themselves to avoid 0s within real and synthetic 
    rd_dist_rr = dist_rr[~np.eye(dist_rr.shape[0],dtype=bool)].reshape(dist_rr.shape[0],-1)
    rd_dist_ff = dist_ff[~np.eye(dist_ff.shape[0],dtype=bool)].reshape(dist_ff.shape[0],-1) 

    # Computing first and second smallest nearest neighbour distances between real and synthetic
    smallest_two_indexes_rf = [dist_rf[i].argsort()[:2] for i in range(len(dist_rf))]
    smallest_two_rf = [dist_rf[i][smallest_two_indexes_rf[i]] for i in range(len(dist_rf))]       
    # Computing first and second smallest nearest neighbour distances within real
    smallest_two_indexes_rr = [rd_dist_rr[i].argsort()[:2] for i in range(len(rd_dist_rr))]
    smallest_two_rr = [rd_dist_rr[i][smallest_two_indexes_rr[i]] for i in range(len(rd_dist_rr))]
    # Computing first and second smallest nearest neighbour distances within synthetic
    smallest_two_indexes_ff = [rd_dist_ff[i].argsort()[:2] for i in range(len(rd_dist_ff))]
    smallest_two_ff = [rd_dist_ff[i][smallest_two_indexes_ff[i]] for i in range(len(rd_dist_ff))]

    # Computing 5th percentiles for DCR and NNDR between and within real and synthetic datasets
    min_dist_rf = np.array([i[0] for i in smallest_two_rf])
    fifth_perc_rf = np.percentile(min_dist_rf,5)
    min_dist_rr = np.array([i[0] for i in smallest_two_rr])
    fifth_perc_rr = np.percentile(min_dist_rr,5)
    min_dist_ff = np.array([i[0] for i in smallest_two_ff])
    fifth_perc_ff = np.percentile(min_dist_ff,5)
    nn_ratio_rf = np.array([i[0]/i[1] for i in smallest_two_rf])
    nn_fifth_perc_rf = np.percentile(nn_ratio_rf,5)
    nn_ratio_rr = np.array([i[0]/i[1] for i in smallest_two_rr])
    nn_fifth_perc_rr = np.percentile(nn_ratio_rr,5)
    nn_ratio_ff = np.array([i[0]/i[1] for i in smallest_two_ff])
    nn_fifth_perc_ff = np.percentile(nn_ratio_ff,5)

    return np.array([fifth_perc_rf,fifth_perc_rr,fifth_perc_ff,nn_fifth_perc_rf,nn_fifth_perc_rr,nn_fifth_perc_ff]).reshape(1,6) 


def plot_missing_data_histogram(df):
    """
    Plots a bar chart showing the percentage of missing values per column.
    Only columns with missing data are shown.
    
    Args:
        df (pd.DataFrame): The DataFrame to analyze.
    """
    missing_percent = df.isnull().mean() * 100
    missing_percent = missing_percent[missing_percent > 0]

    if missing_percent.empty:
        print("✅ No missing values in the DataFrame.")
        return

    plt.figure(figsize=(15, 8))
    missing_percent.plot(kind='bar', color='skyblue')
    plt.title('Percentage of Missing Data per Column')
    plt.xlabel('Columns')
    plt.ylabel('Missing Data (%)')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=5)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


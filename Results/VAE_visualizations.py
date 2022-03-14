import seaborn as sns
import matplotlib.pyplot as plt

from Models.VAE_model.run_vae_model import zeros_list, ones_list


plt.figure(figsize=(18, 10))
sns.boxplot(data=zeros_list)

plt.figure(figsize=(18, 10))
sns.boxplot(data=ones_list)

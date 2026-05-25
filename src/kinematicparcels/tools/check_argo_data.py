import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Jacopo\OneDrive - CNR\ARGO\ACC\PR_PF_1900978.csv")
df = pd.read_csv(r"C:\Users\Jacopo\OneDrive - CNR\ARGO\ACC\PR_PF_1900042.csv")
df = pd.read_csv(r"C:\Users\Jacopo\OneDrive - CNR\ARGO\ACC\PR_PF_6903207.csv")
df["PRES_ADJUSTED (decibar)"].plot()
plt.show()

df.plot.scatter(x="LONGITUDE (degree_east)", y = 'LATITUDE (degree_north)', c = 'DATE (YYYY-MM-DDTHH:MI:SSZ)')
plt.show()

df['PRES_ADJUSTED (decibar)'].plot.hist()
plt.show()
import matplotlib.pyplot as plt
import pandas as pd
data = {'Risk':  ['1080', '0', '256', '904', '275', '305', '602', '461', '487', '502', '577', '544'],
        'Information Loss': ['0', '100000', '95847.94661446511', '5625.131238930451', '76246.69744631831', '39189.82462866574', '23287.675320933933', '82512.61788773359', '33981.85246758442', '43027.4234482801', '46858.79932428555', '62611.909455291774'] }
df = pd.DataFrame (data, columns = ['Risk','Information Loss'])
print(df)
df['Risk'] = df['Risk'].astype(float)
df['Information Loss'] = df['Information Loss'].astype(float)
plt.scatter(df['Risk'], df['Information Loss'])
plt.title('Risk vs Information Loss')
plt.xlabel('Risk')
plt.ylabel('Information Loss')
plt.show()
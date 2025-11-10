import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r'..\dados\dataset_corrigido.csv')


target = 'vegetacao_natural'

df_model = df.drop(columns=['id_municipio', 'soma_partes'], errors='ignore')

df_model = pd.get_dummies(df_model, drop_first=True)

X = df_model.drop(columns=[target])
y = df_model[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

df_final = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
df_final.to_csv(r'..\dados\dataset_pronto_para_modelagem.csv', index=False)

print("Dataset preparado e salvo em 'dados/dataset_pronto_para_modelagem.csv'")
